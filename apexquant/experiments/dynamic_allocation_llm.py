"""Data-free per-layer pass allocation on a language model.

Composes HIGGS's linearity theorem with our geometric decay law:

    predicted error  ~=  sum_l alpha_l * rho_l ** (2 * P_l)

The ``alpha_l`` come from noise insertion scored by KL against the unquantized
model on *randomly sampled tokens*, so no calibration corpus is touched at any
point; the ``rho_l`` are quadratures over the known post-rotation density, so
they are available before a weight is read. What remains is a multiple-choice
knapsack over the pass counts, solved exactly.

    uv run python experiments/dynamic_allocation_llm.py --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
    uv run python experiments/dynamic_allocation_llm.py --model microsoft/phi-1_5

The reference model must be resident to score KL, which is why this runs locally
for 1-3B models and moves to a pod for Gemma-3-12B (see ``serving_evals/``).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apexquant import (  # noqa: E402
    ApexQuantConfig,
    knapsack_pass_allocation,
    validated_pass_allocation,
    layer_damage_table,
    linear_error_coefficients,
    quantize_linear_modules,
    random_token_divergence,
)
from apexquant.codebook import apex_beta_levels  # noqa: E402
from apexquant.quantize.resolve_group_size import resolve_group_size  # noqa: E402


def stage_rho(bits: int, group_size: int) -> float:
    """rho for the scalar density-matched stage at this fan-in, a pure quadrature."""
    levels = apex_beta_levels(bits, group_size)
    u = torch.linspace(-1.0, 1.0, 40001, dtype=torch.float64)
    w = (1.0 - u * u).clamp_min(0.0) ** ((group_size - 3) / 2.0)
    w = w / w.sum()
    q = levels.double()[(u[:, None] - levels.double()[None, :]).abs().argmin(1)]
    return float((group_size * ((u - q) ** 2 * w).sum()).sqrt())


def decoder_linears(model: nn.Module, config: ApexQuantConfig) -> dict[str, nn.Linear]:
    """Every linear the quantizer will actually touch, excluding the LM head."""
    from apexquant.nn.linear_is_fusable import linear_is_fusable
    return {
        n: m for n, m in model.named_modules()
        if isinstance(m, nn.Linear)
        and linear_is_fusable(m.in_features, config, False)
        and "lm_head" not in n and "embed_out" not in n
    }


def wikitext_kl(model, reference, tokens, device) -> float:
    """Mean KL(reference || model) in nats per token over held-out chunks.

    ``reference`` is either a list of cached per-chunk log-probabilities or a
    resident module. The cache is faster and the module is far smaller: at a
    262k vocab and seqlen 8192 one cached chunk is 8.6 GB, so past a few billion
    parameters holding the reference itself is the cheaper of the two.
    """
    resident = isinstance(reference, nn.Module)
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(tokens.shape[0]):
            ids = tokens[i : i + 1].to(device)
            log_q = torch.log_softmax(model(ids).logits.float(), dim=-1)
            log_p = (torch.log_softmax(reference(ids).logits.float(), dim=-1)
                     if resident else reference[i].to(device))
            kl = (log_p.exp() * (log_p - log_q)).sum(-1)
            total += float(kl.sum())
            count += kl.numel()
            del log_p, log_q
    return total / count



def host_copy(model):
    """A CPU copy of ``model`` that never puts a second copy on the device.

    ``copy.deepcopy(model).cpu()`` allocates the clone beside the original and
    moves it afterwards, so it needs twice the model on one card and dies at 7B.
    Moving the original out first costs one round trip and nothing else; the
    original is returned to its device before this function does.
    """
    device = next(model.parameters()).device
    model.to("cpu")
    clone = copy.deepcopy(model)
    model.to(device)
    return clone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=512)
    ap.add_argument("--max-passes", type=int, default=4)
    ap.add_argument("--min-passes", type=int, default=1,
                    help="Floor per layer. Coefficients are fitted over the probe "
                         "levels; a layer left at P passes sits at t=rho**P, so a floor "
                         "of 1 at 2-bit extrapolates to t=0.34, outside the fit.")
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.03, 0.05, 0.07, 0.09, 0.11, 0.13])
    ap.add_argument("--probe-batch", type=int, default=2)
    ap.add_argument("--probe-seq", type=int, default=256)
    ap.add_argument("--eval-chunks", type=int, default=16)
    ap.add_argument("--eval-seqlen", type=int, default=2048)
    ap.add_argument("--reference-resident", action="store_true",
                    help="Keep the dense reference on the device and score KL by "
                         "running both models per chunk, instead of caching its "
                         "log-probabilities. Caching costs seqlen*vocab*4 bytes "
                         "per chunk, which is 8.6 GB/chunk at Gemma's vocab and "
                         "seqlen 8192; holding a second 12B model is cheaper.")
    ap.add_argument("--method", choices=("measured", "noise"), default="measured",
                    help="How to score each layer. 'measured' quantizes the layer "
                         "for real at every candidate pass count; 'noise' fits the "
                         "linearity-theorem coefficient from Gaussian noise and "
                         "extrapolates through rho**(2P).")
    ap.add_argument("--baseline", type=int, default=None,
                    help="Expansion point for the measured table. Unset linearizes "
                         "around the dense model, as the linearity theorem states; "
                         "an integer quantizes every layer to that depth first and "
                         "measures signed marginals against it, which is the regime "
                         "an allocator actually moves within.")
    ap.add_argument("--validate", action="store_true", default=True,
                    help="Score the allocation against uniform on held-out random "
                         "tokens and keep the winner.")
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    ap.add_argument("--validate-seed", type=int, default=1234,
                    help="Held-out probe seed. Must differ from the seed the "
                         "allocation was chosen against, or the gate certifies "
                         "its own overfitting.")
    ap.add_argument("--validate-margin", type=float, default=1.0)
    ap.add_argument("--validate-batch", type=int, default=16)
    ap.add_argument("--validate-seq", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device).eval()

    config = ApexQuantConfig(bits=args.bits, group_size=args.group_size,
                              passes=1, codebook="apex")
    layers = decoder_linears(model, config)
    sizes = {n: m.in_features * m.out_features for n, m in layers.items()}
    rho = {n: stage_rho(args.bits, resolve_group_size(m.in_features, args.group_size))
           for n, m in layers.items()}
    bits_per_pass = args.bits + 16.0 / args.group_size
    print(f"{len(layers)} decoder linears, {sum(sizes.values())/1e9:.3f}B params, "
          f"rho in [{min(rho.values()):.4f}, {max(rho.values()):.4f}], "
          f"{bits_per_pass:.4f} bits/pass", flush=True)

    vocab = model.config.vocab_size

    # --- data-free coefficients -------------------------------------------
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(0)
        ids = torch.randint(0, vocab, (args.probe_batch, args.probe_seq),
                            generator=g).to(device)
        ref_log = torch.log_softmax(model(ids).logits.float(), dim=-1)

    pass_counts = tuple(range(1, args.max_passes + 1))
    t0 = time.time()
    seen = [0]

    def report(name, alpha):
        seen[0] += 1
        if seen[0] % 25 == 0:
            print(f"  {seen[0]}/{len(layers)} layers, {time.time()-t0:.0f}s",
                  flush=True)

    probe = lambda m: random_token_divergence(  # noqa: E731
        ref_log, m, vocab, batch=args.probe_batch, seq_len=args.probe_seq, seed=0)

    damage = None
    if args.method == "measured":
        damage = layer_damage_table(
            model, probe, list(layers), config, pass_counts,
            baseline_passes=(None if args.baseline is None else args.baseline),
            progress=lambda n, row: report(n, 0.0),
        )
        # An alpha-shaped summary, so the reporting below is common to both
        # methods: what a layer costs at the shallowest tier it may be given.
        alphas = {n: damage[n][pass_counts[0]] for n in damage}
    else:
        alphas = linear_error_coefficients(
            model, probe, layers=list(layers), levels=tuple(args.levels),
            progress=report,
        )
    probe_seconds = time.time() - t0
    ranked = sorted(alphas.items(), key=lambda kv: -kv[1])
    print(f"coefficients in {probe_seconds:.0f}s; "
          f"top: {', '.join(f'{n}={a:.3g}' for n, a in ranked[:3])}", flush=True)
    print(f"  spread: max/median = "
          f"{ranked[0][1] / max(sorted(alphas.values())[len(alphas)//2], 1e-30):.1f}x",
          flush=True)

    with torch.no_grad():
        g2 = torch.Generator(device="cpu").manual_seed(args.validate_seed)
        ids2 = torch.randint(0, vocab, (args.validate_batch, args.validate_seq),
                             generator=g2).to(device)
        arm_ref_log = torch.log_softmax(model(ids2).logits.float(), dim=-1)

    # --- evaluation reference ---------------------------------------------
    from eval_protocol import get_test_tokens  # noqa: E402
    tokens = get_test_tokens("wikitext2", tok, seqlen=args.eval_seqlen)[: args.eval_chunks]
    if args.reference_resident:
        ref_chunks = model            # scored live, one chunk at a time
        dense = host_copy(model)
    else:
        with torch.no_grad():
            ref_chunks = [torch.log_softmax(model(tokens[i:i+1].to(device)).logits.float(),
                                            dim=-1).cpu()
                          for i in range(tokens.shape[0])]
        dense = host_copy(model)

    results = []
    r0 = list(rho.values())[0]
    print("probe fitted over t in [%.3f, %.3f]; a layer at P passes sits at t=rho**P = %s"
          % (min(args.levels), max(args.levels),
             ", ".join("P%d:%.3f" % (p, r0 ** p) for p in range(1, args.max_passes + 1))),
          flush=True)
    for uniform_passes in (2, 3):
        budget = uniform_passes * bits_per_pass
        alloc = knapsack_pass_allocation(alphas, sizes, rho, bit_budget=budget,
                                         bits_per_pass=bits_per_pass,
                                         max_passes=args.max_passes,
                                         min_passes=args.min_passes,
                                         damage=damage)
        used = sum(sizes[n] * alloc[n] for n in alloc) * bits_per_pass / sum(sizes.values())
        hist = {p: sum(1 for v in alloc.values() if v == p) for p in sorted(set(alloc.values()))}
        print(f"\nbudget {budget:.3f} bits/weight -> allocation uses {used:.3f}, "
              f"passes {hist}", flush=True)

        # Predicted gain over uniform, from the fit alone. This is the go/no-go
        # signal: the two models measured so far agree that whether allocation
        # beats uniform is governed by how far apart the coefficients are, not by
        # how well they rank. A spread near 1 means the optimum is uniform and any
        # ranking error moves you off it for nothing.
        if damage is not None:
            pred_alloc = sum(damage[n][alloc[n]] for n in alloc)
            pred_unif = sum(damage[n][uniform_passes] for n in alloc)
        else:
            pred_alloc = sum(alphas[n] * rho[n] ** (2 * alloc[n]) for n in alloc)
            pred_unif = sum(alphas[n] * rho[n] ** (2 * uniform_passes) for n in alloc)
        print(f"  predicted gain over uniform: {pred_unif / max(pred_alloc, 1e-30):.2f}x",
              flush=True)
        row = {"budget": budget, "uniform_passes": uniform_passes,
               "allocated_bits": used, "histogram": hist,
               "predicted_gain": pred_unif / max(pred_alloc, 1e-30)}

        uniform_map = {n: uniform_passes for n in alloc}

        def build(passes_map):
            arm = copy.deepcopy(dense).to(device).eval()
            quantize_linear_modules(
                arm, config,
                should_quantize=lambda name, m: name in layers,
                passes_for=lambda name, m: passes_map[name],
            )
            return arm

        # The gate: score both configurations end to end on random tokens the
        # allocation was not chosen against (a held-out seed), and keep the
        # winner. Two forward passes, still no calibration corpus, and it is what
        # makes the procedure never worse than uniform on any model.
        def score(passes_map):
            arm = build(passes_map)
            d = random_token_divergence(arm_ref_log, arm, vocab,
                                        batch=args.validate_batch,
                                        seq_len=args.validate_seq,
                                        seed=args.validate_seed)
            del arm
            torch.cuda.empty_cache()
            return d

        chosen = alloc
        if args.validate:
            chosen = validated_pass_allocation(alloc, uniform_map, score,
                                               margin=args.validate_margin)
            row["gate_kept_allocation"] = chosen != uniform_map
            print(f"  gate: {'kept allocation' if chosen != uniform_map else 'fell back to uniform'}",
                  flush=True)

        for label, passes_map in (("uniform", uniform_map), ("allocated", alloc),
                                  ("validated", chosen)):
            arm = build(passes_map)
            kl = wikitext_kl(arm, ref_chunks, tokens, device)
            print(f"  {label:<10} KL = {kl:.5f} nats", flush=True)
            row[label] = kl
            del arm
            torch.cuda.empty_cache()
        row["improvement"] = row["uniform"] / row["allocated"] if row["allocated"] else float("nan")
        row["validated_improvement"] = (row["uniform"] / row["validated"]
                                        if row["validated"] else float("nan"))
        print(f"  allocated {row['improvement']:.3f}x, validated {row['validated_improvement']:.3f}x",
              flush=True)
        results.append(row)

    out = {"model": args.model, "method": args.method, "min_passes": args.min_passes,
           "max_passes": args.max_passes,
           "bits": args.bits, "group_size": args.group_size,
           "probe_seconds": probe_seconds, "levels": args.levels,
           "eval": {"corpus": "wikitext2", "seqlen": args.eval_seqlen,
                    "chunks": int(tokens.shape[0])},
           "coefficients": alphas,
           "damage": damage, "results": results}
    path = (args.out or
            f"results/dynamic_alloc_{args.model.split('/')[-1]}_{args.method}"
            f"_min{args.min_passes}_base{args.baseline}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
