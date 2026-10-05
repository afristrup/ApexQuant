"""Does the linearity theorem actually predict *our* quantization damage?

The allocator is only as good as its premise: that a coefficient fitted from
Gaussian noise insertion predicts the damage a real quantizer does at the same
relative error. HIGGS state this as the theorem's key property -- the scaling
coefficients "do not depend on the quantization method used" -- and it is what
lets them fit with noise and deploy with a quantizer. It is also the step most
likely to fail for us, because our error is a structured residual rather than an
unbiased Gaussian perturbation.

So this measures it head on. For a sample of layers: quantize *only* that layer,
score the real KL it causes, and compare against the prediction
``alpha_l * rho ** (2P)`` from the noise-fitted coefficient. If the theorem
transfers, the two rank together.

    uv run python experiments/linearity_validation.py --model microsoft/phi-1_5
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apexquant import (  # noqa: E402
    ApexQuantConfig,
    linear_error_coefficients,
    quantize_linear_modules,
    random_token_divergence,
)
from apexquant.quantize.resolve_group_size import resolve_group_size  # noqa: E402
from dynamic_allocation_llm import decoder_linears, stage_rho, wikitext_kl  # noqa: E402


def spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=512)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--sample", type=int, default=24)
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.03, 0.05, 0.07, 0.09, 0.11, 0.13])
    ap.add_argument("--probe-batch", type=int, default=2)
    ap.add_argument("--probe-seq", type=int, default=256)
    ap.add_argument("--eval-chunks", type=int, default=4)
    ap.add_argument("--eval-seqlen", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True).to(device).eval()

    config = ApexQuantConfig(bits=args.bits, group_size=args.group_size,
                              passes=args.passes, codebook="apex")
    layers = decoder_linears(model, config)
    names = sorted(layers)
    step = max(1, len(names) // args.sample)
    sampled = names[::step][: args.sample]
    print(f"{len(names)} linears, probing {len(sampled)} of them at P={args.passes}",
          flush=True)

    vocab = model.config.vocab_size
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(0)
        ids = torch.randint(0, vocab, (args.probe_batch, args.probe_seq),
                            generator=g).to(device)
        ref_log = torch.log_softmax(model(ids).logits.float(), dim=-1)

    t0 = time.time()
    alphas = linear_error_coefficients(
        model,
        lambda m: random_token_divergence(ref_log, m, vocab, batch=args.probe_batch,
                                          seq_len=args.probe_seq, seed=0),
        layers=sampled, levels=tuple(args.levels))
    print(f"coefficients for {len(sampled)} layers in {time.time()-t0:.0f}s", flush=True)

    from eval_protocol import get_test_tokens  # noqa: E402
    tokens = get_test_tokens("wikitext2", tok, seqlen=args.eval_seqlen)[: args.eval_chunks]
    with torch.no_grad():
        ref_chunks = [torch.log_softmax(model(tokens[i:i+1].to(device)).logits.float(),
                                        dim=-1).cpu() for i in range(tokens.shape[0])]
    dense = copy.deepcopy(model).cpu()

    rows = []
    for i, name in enumerate(sampled):
        arm = copy.deepcopy(dense).to(device).eval()
        quantize_linear_modules(arm, config,
                                should_quantize=lambda n, m, target=name: n == target)
        actual = wikitext_kl(arm, ref_chunks, tokens, device)
        r = stage_rho(args.bits, resolve_group_size(layers[name].in_features,
                                                    args.group_size))
        predicted = alphas[name] * r ** (2 * args.passes)
        rows.append({"layer": name, "alpha": alphas[name],
                     "predicted": predicted, "actual": actual})
        print(f"  [{i+1}/{len(sampled)}] {name:<40} pred {predicted:.3e}  "
              f"actual {actual:.3e}", flush=True)
        del arm
        torch.cuda.empty_cache()

    pred = [r["predicted"] for r in rows]
    act = [r["actual"] for r in rows]
    rs = spearman(pred, act)
    ratio = sorted(a / p if p > 0 else float("inf") for a, p in zip(act, pred))
    print(f"\nSpearman(predicted, actual) = {rs:.3f} over {len(rows)} layers")
    print(f"actual/predicted ratio: median {ratio[len(ratio)//2]:.2f}, "
          f"range [{ratio[0]:.2f}, {ratio[-1]:.2f}]")
    print("The theorem needs the ranking, not the scale: a constant ratio is fine,")
    print("a low Spearman is not, because the allocator only ever uses the order.")

    out = {"model": args.model, "passes": args.passes, "bits": args.bits,
           "spearman": rs, "rows": rows}
    path = args.out or f"results/linearity_{args.model.split('/')[-1]}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
