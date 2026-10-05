"""KL against a reference too large to hold beside the model under test.

``eval_protocol.kl_to_reference`` needs both models resident. For Gemma-3-12B that
is two 23 GiB bf16 towers, which no single 24 GiB card holds, and it is why
tab:kl carried a provenance note saying the table needed a rented host.

It does not. The reference is scored once and its per-window log-probabilities
are cached to disk; every tier is then scored alone and read against the cache.
The cost is disk: one window of 2048 tokens over Gemma's 262k vocabulary is
1.07 GiB in fp16, so a 32-window run caches about 34 GiB and reads it back once
per tier. That is cheap against re-scoring an 12B reference ten times.

Two details that are not optional. The compute dtype is the source model's own,
because Gemma's hidden states pass 6e5 and evaluating it in fp16 overflows to inf
and returns NaN for every metric. And the reference tower is offloaded with
``device_map`` rather than pinned, since 23 GiB of weights leaves no room for a
262k-wide logit tensor on a 24 GiB card.

    uv run python experiments/kl_two_phase.py \
        --reference google/gemma-3-12b-it --artifact /mnt/bulk/gemma/trellis-p3 \
        --all-tiers --cache /mnt/bulk/refcache/gemma --out results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_progressive import dispatch, gpu_budget  # noqa: E402
from eval_protocol import get_test_tokens  # noqa: E402
from load_progressive import load_progressive  # noqa: E402

CACHE_DTYPE = torch.float16
LOGIT_CHUNK = 512          # rows of logits turned into fp32 log-probs at once


def offloaded(repo: str, dtype: torch.dtype, gpu_gib: float, host_gib: float):
    """The reference tower, with whatever does not fit left on the host.

    Use AutoModelForCausalLM and nothing cleverer. Reaching for the text-only
    class on a multimodal checkpoint looks right and is not: Gemma-3-12B-it
    stores its embeddings and final norm under ``language_model.model.*``, so
    ``Gemma3ForCausalLM`` silently leaves ``model.embed_tokens.weight`` and
    ``model.norm.weight`` randomly initialised. It loads, it ties the head, it
    reports one missing tensor, and it answers "The capital of France is" with
    a Tamil fragment. The auto class resolves to the conditional-generation
    layout the checkpoint actually has, which is also the layout the artifacts
    were written in, so reference and tier are compared like for like.
    """
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        repo, dtype=dtype, device_map="auto",
        max_memory={0: f"{gpu_gib}GiB", "cpu": f"{host_gib}GiB"}).eval()


@torch.no_grad()
def logprobs(model, window: torch.Tensor) -> torch.Tensor:
    """Log-probabilities for one window, on the host in ``CACHE_DTYPE``."""
    logits = model(window.unsqueeze(0).to("cuda"), use_cache=False).logits[0]
    out = torch.empty(logits.shape, dtype=CACHE_DTYPE, device="cpu")
    for t0 in range(0, logits.shape[0], LOGIT_CHUNK):
        piece = logits[t0:t0 + LOGIT_CHUNK].float().log_softmax(dim=-1)
        out[t0:t0 + LOGIT_CHUNK] = piece.to(CACHE_DTYPE).cpu()
        del piece
    del logits
    return out


def write_cache(model, tokens: torch.Tensor, cache: pathlib.Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for i in range(tokens.shape[0]):
        path = cache / f"w{i:04d}.pt"
        if path.exists():
            continue
        torch.save(logprobs(model, tokens[i]), path)
        print(f"#   cached window {i + 1}/{tokens.shape[0]}", flush=True)


def kl_against_cache(model, tokens: torch.Tensor,
                     cache: pathlib.Path) -> tuple[float, list[float]]:
    """Mean batchmean KL(reference || model), and the per-window values.

    The per-window list is what says whether two tiers are separable. Four rows
    of tab:kl sit inside 0.07 nats at P=3, and a mean over 32 windows with no
    spread beside it cannot support an ordering claim between them.
    """
    per_window = []
    total = 0.0
    for i in range(tokens.shape[0]):
        lp = torch.load(cache / f"w{i:04d}.pt")
        lq_all = logprobs(model, tokens[i])
        rows, acc = lp.shape[0], 0.0
        for t0 in range(0, rows, LOGIT_CHUNK):
            p = lp[t0:t0 + LOGIT_CHUNK].float()
            q = lq_all[t0:t0 + LOGIT_CHUNK].float()
            acc += float((p.exp() * (p - q)).sum())
            del p, q
        per_window.append(acc / rows)
        total += acc / rows
        del lp, lq_all
    return total / tokens.shape[0], per_window


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--all-tiers", action="store_true")
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=32,
                    help="windows; 32x2048 matches the recorded Gemma run")
    ap.add_argument("--headroom-gib", type=float, default=3.5,
                    help="reserved per card for activations and the logit tensor")
    ap.add_argument("--ref-gpu-gib", type=float, default=14.0,
                    help="card ceiling for the reference; the rest offloads")
    ap.add_argument("--host-gib", type=float, default=45.0,
                    help="host ceiling for the offloaded remainder")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.artifact, "progressive_config.json")) as fh:
        cfg = json.load(fh)
    tiers = (list(range(1, cfg["passes"] + 1)) if args.all_tiers
             else [args.passes or cfg["passes"]])
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.reference)
    tokens = get_test_tokens("wikitext2", tok, args.seqlen)[:args.limit]
    print(f"# {tokens.shape[0]} windows of {args.seqlen}, stage {cfg['stage']}, "
          f"g={cfg['group_size']}, trellis_bits={cfg.get('trellis_bits')}", flush=True)

    cache = pathlib.Path(args.cache)
    if not (cache / f"w{tokens.shape[0] - 1:04d}.pt").exists():
        print("# phase 1: scoring the reference", flush=True)
        ref = offloaded(args.reference, dtype, args.ref_gpu_gib, args.host_gib)
        write_cache(ref, tokens, cache)
        del ref
        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    for p in tiers:
        print(f"# phase 2: tier P={p}", flush=True)
        model, _, _ = load_progressive(args.artifact, passes=p, dtype=dtype, device="cpu")
        # gpu_budget sizes host memory from AVAILABLE pages, but the model is
        # already sitting in them, so it undercounts by the model's own footprint
        # and accelerate then assigns the overflow to disk and dies asking for an
        # offload_dir. The card budget is still measured from free VRAM; only the
        # host figure is stated outright.
        budget = gpu_budget(args.headroom_gib) or {}
        budget["cpu"] = f"{args.host_gib}GiB"
        model = dispatch(model, "auto", budget).eval()
        kl, per_window = kl_against_cache(model, tokens, cache)
        bits = p * (cfg["bits"] + 16.0 / cfg["group_size"])
        rows.append({"stage": cfg["stage"], "passes": p, "effective_bits": bits,
                     "trellis_bits": cfg.get("trellis_bits"), "kl": kl,
                     "per_window": per_window,
                     "windows": int(tokens.shape[0]), "seqlen": args.seqlen})
        print(f"| {cfg['stage']} | P={p} | {bits:.4f} bit | KL {kl:.4f} |", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
