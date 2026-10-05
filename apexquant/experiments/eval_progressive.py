"""Evaluate a progressive artifact under the QuIP#/QTIP/YAQA protocol.

Runs the metrics their published tables report, at their sequence length, so
the numbers are comparable rather than merely similar: WikiText-2 and C4
perplexity at seqlen 8192, and optionally batchmean KL against the FP16
reference, which is YAQA's headline metric.

Sizing. Evaluation reconstructs dense weights, so a 70B artifact needs ~141 GB
of fp16 regardless of how few bits it was stored in; packed-form execution would
need a fused kernel for the stage's code format. Perplexity therefore needs
enough aggregate VRAM for the dense model, and KL needs that *plus* the
reference model resident at the same time. Pass `--device-map auto` to spread
across cards.

    # one tier, perplexity only
    uv run python experiments/eval_progressive.py \
        --artifact /mnt/bulk/llama70b/e8p-p3 --passes 2 --device-map auto

    # every tier of an artifact, plus KL against the reference
    uv run python experiments/eval_progressive.py \
        --artifact /mnt/bulk/llama70b/e8p-p3 --all-tiers --kl --device-map auto
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import torch

# Both this directory and the library, derived from __file__ rather than from
# the working directory: a pod runs these scripts by path, from wherever it
# happens to be, and `pip install -e .` is not what puts the package there.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))


from eval_protocol import (  # noqa: E402
    DEFAULT_SEQLEN,
    get_test_tokens,
    kl_to_reference,
    perplexity,
)
from load_progressive import load_progressive  # noqa: E402


# Share of free host memory offered to accelerate for offload. Not all of
# it: the process still needs room to assemble each tier before dispatch.
HOST_FRACTION = 0.8


def gpu_budget(headroom_gib: float | None) -> dict | None:
    """Per-card memory ceilings that leave room for the forward pass.

    `infer_auto_device_map` fills each card to its stated capacity before
    spilling, and its default capacity is the whole card. On a 70B bf16 model at
    131.4 GiB spread over 141 GB cards that packs the first card to the last
    hundred megabytes, leaving nothing for activations or the logit tensor, which
    at a 128k vocabulary is 3.9 GiB per window at seqlen 8192. Reserving a fixed
    slice per card is what makes a run that fits on paper also fit in practice.

    Expressed as headroom rather than as an absolute ceiling so the same setting
    is right on 80 GB and 141 GB cards; a ceiling copied from one is either
    wasteful or impossible on the other.

    Measured against FREE memory, not total. accelerate treats `max_memory` as an
    absolute per-device budget rather than an allowance on top of what is already
    resident, so a dict built from total capacity is correct for the first model
    onto the cards and wrong for every one after it. That is not hypothetical: the
    tier took 114 GiB of card 0, the reference was then told it could use 116 GiB
    there, and it tried to place 27 GiB into the 25 GiB that was left.
    """
    if not headroom_gib:
        return None
    # Reserved-but-unallocated blocks read as used to the driver, so hand them
    # back before measuring or the budget shrinks with every tier scored.
    torch.cuda.empty_cache()
    budget = {}
    for i in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(i)[0] / 2**30
        budget[i] = f"{max(0.5, free - headroom_gib):.1f}GiB"
    # Name host memory explicitly. A max_memory dict of GPUs alone tells
    # accelerate that host memory does not exist, so anything that does not fit
    # the cards is assigned to *disk*, and dispatch_model then dies asking for an
    # offload_dir it was never given. Overflowing to host is slow but correct and
    # is what the caller means; overflowing to disk is a crash after the load.
    # The headroom is for activations and logits, which live on the card, so it
    # is not subtracted here; host memory only needs to keep a little back for
    # the process itself.
    host = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 2**30
    budget["cpu"] = f"{max(1.0, host * HOST_FRACTION):.1f}GiB"
    return budget


def dispatch(model, device_map: str | None, max_memory: dict | None = None):
    """Spread a model across devices when it will not fit on one.

    Returns the model unchanged when `device_map` is unset, which is the right
    behaviour for anything that fits a single card.
    """
    if not device_map:
        return model
    from accelerate import dispatch_model, infer_auto_device_map
    dmap = infer_auto_device_map(
        model, max_memory=max_memory,
        no_split_module_classes=getattr(model, "_no_split_modules", None))
    return dispatch_model(model, device_map=dmap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--all-tiers", action="store_true",
                    help="evaluate every prefix depth 1..P")
    ap.add_argument("--datasets", default="wikitext2",
                    help="comma-separated: wikitext2,c4")
    ap.add_argument("--seqlen", type=int, default=DEFAULT_SEQLEN)
    ap.add_argument("--kl", action="store_true",
                    help="also report KL against the FP16 reference; needs room "
                         "for both models at once")
    ap.add_argument("--device-map", default=None,
                    help="'auto' to spread across GPUs")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N chunks (smoke tests)")
    ap.add_argument("--dtype", default=None, choices=("float16", "bfloat16"),
                    help="Compute dtype for both the tier and the KL reference. "
                         "Defaults to the source model's own torch_dtype, which "
                         "is the only setting that is right by construction: "
                         "Gemma's hidden states pass 6e5, an order of magnitude "
                         "past fp16's range, so evaluating it in fp16 overflows "
                         "to inf and every metric comes back NaN. The overflow "
                         "shows up first at the shallowest tier, where the "
                         "quantization error is largest.")
    ap.add_argument("--gpu-headroom-gib", type=float, default=None,
                    help="reserve this much of every card for activations and "
                         "logits rather than packing it with weights; needed "
                         "when a model nearly fills the cards it is spread over")
    ap.add_argument("--out", default=None, help="write results as JSON")
    args = ap.parse_args()

    with open(os.path.join(args.artifact, "progressive_config.json")) as fh:
        cfg = json.load(fh)
    tiers = (list(range(1, cfg["passes"] + 1)) if args.all_tiers
             else [args.passes or cfg["passes"]])
    dsets = args.datasets.split(",")

    print(f"# artifact: {args.artifact}")
    print(f"# stage={cfg['stage']} group_size={cfg['group_size']} "
          f"seed={cfg['seed']} source={cfg['source_model']}")
    print(f"# protocol: seqlen {args.seqlen}, upstream-matched")
    if args.seqlen != 2048:
        print("# NOTE: this repository's other tables use seqlen 2048. "
              "Perplexity falls with context length; do not pool the two.")

    # Resolve the compute dtype once, from the source model's own config unless
    # overridden. Hardcoding fp16 here is what made Gemma's P=1 row NaN.
    def resolve_dtype() -> tuple[torch.dtype, str]:
        if args.dtype is not None:
            return getattr(torch, args.dtype), "--dtype"
        # The quantizer copies the source config.json into the artifact, so the
        # answer is on disk. Prefer it over the Hub: it needs no network and no
        # token, and a gated source would otherwise turn a dtype lookup into a
        # 401 that kills the whole evaluation.
        local = os.path.join(args.artifact, "config.json")
        if os.path.isfile(local):
            with open(local) as fh:
                src_cfg = json.load(fh)
            raw = src_cfg.get("torch_dtype") or src_cfg.get("dtype")
            if isinstance(raw, str) and hasattr(torch, raw):
                return getattr(torch, raw), "the artifact's config.json"
        try:
            from transformers import AutoConfig
            src = AutoConfig.from_pretrained(cfg["source_model"])
            raw = getattr(src, "torch_dtype", None) or getattr(src, "dtype", None)
            if isinstance(raw, torch.dtype):
                return raw, "the source model config"
            if isinstance(raw, str) and hasattr(torch, raw):
                return getattr(torch, raw), "the source model config"
        except Exception as exc:
            print(f"# could not read the source config ({exc.__class__.__name__}); "
                  f"falling back to float16")
        return torch.float16, "the fallback"

    dtype, dtype_src = resolve_dtype()
    print(f"# compute dtype: {dtype} (from {dtype_src})")

    reference = None
    results: dict[str, dict] = {}
    print(f"\n| tier | eff. bits | " + " | ".join(dsets)
          + (" | KL |" if args.kl else " |"))
    print("|---|---:|" + "---:|" * (len(dsets) + (1 if args.kl else 0)))

    for P in tiers:
        model, tok, _ = load_progressive(args.artifact, passes=P,
                                         device="cpu" if args.device_map else None,
                                         dtype=dtype)
        budget = gpu_budget(args.gpu_headroom_gib)
        model = dispatch(model, args.device_map, budget)
        eff = P * (cfg["bits"] + 16.0 / cfg["group_size"])
        row: dict[str, float] = {}
        for ds in dsets:
            toks = get_test_tokens(ds, tok, args.seqlen)
            if args.limit:
                toks = toks[: args.limit]
            row[ds] = perplexity(model, toks)
        if args.kl:
            if reference is None:
                from transformers import AutoModelForCausalLM
                # The tier is already resident, so accelerate sees the cards as
                # they now are and places the reference around it. The same
                # headroom applies: without it the last card it lands on has no
                # room left for the logits the KL reduction needs.
                reference = AutoModelForCausalLM.from_pretrained(
                    cfg["source_model"], dtype=dtype,
                    device_map=args.device_map or None,
                    max_memory=gpu_budget(args.gpu_headroom_gib))
                if not args.device_map:
                    reference = reference.cuda()
                reference.eval()
            toks = get_test_tokens(dsets[0], tok, args.seqlen)
            if args.limit:
                toks = toks[: args.limit]
            row["kl"] = kl_to_reference(model, reference, toks)
        results[str(P)] = {"effective_bits": eff, **row}
        if any(v != v or v in (float("inf"), float("-inf")) for v in row.values()):
            print(f"### WARNING: tier P={P} produced a non-finite metric. "
                  f"This is overflow, not a score: re-run with "
                  f"--dtype bfloat16 if the model's activations exceed fp16 range.")
        cells = " | ".join(f"{row[d]:.4f}" for d in dsets)
        kl_cell = f" | {row['kl']:.5f}" if args.kl else ""
        print(f"| P={P} | {eff:.4f} | {cells}{kl_cell} |", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"artifact": args.artifact, "config": cfg,
                       "seqlen": args.seqlen, "results": results}, fh, indent=2)
        print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()
