"""Run a real trellis artifact through the fused kernel, end to end.

`load_progressive --fused` keeps the codes packed and swaps every quantised
`nn.Linear` for a `ProgressiveTrellisLinear`. This checks that the swap produces
the same model, on the artifact rather than on synthetic codes, and reports what
it costs.

The equivalence check needs no test-only code path. `ProgressiveTrellisLinear`
routes a single row to the fused mat-vec and anything wider to the dense
fallback, so feeding the *same* token as a batch of one and as a batch of two
runs the two paths against each other through every layer of the model. Row 0 of
the batch-two logits must match the batch-one logits.

They will not match bit for bit, and the reason is worth stating rather than
absorbing into a tolerance: the fused path accumulates in fp32 against an fp32
table, while the dense path rounds each reconstructed weight to the model dtype
first. The fused path is the more accurate of the two.

    uv run python experiments/fused_trellis_model_check.py \
        --artifact /mnt/bulk/gemma3/trellis-g256-p3
"""

from __future__ import annotations

import argparse
import time

import torch

from load_progressive import load_progressive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--prompt", default="The capital of Denmark is")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"),
                    help="Gemma needs bfloat16: its hidden states pass 6e5, which "
                         "is an order of magnitude past what fp16 can hold, so "
                         "the fp16 default overflows to NaN part-way up the stack")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, tok, cfg = load_progressive(args.artifact, args.passes, dtype=dtype,
                                      fused=True)
    load_s = time.perf_counter() - t0

    # Compare like with like: what the quantised layers hold now, against what
    # the same layers would hold reconstructed to fp16. Peak allocation is not
    # the comparison -- it also carries the embeddings and every passthrough
    # tensor, which this path does not change.
    from apexquant.vq.progressive_trellis_linear import ProgressiveTrellisLinear
    packed_gib = sum(m.words.nbytes + m.scale.nbytes for m in model.modules()
                     if isinstance(m, ProgressiveTrellisLinear)) / 2**30
    dense_gib = sum(t["in_features"] * t["out_features"]
                    for t in cfg["quantised_tensors"].values()) * 2 / 2**30
    print(f"# loaded in {load_s:.1f}s; peak {torch.cuda.max_memory_allocated()/2**30:.2f}"
          f" GiB allocated over the whole model\n#   quantised layers hold "
          f"{packed_gib:.3f} GiB packed vs {dense_gib:.3f} GiB reconstructed to "
          f"fp16 ({dense_gib / packed_gib:.2f}x)")

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(model.device)
    one = ids[:, :1]

    with torch.no_grad():
        fused = model(one).logits[0, -1].float()
        both = model(one.repeat(2, 1)).logits[0, -1].float()
    rel = ((fused - both).abs().max() / both.abs().max()).item()
    agree = (fused.argmax() == both.argmax()).item()
    print(f"\n# fused mat-vec vs dense fallback, all {len(cfg['quantised_tensors'])} "
          f"layers, one token:\n#   max rel {rel:.2e}, argmax agrees: {agree}\n"
          f"#   (one {args.dtype} ulp is {torch.finfo(dtype).eps:.1e}; the two "
          "paths agree to ~5e-07 per layer,\n#   and what accumulates over the "
          "stack is the activation dtype between layers)")

    # Decode, which is the regime the kernel exists for: every step is one row.
    with torch.no_grad():
        out = model(ids, use_cache=True)
        cur, past = out.logits[:, -1].argmax(-1, keepdim=True), out.past_key_values
        got = [cur.item()]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.tokens - 1):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur = out.logits[:, -1].argmax(-1, keepdim=True)
            got.append(cur.item())
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / (args.tokens - 1) * 1e3

    print(f"\n# greedy decode at {ms:.1f} ms/token "
          f"({1000 / ms:.1f} tok/s), batch one\n"
          f"#   {args.prompt!r} -> {tok.decode(got)!r}")
    print("\nRESULT:", "fused model matches the dense path"
          if rel < 1e-2 and agree else "MISMATCH -- see above")


if __name__ == "__main__":
    main()
