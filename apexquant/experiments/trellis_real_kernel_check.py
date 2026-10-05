"""Verify and time the fused mat-vec against the trellis code we actually ship.

Why this exists. `triton_gemv_tcq` is a fused mat-vec for a code with the
bitshift trellis's *shape* -- L=16, tail-biting inside one 32-bit word, a
1MAD-computed codeword -- and is explicit that it is not the code `TrellisCodec`
emits. `TrellisCodec._walk_kernel` decodes the right code but materialises every
weight in DRAM. `triton_gemv_trellis` is the first path that decodes the shipped
format inside the mat-vec, so it is the first one whose cost can be attributed to
the trellis checkpoints rather than to something shaped like them. It also folds
every residual pass into one launch, which is what the progressive ladder needs:
a P=3 checkpoint that costs three launches and three output round-trips per
layer is three times a decode that was already the bottleneck.

Five things are checked, in this order, because each is meaningless without the
one before it:

1. **Exactness, element by element.** Driving one pass with a one-hot activation
   and the others with zeros makes ``out[n]`` a single product, so it must equal
   the dense reference *bit for bit*, not approximately. The probe positions sit
   where the window logic is most likely to be wrong: the zero-padded prefix of a
   trellis run, both sides of a scale-group boundary, and both sides of a word
   boundary. Probing one pass at a time is also what makes the pass strides
   testable exactly rather than in aggregate. A window off by one symbol still
   decodes to a plausibly Gaussian weight, so an approximate check on random
   activations would not reliably catch it -- this one does.
2. **The GEMV itself.** Random activations against a dense reference, which
   catches strides, split-K and masking rather than the decode.
3. **Fusion against sequence.** One P-pass launch must agree with P single-pass
   launches summed, to within fp32 atomic reordering. This is the check that the
   pass axis changed the schedule and nothing else.
4. **Cost**, against the path that runs today: reconstruct the group to dense
   through `TrellisCodec.decode`, then hand it to cuBLAS; and against the
   per-pass path the fusion replaces.
5. **What the ceiling is made of.** The table sweep holds bandwidth and work per
   weight fixed and varies only the resident codeword table, at P=1 and P=3, so
   it prices the gather and says whether adding passes makes the contention for
   it worse.

Codes are drawn at random rather than produced by Viterbi. The encoder is
irrelevant to whether the decoder is right, any symbol sequence is a legal input,
and random symbols cover the state space better than a Viterbi path would --
which matters, because the one thing being tested is the state-to-codeword map.

    uv run python experiments/trellis_real_kernel_check.py
"""

from __future__ import annotations

import time

import torch

from apexquant.kernels.triton_gemv_trellis import (
    GEMV_TRELLIS_AVAILABLE,
    trellis_window_table,
    triton_gemv_trellis,
)
from apexquant.vq.scalar_codec import pack_indices
from apexquant.vq.trellis_codec import TrellisCodec

DEV = torch.device("cuda")
GS = 512


def make_case(codec: TrellisCodec, passes: int, n: int, k: int, gs: int,
              seed: int):
    """Random codes for ``passes`` residual passes, plus their reconstruction.

    Returns:
        ``(words, scale, dense)`` -- int32 words ``(P, n, k*bits/32)``, float32
        scales ``(P, n, k/gs)``, and the float32 ``(P, n, k)`` weights the kernel
        must reproduce, one per pass.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    ngroups = k // gs
    sym = torch.randint(0, 1 << codec.bits, (passes, n * ngroups, gs),
                        generator=g, device=DEV, dtype=torch.uint8)
    codes = pack_indices(sym, codec.bits).reshape(passes, n, ngroups, -1)

    scale = torch.rand((passes, n, ngroups), generator=g, device=DEV).add(0.5)
    scale = scale.to(torch.float16).to(torch.float32)
    # Exactly the product `progressive_artifact.dequantize` forms per pass: the
    # codec folds its own scale into the value, the group scale multiplies that.
    dense = (codec.decode(codes) * scale.unsqueeze(-1)).reshape(passes, n, k)

    words = codes.reshape(passes, n, -1).contiguous().view(torch.int32)
    return words, scale.contiguous(), dense


def probe_positions(gs: int, k: int, m: int, vpw: int) -> list[int]:
    """Coordinates where a wrong window is most likely to show up."""
    pos = set()
    pos.update(range(m + 1))                       # zero-padded run prefix
    pos.update(range(vpw - 1, vpw + 2))            # first word boundary
    for boundary in (gs, 2 * gs):                  # scale-group boundaries
        if boundary < k:
            pos.update(range(boundary - 2, boundary + m + 1))
    g = torch.Generator().manual_seed(11)          # a fixed probe set, so a
    pos.update(int(v) for v in torch.randint(0, k, (16,), generator=g))
    return sorted(p for p in pos if 0 <= p < k)    # rerun compares like for like


def check_exact(codec: TrellisCodec, passes: int, n: int, k: int, gs: int,
                seed: int) -> tuple:
    """One-hot probes, one pass at a time: every probed weight must match bit
    for bit, which also pins each pass's stride independently."""
    words, scale, dense = make_case(codec, passes, n, k, gs, seed)
    table = trellis_window_table(codec)
    m, vpw = codec.trellis_bits // codec.bits, 32 // codec.bits

    worst, bad, total = 0.0, 0, 0
    for j in probe_positions(gs, k, m, vpw):
        for p in range(passes):
            rx = torch.zeros((passes, k), device=DEV, dtype=torch.float32)
            rx[p, j] = 1.0
            got = triton_gemv_trellis(rx, words, scale, table, gs, codec.bits,
                                      codec.trellis_bits).reshape(-1)
            total += 1
            if not torch.equal(got, dense[p, :, j]):
                bad += 1
                worst = max(worst, (got - dense[p, :, j]).abs().max().item())
    return total, bad, worst


def check_gemv(codec: TrellisCodec, passes: int, n: int, k: int, gs: int,
               seed: int) -> float:
    """Random activations against a dense reference; relative max error."""
    words, scale, dense = make_case(codec, passes, n, k, gs, seed)
    table = trellis_window_table(codec)
    rx = torch.randn((passes, k), device=DEV)
    got = triton_gemv_trellis(rx, words, scale, table, gs, codec.bits,
                              codec.trellis_bits)
    ref = sum(rx[p].unsqueeze(0) @ dense[p].T for p in range(passes))
    return ((got - ref).abs().max() / ref.abs().max()).item()


def check_fused_vs_sequential(codec: TrellisCodec, passes: int, n: int, k: int,
                              gs: int, seed: int) -> float:
    """One P-pass launch against P single-pass launches summed."""
    words, scale, dense = make_case(codec, passes, n, k, gs, seed)
    table = trellis_window_table(codec)
    rx = torch.randn((passes, k), device=DEV)
    args = (table, gs, codec.bits, codec.trellis_bits)
    fused = triton_gemv_trellis(rx, words, scale, *args)
    seq = sum(triton_gemv_trellis(rx[p:p + 1], words[p:p + 1], scale[p:p + 1],
                                  *args) for p in range(passes))
    return ((fused - seq).abs().max() / seq.abs().max()).item()


def timed(fn, iters: int = 200, blocks: int = 5) -> float:
    """Best of ``blocks`` timing runs, after enough warmup to reach clocks.

    Ten warmup iterations and one block is not enough on this card: the first
    measurement in a loop lands mid clock-ramp and reads 3-5x slow, which is
    enough to invert the ordering of the arms being compared. Taking the minimum
    rather than the mean also drops the blocks that a background process shared.
    """
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(blocks):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / iters * 1e3)
    return best


def main() -> None:
    if not GEMV_TRELLIS_AVAILABLE:
        raise SystemExit("triton is not available")
    torch.manual_seed(0)
    P = 3                                    # the 6-bit tier of the ladder

    print("# fused trellis mat-vec: exactness, the GEMV, fusion, then cost\n")

    # --- 1-3: correctness across every (k, L) the codec supports ------------
    print(f"# (k, L) sweep at P={P}, N=256, K=1024, group=512. `exact` counts "
          "one-hot probes\n# bit-identical to the dense reference, one pass at "
          "a time; `rel err` is the\n# random-activation GEMV; `vs seq` is the "
          "fused launch against P separate ones.\n")
    print(f"{'k':>3}{'L':>4}{'M':>4}{'states':>8}{'table':>9}"
          f"{'exact':>13}{'rel err':>11}{'vs seq':>11}  verdict")
    pairs = [(2, 12), (2, 16), (2, 8), (4, 12), (4, 16), (1, 12), (8, 16)]
    all_ok = True
    for bits, tb in pairs:
        codec = TrellisCodec(GS, bits=bits, trellis_bits=tb, device=DEV)
        npos, bad, worst = check_exact(codec, P, 256, 1024, GS, seed=1)
        rel = check_gemv(codec, P, 256, 1024, GS, seed=2)
        seq = check_fused_vs_sequential(codec, P, 256, 1024, GS, seed=3)
        ok = bad == 0 and rel < 1e-3 and seq < 1e-5
        all_ok &= ok
        note = "OK" if ok else f"FAIL (worst {worst:.3e})"
        print(f"{bits:>3}{tb:>4}{tb // bits:>4}{1 << tb:>8}"
              f"{(1 << tb) * 4 / 1024:>7.0f} K{npos - bad:>8}/{npos:<4}"
              f"{rel:>11.2e}{seq:>11.2e}  {note}")

    # --- 4: cost on the shipped configuration -------------------------------
    print(f"\n# shipped config (k=2, L=12) at P={P} on real shapes. `per-pass` "
          f"is {P} separate\n# launches, the path the fusion replaces. "
          "`walk+dense` is what runs today:\n# reconstruct through "
          "TrellisCodec.decode, then cuBLAS, once per pass.\n")
    codec = TrellisCodec(GS, bits=2, trellis_bits=12, device=DEV)
    table = trellis_window_table(codec)
    print(f"{'layer':<18}{'N':>6}{'K':>7}{'P=1 ms':>9}{'fused ms':>10}"
          f"{'vs P=1':>8}{'per-pass ms':>13}{'walk+dense ms':>15}{'gain':>8}")
    shapes = [("q_proj-like", 4096, 4096), ("down_proj-like", 4096, 14336),
              ("qwen down_proj", 3584, 18944)]
    for label, n, k in shapes:
        words, scale, dense = make_case(codec, P, n, k, GS, seed=4)
        codes = words.view(torch.uint8).reshape(P, n, k // GS, -1)
        rx = torch.randn((P, k), device=DEV)
        args = (table, GS, 2, 12)

        got = triton_gemv_trellis(rx, words, scale, *args)
        ref = sum(rx[p].unsqueeze(0) @ dense[p].T for p in range(P))
        all_ok &= ((got - ref).abs().max() / ref.abs().max()).item() < 1e-3

        t_one = timed(lambda: triton_gemv_trellis(rx[:1], words[:1], scale[:1],
                                                  *args))
        t_fused = timed(lambda: triton_gemv_trellis(rx, words, scale, *args))
        t_seq = timed(lambda: sum(
            triton_gemv_trellis(rx[p:p + 1], words[p:p + 1], scale[p:p + 1],
                                *args) for p in range(P)))
        t_walk = timed(lambda: sum(
            (codec.decode(codes[p]) * scale[p].unsqueeze(-1)).reshape(n, k)
            @ rx[p] for p in range(P)))
        print(f"{label:<18}{n:>6}{k:>7}{t_one:>9.3f}{t_fused:>10.3f}"
              f"{t_fused / t_one:>7.2f}x{t_seq:>13.3f}{t_walk:>15.3f}"
              f"{t_walk / t_fused:>7.1f}x")
        del words, scale, dense, codes
        torch.cuda.empty_cache()

    # --- 5: is the table the ceiling, and does P make it worse? -------------
    print("\n# table-size sweep at k=2, N=4096, K=14336. Identical bandwidth "
          "and identical\n# work per weight; only the resident table differs. "
          "`GB/s` counts code bytes\n# only, against the 4090's 1008 GB/s peak.\n")
    n, k = 4096, 14336
    print(f"{'L':>3}{'states':>8}{'table':>9}{'P=1 ms':>9}{'P=1 GB/s':>10}"
          f"{'P=3 ms':>9}{'P=3 GB/s':>10}{'P=3/P=1':>9}")
    for tb in (8, 12, 16):
        c = TrellisCodec(GS, bits=2, trellis_bits=tb, device=DEV)
        tbl = trellis_window_table(c)
        words, scale, _ = make_case(c, P, n, k, GS, seed=5)
        rx = torch.randn((P, k), device=DEV)
        t1 = timed(lambda: triton_gemv_trellis(rx[:1], words[:1], scale[:1],
                                               tbl, GS, 2, tb))
        t3 = timed(lambda: triton_gemv_trellis(rx, words, scale, tbl, GS, 2, tb))
        bytes1 = n * k * 2 / 8
        print(f"{tb:>3}{1 << tb:>8}{(1 << tb) * 4 / 1024:>7.0f} K{t1:>9.3f}"
              f"{bytes1 / (t1 * 1e-3) / 1e9:>10.0f}{t3:>9.3f}"
              f"{P * bytes1 / (t3 * 1e-3) / 1e9:>10.0f}{t3 / t1:>8.2f}x")
        del words, scale, tbl
        torch.cuda.empty_cache()

    print("\nRESULT:", "all checks passed" if all_ok else "FAILURES ABOVE")


if __name__ == "__main__":
    main()
