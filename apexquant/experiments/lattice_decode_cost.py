"""What does a lattice stage cost at decode time, at matched bandwidth?

tab:stages shows an E8-lattice stage cuts the per-pass distortion from 0.1171 to
0.0906 at the same two bits per coordinate, and the paper concedes it has not run
one end to end. This measures the decode half of that concession.

The comparison is bandwidth-neutral by construction. Both arms load the same
number of int32 words per row and read the same per-group scales:

  scalar   16 two-bit indices per word; 4-entry codebook, register resident.
  E8 table two 16-bit codes per word, each naming an 8-dimensional lattice point;
           2^16 x 8 codebook, 1 MiB, necessarily in global memory.
  E8 gen   the same 16 bits per 8 coordinates, but split 8 bits of source index
           and 8 sign bits, so the table is 256 x 8 = 4 KiB and stays in L1.
           This is the regime QuIP#'s E8P and QTIP's computed codes occupy, and
           without it the stored-table number is a strawman.
  TCQ 1MAD no table at all: each word is a tail-biting bitshift-trellis block of
           16 weights, and each weight's codeword is computed from its 16-bit
           window by QTIP's 1MAD (multiply-add, byte-sum, affine rescale).
           The table shrinks 1 MiB -> 4 KiB -> 0.

So the latency difference is decode cost alone. We also check correctness against
a torch reference, because a fast wrong kernel measures nothing.

    uv run python experiments/lattice_decode_cost.py
"""

from __future__ import annotations

import argparse
import time

import torch

from apexquant.kernels.triton_gemv_e8 import (
    GEMV_E8_AVAILABLE,
    triton_gemv_e8,
)
from apexquant.kernels.triton_gemv_e8p_real import (
    GEMV_E8P_REAL_AVAILABLE,
    triton_gemv_e8p_real,
)
from apexquant.vq.e8p_codec import E8PCodec
from apexquant.kernels.triton_gemv_e8p import (
    GEMV_E8P_AVAILABLE,
    triton_gemv_e8p,
)
from apexquant.kernels.triton_gemv_mp import (
    GEMV_MP_AVAILABLE,
    triton_gemv_mp,
)
from apexquant.kernels.triton_gemv_tcq import (
    GEMV_TCQ_AVAILABLE,
    tcq_decode_reference,
    triton_gemv_tcq,
)

DEV = torch.device("cuda")

# Real decoder-linear shapes from the models in the paper (N out, K in).
SHAPES = [
    ("Mistral-7B q_proj", 4096, 4096),
    ("Mistral-7B down_proj", 4096, 14336),
    ("Qwen2.5-7B down_proj", 3584, 18944),
]


def e8_codebook(n_codewords: int = 1 << 16, d: int = 8) -> torch.Tensor:
    """The `n_codewords` shortest E8 points, scaled to unit second moment.

    Same construction as experiments/vector_stage_rho.py. The scale is
    irrelevant to timing and matters only for the correctness check.
    """
    rng = torch.arange(-4, 4, device=DEV, dtype=torch.int8)
    grids = torch.meshgrid(*([rng] * d), indexing="ij")
    pts = torch.stack([g.reshape(-1) for g in grids], dim=1).to(torch.float32)
    even = pts[(pts.sum(1).to(torch.int64) % 2) == 0]
    e8 = torch.cat([even, even + 0.5], dim=0)
    nrm = e8.pow(2).sum(1)
    keep = nrm <= 12.0 + 1e-6
    e8, nrm = e8[keep], nrm[keep]
    e8 = e8[nrm.argsort()][:n_codewords].contiguous()
    return (e8 / e8.pow(2).sum(1).mean().sqrt()).contiguous()


def bench(fn, iters: int = 50, warmup: int = 25, reps: int = 5) -> float:
    """Min-of-reps mean latency, in ms.

    Triton JIT-compiles on first call and the first shape benchmarked otherwise
    absorbs that cost, which made an earlier version of this script report the
    smallest layer as the slowest. We warm up generously and take the minimum
    over repeated timing blocks, which is the robust estimator for a kernel that
    is being perturbed only from above.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3 / iters)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=512)
    ap.add_argument("--iters", type=int, default=80)
    args = ap.parse_args()

    if not (GEMV_E8_AVAILABLE and GEMV_MP_AVAILABLE and GEMV_E8P_AVAILABLE
            and GEMV_TCQ_AVAILABLE):
        raise SystemExit("triton unavailable")

    gs = args.group_size
    cb = e8_codebook().to(torch.float16)
    cb32 = cb.to(torch.float32)
    # Source table for the generated arm: 256 x 8, the shortest E8 points again.
    src = e8_codebook(n_codewords=256).to(torch.float16)
    src32 = src.to(torch.float32)
    props = torch.cuda.get_device_properties(0)
    print("# Lattice vs scalar stage: decode cost at matched bandwidth\n")
    print(f"# 2 bits/coordinate in every arm; group_size={gs}.")
    print(f"# stored table {tuple(cb.shape)} = {cb.numel()*2/2**20:.2f} MiB; "
          f"generated source {tuple(src.shape)} = {src.numel()*2/2**10:.1f} KiB.")
    print(f"# {props.name}, {props.shared_memory_per_block // 1024} KiB shared "
          "memory per block: only the generated table can be resident, and the "
          "computed code needs no table at all.\n")
    # The real E8P arm: the format `vq.e8p_codec` actually emits, so this row is
    # about E8P rather than about something with E8P's shape. It pays the parity
    # popcount and the +/-1/4 shift the generated arm skips, and its kappa is
    # folded into the source table at launch rather than applied per element.
    real_codec = E8PCodec(d=gs, seed=0, device=DEV) if GEMV_E8P_REAL_AVAILABLE else None
    if real_codec is not None:
        real_src = real_codec.S.to(torch.float32).contiguous()
        real_par = real_codec.par.to(torch.int32).contiguous()
        print(f"# real E8P: kappa={real_codec.scale:.8f}, source "
              f"{tuple(real_src.shape)} fp32 = {real_src.numel()*4/2**10:.1f} KiB.\n")

    print("| layer | N | K | scalar 2-bit (ms) | E8 stored (ms) | ratio | "
          "E8 generated (ms) | ratio | E8P real (ms) | ratio | TCQ 1MAD (ms) | "
          "ratio | max abs err (stored / gen / real / tcq) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    # Pre-build every (kernel, shape) pair so no timed block pays JIT.
    prebuilt = []
    for name, N, K in SHAPES:
        wpr = K // 16
        rx = torch.randn(K, device=DEV, dtype=torch.float32)
        scale = torch.rand(N, K // gs, device=DEV, dtype=torch.float32) + 0.5
        words = torch.randint(-(2**31), 2**31 - 1, (N, wpr), device=DEV,
                              dtype=torch.int32)
        levels = torch.tensor([-1.5, -0.5, 0.5, 1.5], device=DEV,
                              dtype=torch.float32)
        prebuilt.append((name, N, K, rx, scale, words, levels))
        triton_gemv_mp(rx.unsqueeze(0).contiguous(), words.unsqueeze(0).contiguous(),
                       scale.unsqueeze(0).contiguous(), levels, gs, 2)
        triton_gemv_e8(rx, words, scale, cb, gs)
        triton_gemv_e8p(rx, words, scale, src, gs)
        triton_gemv_tcq(rx, words, scale, gs)
        if real_codec is not None:
            triton_gemv_e8p_real(rx, words, scale, real_src, real_par, gs,
                                 codec_scale=real_codec.scale)
    torch.cuda.synchronize()

    for name, N, K, rx, scale, words, levels in prebuilt:
        words_per_row = K // 16  # identical for both arms
        rx_mp = rx.unsqueeze(0).contiguous()
        words_mp = words.unsqueeze(0).contiguous()
        scale_mp = scale.unsqueeze(0).contiguous()

        t_scalar = bench(lambda: triton_gemv_mp(rx_mp, words_mp, scale_mp,
                                                levels, gs, 2), args.iters)
        t_e8 = bench(lambda: triton_gemv_e8(rx, words, scale, cb, gs),
                     args.iters)
        t_e8p = bench(lambda: triton_gemv_e8p(rx, words, scale, src, gs),
                      args.iters)
        t_tcq = bench(lambda: triton_gemv_tcq(rx, words, scale, gs), args.iters)
        t_real = (bench(lambda: triton_gemv_e8p_real(
            rx, words, scale, real_src, real_par, gs,
            codec_scale=real_codec.scale), args.iters)
            if real_codec is not None else float("nan"))

        codes = torch.stack([(words >> 0) & 0xFFFF, (words >> 16) & 0xFFFF],
                            dim=2).reshape(N, -1).long()
        sc = scale.repeat_interleave(gs, dim=1)[:, :K]

        got = triton_gemv_e8(rx, words, scale, cb, gs)
        ref = (cb32[codes].reshape(N, -1) * sc) @ rx
        err = (got.squeeze(0) - ref).abs().max().item()

        # generated arm: 8 bits index, 8 bits of per-coordinate sign
        gidx = codes & 0xFF
        gsgn = (codes >> 8) & 0xFF
        bits = (gsgn.unsqueeze(-1) >> torch.arange(8, device=DEV)) & 1
        wgen = (src32[gidx] * (1.0 - 2.0 * bits.float())).reshape(N, -1)
        got_p = triton_gemv_e8p(rx, words, scale, src, gs)
        ref_p = (wgen * sc) @ rx
        err_p = (got_p.squeeze(0) - ref_p).abs().max().item()

        # computed arm: the same words, decoded by arithmetic and nothing else
        got_t = triton_gemv_tcq(rx, words, scale, gs)
        ref_t = (tcq_decode_reference(words) * sc) @ rx
        err_t = (got_t.squeeze(0) - ref_t).abs().max().item()

        # real arm: decode the same words through the shipped codec and compare
        err_r = float("nan")
        if real_codec is not None:
            got_r = triton_gemv_e8p_real(rx, words, scale, real_src, real_par, gs,
                                         codec_scale=real_codec.scale)
            wreal = real_codec.decode(
                codes.to(torch.int32).to(torch.uint16).reshape(N, -1)
            ).reshape(N, -1)
            ref_r = (wreal * sc) @ rx
            err_r = (got_r.squeeze(0) - ref_r).abs().max().item()

        print(f"| {name} | {N} | {K} | {t_scalar:.3f} | {t_e8:.3f} | "
              f"{t_e8/t_scalar:.2f}x | {t_e8p:.3f} | {t_e8p/t_scalar:.2f}x | "
              f"{t_real:.3f} | {t_real/t_scalar:.2f}x | "
              f"{t_tcq:.3f} | {t_tcq/t_scalar:.2f}x | "
              f"{err:.1e} / {err_p:.1e} / {err_r:.1e} / {err_t:.1e} |", flush=True)

    print("\n# All five arms move identical weight bytes; ratios are decode cost "
          "alone.\n# The 'E8P real' column is the format the shipped artifacts use: "
          "8-bit index,\n# 7 stored signs, the 8th recovered from parity, and a "
          "+/-1/4 shift. The\n# 'generated' column is the decode-shape proxy, "
          "which skips the last two.")


if __name__ == "__main__":
    main()
