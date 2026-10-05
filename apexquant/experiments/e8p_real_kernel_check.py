"""Verify and time the real-E8P fused mat-vec against the decode-shape proxy.

Why this exists. `triton_gemv_e8p` decodes a code with E8P's *shape* (8-bit
index plus 8 free sign bits) and is explicit that it is not E8P. The paper
therefore could not attribute its 0.9-1.3x decode figure to E8P, and the claim
was weakened to say so. `triton_gemv_e8p_real` decodes the format
`apexquant.vq.e8p_codec` actually emits, so the figure can be measured
about E8P itself.

Two things are checked, in this order, because the second is meaningless without
the first:

1. **Correctness.** The kernel's output must match a dense reference built by
   decoding the same codes through `E8PCodec.decode`. A fast kernel that decodes
   the wrong codeword is worthless, and the parity recovery of coordinate 7 is
   exactly the sort of thing that silently produces plausible garbage.
2. **Decode cost.** Timed against the scalar arm at matched weight bandwidth,
   the way `lattice_decode_cost.py` does it, so a ratio is decode cost alone.

    uv run python experiments/e8p_real_kernel_check.py
"""

from __future__ import annotations

import time

import torch

from apexquant.kernels.triton_gemv_e8p_real import (
    GEMV_E8P_REAL_AVAILABLE,
    triton_gemv_e8p_real,
)
from apexquant.vq.e8p_codec import E8P_DIM, E8PCodec

DEV = torch.device("cuda")
GS = 512


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack uint16 E8P codes two-per-int32-word, matching the kernel's layout."""
    flat = codes.reshape(codes.shape[0], -1).to(torch.int32)
    lo, hi = flat[:, 0::2], flat[:, 1::2]
    return (lo | (hi << 16)).contiguous()


def dense_reference(codec: E8PCodec, codes: torch.Tensor,
                    scale: torch.Tensor, gs: int) -> torch.Tensor:
    """Rebuild the dense weight the kernel should be multiplying by."""
    out_f, ngroups, cpg = codes.shape
    vals = codec.decode(codes).reshape(out_f, ngroups, gs)
    return (vals * scale.unsqueeze(-1)).reshape(out_f, ngroups * gs)


def main() -> None:
    if not GEMV_E8P_REAL_AVAILABLE:
        raise SystemExit("triton unavailable")
    torch.manual_seed(0)
    codec = E8PCodec(d=GS, seed=0, device=DEV)
    parity = codec.par.to(torch.int32).contiguous()
    source = codec.S.to(torch.float32).contiguous()

    print("# real-E8P fused mat-vec: correctness, then decode cost")
    print(f"# source table {tuple(source.shape)} fp32 = "
          f"{source.numel()*4/1024:.1f} KiB resident, parity {parity.numel()} B\n")

    shapes = [("q_proj-like", 4096, 4096), ("down_proj-like", 4096, 14336)]
    print(f"# codec scale kappa = {codec.scale:.8f} (folded into the source "
          f"table and the shift)\n")
    print(f"{'shape':<16}{'N':>6}{'K':>7}{'max rel err':>14}{'kernel ms':>11}"
          f"{'dense ms':>10}{'ratio':>8}  verdict")
    for label, N, K in shapes:
        ngroups = K // GS
        W = (torch.randn(N, K, device=DEV) / 64).float()
        u = W.reshape(N, ngroups, GS)
        sc = u.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        sc = sc.to(torch.float16).to(torch.float32)
        codes = codec.encode((u / sc).reshape(-1, GS))
        codes = codes.reshape(N, ngroups, GS // E8P_DIM)
        scale = sc.squeeze(-1)

        words = pack_codes(codes)
        rx = torch.randn(K, device=DEV)
        Wd = dense_reference(codec, codes, scale, GS)

        got = triton_gemv_e8p_real(rx, words, scale, source, parity, GS,
                                   codec_scale=codec.scale)
        ref = (rx.unsqueeze(0) @ Wd.T)
        rel = ((got - ref).abs().max() / ref.abs().max()).item()
        verdict = "OK" if rel < 1e-3 else "WRONG"

        def timed(fn, iters=50):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t) / iters * 1e3

        t_k = timed(lambda: triton_gemv_e8p_real(rx, words, scale, source,
                                                 parity, GS,
                                                 codec_scale=codec.scale))
        t_d = timed(lambda: rx.unsqueeze(0) @ Wd.T)
        print(f"{label:<16}{N:>6}{K:>7}{rel:>14.2e}{t_k:>11.3f}{t_d:>10.3f}"
              f"{t_k/t_d:>7.2f}x  {verdict}")
        del W, u, codes, words, Wd
        torch.cuda.empty_cache()

    print("\n# NOTE: the dense column is an fp32 reference matmul, not the "
          "scalar 2-bit\n# fused arm. For the paper's decode-cost ratio, run "
          "this alongside\n# experiments/lattice_decode_cost.py on an idle GPU.")


if __name__ == "__main__":
    main()
