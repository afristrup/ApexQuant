"""Fused dequant+matmul as a Triton GEMM (GPU fast-path).

The float weight is never stored or loaded: inside the K-loop the kernel reads a
packed nibble, gathers its Lloyd-Max level from a 16-entry codebook resident in
registers, scales it by the group scale, and accumulates -- so the expensive HBM
traffic is 4 bits/weight instead of 16. Input rotation is done once in torch and
the rotated activations ``rx`` are handed to the kernel, matching
:func:`torch_fused_dequant_matmul` exactly.

Triton is optional; import errors are swallowed and ``TRITON_AVAILABLE`` is False
so the dispatcher can fall back to torch.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    TRITON_AVAILABLE = False

# Largest K-tile the rotated-activation + scale tiles fit in shared memory for the
# 64x64 fp32 output tile used off-Hopper. Empirically 128 fits the ~99 KB Ampere/
# Ada budget and 256 does not; groups wider than this loop over multiple K-tiles.
_MAX_BLOCK_K = 128


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_kernel(
        rx_ptr, packed_ptr, scale_ptr, levels_ptr, out_ptr,
        M, N, K, num_groups,
        stride_rxm, stride_rxk,
        stride_pn, stride_pk,
        stride_sn, stride_sg,
        stride_om, stride_on,
        GROUP_SIZE: tl.constexpr, BITS: tl.constexpr, PER_BYTE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            # Rotated activations tile: (BLOCK_M, BLOCK_K).
            rx = tl.load(
                rx_ptr + offs_m[:, None] * stride_rxm + offs_k[None, :] * stride_rxk,
                mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0,
            )

            # Packed low-bit indices: PER_BYTE per byte; byte column is
            # k // PER_BYTE, sub-index picked by k % PER_BYTE (low bits first).
            byte = tl.load(
                packed_ptr + offs_n[:, None] * stride_pn + (offs_k[None, :] // PER_BYTE) * stride_pk,
                mask=(offs_n[:, None] < N) & k_mask[None, :], other=0,
            ).to(tl.int32)
            sub = offs_k[None, :] % PER_BYTE
            idx = (byte >> (BITS * sub)) & ((1 << BITS) - 1)

            # Dequant = gather from the 2**BITS-entry codebook (fused, not stored).
            code = tl.load(levels_ptr + idx)

            # Per-(row, group) scale; group = k // GROUP_SIZE.
            grp = offs_k[None, :] // GROUP_SIZE
            sc = tl.load(
                scale_ptr + offs_n[:, None] * stride_sn + grp * stride_sg,
                mask=(offs_n[:, None] < N) & k_mask[None, :], other=0.0,
            )
            w = code * sc  # (BLOCK_N, BLOCK_K) reconstructed weight tile

            acc += tl.dot(rx, tl.trans(w))

        out = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_fused_dequant_matmul(rx: torch.Tensor, packed: torch.Tensor,
                                scale: torch.Tensor, levels: torch.Tensor,
                                group_size: int, bits: int = 4) -> torch.Tensor:
    """Run one residual pass on the GPU: ``out[m,n] = sum_k rx[m,k] * w[n,k]``.

    ``w[n,k]`` is reconstructed on the fly from ``packed``/``scale``/``levels``.

    Args:
        rx: Pre-rotated activations ``(M, K)``, contiguous, float32, on CUDA.
        packed: Packed indices ``(N, K * bits // 8)`` uint8 -- must be a true 2-D
            tensor (byte-column stride 1); the caller flattens ``(N, groups, ...)``.
        scale: Per-(row, group) scale ``(N, num_groups)`` float32.
        levels: Codebook ``(2**bits,)`` float32.
        group_size: Coordinates per group.
        bits: Codebook bit-width (2, 4, or 8); ``8 // bits`` indices per byte.

    Returns:
        Output ``(M, N)`` float32.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not available")

    M, K = rx.shape
    N = packed.shape[0]
    num_groups = scale.shape[1]
    out = torch.empty((M, N), dtype=torch.float32, device=rx.device)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    # Hopper (H100/Gefion, sm_90) launch: 8 warps cooperate on each output tile
    # and 3 pipeline stages overlap TMA loads with wgmma. Tune per node with the
    # `benchmark` CLI; these are sensible sm_90 defaults.
    is_hopper = torch.cuda.get_device_capability(rx.device)[0] >= 9
    # BLOCK_K is a GEMM tile, *not* the group size: the kernel resolves each
    # column's group and codebook byte per element (`offs_k // GROUP_SIZE`), so a
    # tile may span or subdivide a group freely. Tying BLOCK_K to GROUP_SIZE would
    # blow the shared-memory budget for wide groups -- a group_size of 256 needs a
    # 256-deep fp32 tile (~192 KB), over the ~99 KB Ampere/Ada limit. Cap it so
    # the rotated-activation tile always fits; wider groups just loop more K-steps.
    block_k = min(group_size, _MAX_BLOCK_K)
    _fused_kernel[grid](
        rx, packed, scale, levels, out,
        M, N, K, num_groups,
        rx.stride(0), rx.stride(1),
        packed.stride(0), packed.stride(1),
        scale.stride(0), scale.stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE=group_size, BITS=bits, PER_BYTE=8 // bits,
        BLOCK_M=128 if is_hopper else 64,
        BLOCK_N=128 if is_hopper else 64,
        BLOCK_K=block_k,
        num_warps=8 if is_hopper else 4,
        num_stages=3 if is_hopper else 2,
    )
    return out
