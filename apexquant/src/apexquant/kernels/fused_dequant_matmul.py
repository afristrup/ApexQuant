"""Dispatch fused dequant+matmul to the best available backend."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from ..quantize.quantized_weight import QuantizedWeight
from ..quantize.require_unguarded import require_unguarded
from ..rotation.random_rotation import random_rotation
from ..rotation.rotation_matrix import MATMUL_ROTATION_MAX_DIM, rotation_matrix
from .gemlite_splitk_nonuniform import GEMLITE_SPLITK_AVAILABLE, gemlite_splitk_nonuniform
from .torch_fused_dequant_matmul import torch_fused_dequant_matmul
from .triton_fused_dequant_matmul import TRITON_AVAILABLE, triton_fused_dequant_matmul
from .triton_gemm_mid import GEMM_MID_AVAILABLE, triton_gemm_mid
from .triton_gemv_block import GEMV_BLOCK_AVAILABLE, triton_gemv_block
from .triton_gemv_dequant_matmul import GEMV_TRITON_AVAILABLE, triton_gemv_dequant_matmul
from .triton_gemv_mp import GEMV_MP_AVAILABLE, triton_gemv_mp
from .triton_gemv_mp_lop3 import GEMV_MP_LOP3_AVAILABLE, triton_gemv_mp_lop3


# Triton's `tl.dot` needs a contraction dim >= 16; the tile's BLOCK_K is the
# group size, so a group smaller than this must use the torch reference. Real EO
# layers are far wider (Clay's narrowest is the 64-wide patch proj), so this only
# guards full-coverage fusion of unusually narrow layers.
_MIN_KERNEL_GROUP = 16


def _use_gemlite_fork(qw: QuantizedWeight, rows: int) -> bool:
    """Route 2-bit multi-row forwards to the vendored gemlite split-K GEMM.

    Measured policy (graph-captured, 3-pass, scratchpad gl_fork_dispatch_bench):
    the tensor-core fork amortises the decode across rows and wins 1.1-3.4x over
    the block mat-vec / mid-M GEMM on layers from Clay's MLP up (>= 2^22 weights)
    at M >= 4, and from ~2x that size already at M = 3. It loses at M = 2 (tile
    padding), on tiny layers (kernel-launch + host pass-sum overhead), and is
    unmeasured past M = 64, where the mid-M GEMM keeps the regime it was tuned
    for. The 4-level decode is 2-bit only.
    """
    if not (GEMLITE_SPLITK_AVAILABLE and qw.bits == 2 and rows <= 64):
        return False
    weight_numel = qw.out_features * qw.num_groups * qw.group_size
    return (rows >= 4 and weight_numel >= 1 << 22) or (rows >= 3 and weight_numel >= 1 << 23)


@functools.lru_cache(maxsize=None)
def _stacked_rotations(gs: int, seed: int, passes: int, device: str) -> torch.Tensor:
    """(P, gs, gs) float32 rotation matrices for passes seed..seed+P-1, cached --
    re-stacking per call costs an alloc+copy that shows up at decode batch-1."""
    return torch.stack([rotation_matrix(gs, seed + p, device)
                        for p in range(passes)]).contiguous()


def _select_backend(x: torch.Tensor, group_size: int, force_backend: str | None) -> str:
    """Pick a per-pass GEMM backend: Triton > torch."""
    if force_backend is not None:
        return force_backend
    if (x.is_cuda and x.dtype == torch.float32 and group_size >= _MIN_KERNEL_GROUP
            and TRITON_AVAILABLE):
        return "triton"
    return "torch"


def fused_dequant_matmul(x: torch.Tensor, qw: QuantizedWeight,
                         force_backend: str | None = None) -> torch.Tensor:
    """Compute ``x @ dequantize_weight(qw).T`` via the fastest safe backend.

    Backend order on CUDA (float32): Triton -> torch. Off CUDA, or for any other
    dtype, the pure-torch reference runs. Both backends produce identical results
    up to floating point. Set ``force_backend`` to ``"torch"`` or ``"triton"`` to
    override (tests/benchmarks).

    Args:
        x: Activations ``(..., in_features)``.
        qw: Quantized weight.
        force_backend: Optional backend override.

    Returns:
        Output ``(..., out_features)``.
    """
    require_unguarded(qw)
    backend = _select_backend(x, qw.group_size, force_backend)
    if backend == "torch":
        return torch_fused_dequant_matmul(x, qw)
    gemm = triton_fused_dequant_matmul

    if qw.device != x.device:  # keep packed indices/scales/codebook beside x
        qw = qw.to(x.device)
    # Rotate the input once per pass in torch; the GEMM reconstructs the 4-bit
    # weight inside its tile loop. packed is flattened (out, groups, gs//2) ->
    # (out, K//2) so the kernel's byte-column stride is 1.
    lead = x.shape[:-1]
    gs = qw.group_size
    x2 = x.reshape(-1, x.shape[-1])
    pad = qw.num_groups * gs - x2.shape[-1]
    xg = (x2 if pad == 0 else F.pad(x2, (0, pad))).view(-1, qw.num_groups, gs)

    # Batch-1 multi-pass decode: fold ALL residual passes and a K-split into ONE
    # launch -- rotate every pass with a single stacked matmul, then one kernel
    # accumulates the P partial GEMVs via fp32 atomics. This is what keeps
    # multi-pass decode competitive (the per-pass path pays P rotations, P
    # launches and P output round-trips per layer) and what fills the GPU on
    # small-N/large-K down-projections. At 2 bits the kernel is the LOP3
    # packed-dot variant: the per-value decode loop collapses to PTX that
    # dequantises a whole word and dots it in packed fp16, which removes most
    # of the measured decode-instruction tax (per-token stack 10.6 -> 5.5 ms
    # eager, 10.4 -> 4.8 ms graphed at 3 passes; never slower at 1 pass, so
    # single-pass 2-bit routes here too). 4-bit multi-pass keeps the gather
    # kernel; 4-bit single pass stays on the plain GEMV below. All paths take
    # the cached kernel-ready views -- rebuilding them per forward is host
    # overhead that dominates small layers at batch one.
    use_lop3 = qw.bits == 2 and gs % 16 == 0 and GEMV_MP_LOP3_AVAILABLE
    if (backend == "triton" and x2.shape[0] == 1
            and not qw.graph_safe
            and (use_lop3 or (qw.passes > 1 and GEMV_MP_AVAILABLE))):
        P = qw.passes
        words, scale, levels = qw.decode_views()
        if gs <= MATMUL_ROTATION_MAX_DIM:
            R = qw.rotations if qw.rotations is not None else _stacked_rotations(
                gs, qw.seed, P, str(x.device)
            )
            rx = torch.einsum("ng,pgh->pnh", xg[0], R).reshape(P, -1).contiguous()
        else:
            rx = torch.stack([
                random_rotation(xg, qw.seed + p).reshape(-1) for p in range(P)
            ]).contiguous()
        if use_lop3:
            out = triton_gemv_mp_lop3(rx, qw)
        else:
            out = triton_gemv_mp(rx, words, scale, levels, gs, qw.bits)
        return out.view(*lead, qw.out_features)

    # Multi-row forwards. A handful of rows (2-8: the speculative-verify shape,
    # tiny micro-batches) takes the block mat-vec, which keeps mat-vec memory
    # behaviour by serving repeated weight reads from L2. More rows take the
    # all-passes split-K GEMM, which beats the per-pass GEMM 2-8x at Clay's
    # 65-token regime and the torch/cuBLAS reference ~2x even at thousands of
    # rows (tf32 tensor cores). Group-aligned tiles need the group size to be a
    # multiple of 64; narrower groups keep the general kernel below.
    if (backend == "triton" and x2.shape[0] > 1
            and gs % 64 == 0 and GEMM_MID_AVAILABLE and GEMV_BLOCK_AVAILABLE
            and not qw.graph_safe):
        P = qw.passes
        M = x2.shape[0]
        words, scale, levels = qw.decode_views()
        if gs <= MATMUL_ROTATION_MAX_DIM:
            R = qw.rotations if qw.rotations is not None else _stacked_rotations(
                gs, qw.seed, P, str(x.device)
            )
            rx = torch.einsum("mng,pgh->pmnh", xg, R).reshape(P, M, -1).contiguous()
        else:
            rx = torch.stack([
                random_rotation(xg, qw.seed + p).reshape(M, -1)
                for p in range(P)
            ]).contiguous()
        if _use_gemlite_fork(qw, M):
            out = gemlite_splitk_nonuniform(rx, qw)
        else:
            kernel = triton_gemv_block if M <= 8 else triton_gemm_mid
            out = kernel(rx, words, scale, levels, gs, qw.bits)
        return out.view(*lead, qw.out_features)

    # Batch-1 single pass: the dedicated mat-vec (no atomics, minimal overhead).
    if backend == "triton" and x2.shape[0] == 1 and GEMV_TRITON_AVAILABLE:
        words, scale, levels = qw.decode_views()

        def one_pass(p: int) -> torch.Tensor:
            if qw.rotations is None:
                rx = random_rotation(xg, qw.seed + p).reshape(1, -1).contiguous()
            else:
                rx = torch.einsum("mng,gh->mnh", xg, qw.rotations[p].to(x.dtype)).reshape(1, -1)
            return triton_gemv_dequant_matmul(
                rx, words[p].view(torch.uint8), scale[p], levels, gs, qw.bits)

        out = one_pass(0)
        for p in range(1, qw.passes):
            out = out + one_pass(p)
        return out.view(*lead, qw.out_features)

    levels = qw.levels.contiguous()

    def one_pass(p: int) -> torch.Tensor:
        if qw.rotations is None:
            rx = random_rotation(xg, qw.seed + p).reshape(x2.shape[0], -1).contiguous()
        else:
            rx = torch.einsum("mng,gh->mnh", xg, qw.rotations[p].to(x.dtype)).reshape(x2.shape[0], -1)
        packed_2d = qw.packed[p].reshape(qw.out_features, -1).contiguous()
        return gemm(rx, packed_2d, qw.scale[p].contiguous(), levels, gs, qw.bits)

    # Single pass is the common case (4-bit, or 2-bit before residual passes) --
    # return the kernel output directly instead of zeros + accumulate.
    out = one_pass(0)
    for p in range(1, qw.passes):
        out = out + one_pass(p)
    return out.view(*lead, qw.out_features)
