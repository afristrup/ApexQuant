"""Mid-M fused dequant GEMM: gemlite's GEMM_SPLITK forked for our codebook.

The measured story (docs/two_bit_clay_lessons.md sections N and O): at M >= 2 a
tensor-core split-K GEMM amortises the 2-bit decode across rows and beats our
block mat-vec ~2.8x on the speculative-verify shape, but gemlite's stock kernel
is uniform-affine and forcing the density-matched codebook to uniform spacing
(a = 3b) destroys the 2-pass prefix (+371% perplexity on TinyLlama). This module
vendors the one kernel we need with the non-uniform 4-level decode inlined:

    code c in {0,1,2,3} -> level in {-a, -b, +b, +a}
    sign = 2*(c >> 1) - 1;  bit0 ^ bit1 = 1 selects the inner magnitude b,
    applied as straight-line arithmetic (a select-based mux measured
    1.19-1.26x slower at M <= 64; lessons Z).

``a`` and ``b`` enter as runtime scalars (no recompile per layer); the per-group
scale multiplies as in the symmetric grouped mode. Everything else -- tiling,
split-K atomics with an init-to-zero pre-hook, autotune configs -- is reused
from the installed gemlite (pinned 0.5.1.post1) by importing its pruner and
config generators. ``type_id=102`` matches gemlite's fp16/2-bit signature;
the shipped 4090 cache holds no (group-512, 16-per-word) entries, so in
practice every shape autotunes fresh through the configured scope.

Layout notes. Our packed words and gemlite's are bit-identical (value ``k`` sits
in word ``k // 16`` at shift ``2 * (k % 16)``); the only difference is that
gemlite stores B as ``(K // 16, N)`` where we store ``(N, K // 16)``. So the
kernel-ready views are plain transposes of ``decode_views()``, cached on the
QuantizedWeight like the other views. Activations are cast to fp16 (tensor-core
operand); accumulation and output are fp32; expected parity vs the fp32
reference is ~1e-3 relative, the tensor-core noise floor, not the ~1e-6 of the
fp32 mat-vec kernels.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    from gemlite.triton_kernels.config import AUTOTUNE
    from gemlite.triton_kernels.gemm_splitK_kernels import (
        KEYS,
        get_default_config_nvidia,
        get_fast_autotune_config_nvidia,
        get_max_autotune_config_nvidia,
        kernel_config_pruner,
    )
    from gemlite.triton_kernels.utils import get_closest_m, linear_tile, swizzle_tile

    GEMLITE_SPLITK_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEMLITE_SPLITK_AVAILABLE = False


_TYPE_ID_FP16_W2 = 102  # gemlite's input_dtype * 100 + W_nbits for fp16 / 2-bit


def _autotune_configs():
    scope = os.environ.get("FUSED_TURBOQUANT_GL_AUTOTUNE", "fast")
    if scope == "max":
        return get_max_autotune_config_nvidia()
    if scope == "default":
        return get_default_config_nvidia()
    return get_fast_autotune_config_nvidia()


if GEMLITE_SPLITK_AVAILABLE:

    @triton.autotune(
        configs=_autotune_configs(),
        key=KEYS,
        prune_configs_by={"early_config_prune": kernel_config_pruner},
        use_cuda_graph=AUTOTUNE.USE_CUDA_GRAPH,
    )
    @triton.jit
    def _gemm_splitK_nonuniform_kernel(
        a_ptr, b_ptr, c_ptr,
        scales_ptr, zeros_ptr, scales_a_ptr,
        lvl_a, lvl_b,
        M, N, K, M_CLOSEST,
        W_nbits: tl.constexpr,
        group_size: tl.constexpr,
        unpack_mask: tl.constexpr,
        elements_per_sample: tl.constexpr,
        type_id: tl.constexpr,
        a_sizeof: tl.constexpr,
        b_sizeof: tl.constexpr,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_meta_g, stride_meta_n,
        load_scales_as_block,
        input_dtype: tl.constexpr,
        output_dtype: tl.constexpr,
        acc_dtype: tl.constexpr,
        meta_dtype: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr, NUM_STAGES: tl.constexpr,
        A_load_order: tl.constexpr,
        data_contiguous: tl.constexpr,
        meta_evict_policy: tl.constexpr = '',
        atomic_mode: tl.constexpr = 'relaxed',
        a_evict: tl.constexpr = 'evict_last',
        b_evict: tl.constexpr = 'evict_first',
    ):
        """gemlite's gemm_splitK_INT_kernel with the affine dequant replaced by
        the non-uniform symmetric 4-level decode. A: (M, K) fp16. B: packed
        (K // elements_per_sample, N) int32. scales: (K // group_size, N).
        C: (M, N) fp32, atomically accumulated across SPLIT_K programs."""
        pid = tl.program_id(axis=0)
        pid_k = tl.program_id(axis=1)

        if elements_per_sample > 1:
            pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)
        else:
            pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        if data_contiguous:
            offs_bn = offs_n
        else:
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)

        b_ptrs = b_ptr + ((offs_k[:, None] // elements_per_sample) * stride_bk
                          + offs_bn[None, :] * stride_bn)
        q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)[:, None]

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = ((offs_am[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)

        scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n

        stride_mul: tl.constexpr = BLOCK_SIZE_K / group_size
        BLOCK_SIZE_K_U: tl.constexpr = BLOCK_SIZE_K * SPLIT_K
        BLOCK_SIZE_K_P: tl.constexpr = (BLOCK_SIZE_K // elements_per_sample) * SPLIT_K

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(num_pid_k):
            if A_load_order == 0:
                a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

            b = tl.load(b_ptrs, eviction_policy=b_evict)

            if A_load_order == 1:
                a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

            k_m = ((k * SPLIT_K + pid_k) * stride_mul).to(tl.int32)
            scales = tl.load(scales_ptrs + k_m * stride_meta_g,
                             eviction_policy=meta_evict_policy)

            if A_load_order == 2:
                a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

            # Non-uniform 4-level decode: {-a, -b, +b, +a} at codes
            # {0, 1, 2, 3}, as straight-line arithmetic: sign = 2*bit1 - 1
            # and the inner flag u1 = bit0 ^ bit1 pick the level without
            # data-dependent selects (the where-mux this replaces measured
            # 1.19-1.26x slower at M <= 64; lessons Z).
            b = (b >> q_shift) & unpack_mask
            bit1 = (b >> 1) & 1
            u1 = ((b ^ bit1) & 1).to(meta_dtype)
            b = (2 * bit1 - 1).to(meta_dtype) * (lvl_a + u1 * (lvl_b - lvl_a)) * scales

            if A_load_order == 3:
                a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

            acc = tl.dot(a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype)

            a_ptrs += BLOCK_SIZE_K_U * stride_ak
            b_ptrs += BLOCK_SIZE_K_P * stride_bk

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        if SPLIT_K > 1:
            tl.atomic_add(c_ptrs, acc, mask=mask, sem=atomic_mode)
        else:
            tl.store(c_ptrs, acc, mask=mask)


def gemlite_views(qw) -> tuple[list[torch.Tensor], list[torch.Tensor], float, float]:
    """gemlite-layout views ``(words_kn, scales_gn, lvl_a, lvl_b)``, cached.

    ``words_kn[p]`` is ``(K // 16, N)`` int32 (the transpose of our packed rows;
    the in-word bit layout already matches), ``scales_gn[p]`` is ``(G, N)``
    float32. Built once per weight like ``decode_views``.
    """
    views = getattr(qw, "_gemlite_views", None)
    if views is None or views[0][0].device != qw.packed.device:
        words, scale, levels = qw.decode_views()
        words_kn = [words[p].t().contiguous() for p in range(qw.passes)]
        scales_gn = [scale[p].t().contiguous() for p in range(qw.passes)]
        views = (words_kn, scales_gn, float(levels[3]), float(levels[2]))
        object.__setattr__(qw, "_gemlite_views", views)
    return views


def _launch_plane(a: torch.Tensor, w: torch.Tensor, s: torch.Tensor,
                  lvl_a: float, lvl_b: float, group_size: int) -> torch.Tensor:
    """One pass plane: ``c[m, n] = a[m] . dequant(w)[n]``. a (M, K) fp16."""
    M, K = a.shape
    N = w.shape[1]
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        META["SPLIT_K"],
    )
    _gemm_splitK_nonuniform_kernel[grid](
        a, w, c,
        s, s, s,  # zeros_ptr / scales_a_ptr unused, any tensor works
        lvl_a, lvl_b,
        M, N, K, get_closest_m(M),
        2, group_size, 3, 16,
        _TYPE_ID_FP16_W2, a.dtype.itemsize, w.dtype.itemsize,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
        s.stride(0), s.stride(1),
        load_scales_as_block=False,
        input_dtype=tl.float16,
        output_dtype=tl.float32,
        acc_dtype=tl.float32,
        meta_dtype=tl.float32,
        data_contiguous=True,
    )
    return c


def gemlite_splitk_nonuniform(rx: torch.Tensor, qw) -> torch.Tensor:
    """All-passes mid-M forward: ``out[m, n] = sum_p rx[p, m] . w_p[n]``.

    ``rx`` is the per-pass rotated activations ``(P, M, K)`` float32 (cast to
    fp16 here: the tensor-core operand). Each pass runs one split-K kernel into
    its own fp32 buffer (the autotune pre-hook zeroes it, so cross-pass
    accumulation in one buffer is not safe); passes are summed on the host.
    Requires ``qw.bits == 2``.
    """
    if not GEMLITE_SPLITK_AVAILABLE:
        raise RuntimeError("gemlite is not available")

    P, M, K = rx.shape
    words_kn, scales_gn, lvl_a, lvl_b = gemlite_views(qw)
    rx16 = rx.to(torch.float16)

    out = None
    for p in range(P):
        c = _launch_plane(rx16[p], words_kn[p], scales_gn[p], lvl_a, lvl_b, qw.group_size)
        out = c if out is None else out + c
    return out


def gemlite_splitk_nonuniform_tiered(rx: torch.Tensor, qw,
                                     row_passes: tuple[int, ...]) -> torch.Tensor:
    """Mixed-tier batched forward: row ``m`` decodes only its first
    ``row_passes[m]`` passes of the shared artifact.

    Because pass prefixes nest, a heterogeneous batch reads each plane ONCE
    for every row that uses it: the batch's weight traffic is the deepest
    tier's bytes, not the sum over rows. Rows must be sorted by descending
    ``row_passes`` so each plane's rows form a contiguous prefix of the batch
    (the caller sorts and unsorts; the dispatcher-facing experiment shows the
    pattern). Requires ``qw.bits == 2`` and ``max(row_passes) <= qw.passes``.
    """
    if not GEMLITE_SPLITK_AVAILABLE:
        raise RuntimeError("gemlite is not available")
    if list(row_passes) != sorted(row_passes, reverse=True):
        raise ValueError("row_passes must be sorted descending")
    if max(row_passes) > qw.passes:
        raise ValueError("a row requests more passes than the artifact stores")

    P, M, K = rx.shape
    words_kn, scales_gn, lvl_a, lvl_b = gemlite_views(qw)
    N = words_kn[0].shape[1]
    rx16 = rx.to(torch.float16)

    out = torch.zeros((M, N), dtype=torch.float32, device=rx.device)
    for p in range(max(row_passes)):
        m_p = sum(1 for t in row_passes if t > p)  # rows that use plane p
        c = _launch_plane(rx16[p, :m_p], words_kn[p], scales_gn[p],
                          lvl_a, lvl_b, qw.group_size)
        out[:m_p] += c
    return out
