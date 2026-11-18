"""Packed routed-expert GEMV for nested trellis weights."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

from ..kernels.triton_gemv_trellis import check_window_table


def fused_moe_trellis(rx, expert_ids, words, scale, table, group_size, bits, trellis_bits):
    """Decode routed experts, returning ``(passes, routes, out_features)``."""
    if triton is None:
        raise RuntimeError("triton is not available")
    if rx.ndim != 3 or expert_ids.ndim != 1 or words.ndim != 4 or scale.ndim != 4:
        raise ValueError("invalid routed trellis tensor ranks")
    check_window_table(table, trellis_bits)
    passes, routes, width = rx.shape
    _, _, out_features, words_per_row = words.shape
    if expert_ids.numel() != routes or words.shape[:3] != scale.shape[:3]:
        raise ValueError("routed tensor shapes disagree")
    vpw = 32 // bits
    if group_size % vpw or words_per_row * vpw != width:
        raise ValueError("group_size and packed width do not match")
    if trellis_bits % bits or trellis_bits >= 32:
        raise ValueError("trellis_bits must be a multiple of bits and below 32")

    wpg = group_size // vpw
    words_per_chunk = wpg
    parts = torch.empty((passes, routes, out_features), device=rx.device, dtype=torch.float32)

    @triton.jit
    def kernel(rx_ptr, ids_ptr, words_ptr, scale_ptr, table_ptr, out_ptr,
               width, words_per_row, words_per_chunk, routes, out_features,
               srp, srr, srk, sid, swp, swe, swn, ssp, sse, ssn, ssg,
               GROUP_SIZE: tl.constexpr, WORDS_PER_GROUP: tl.constexpr,
               BITS: tl.constexpr, L: tl.constexpr, VPW: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_z = tl.program_id(1)
        p = pid_z // routes
        route = pid_z % routes
        expert = tl.load(ids_ptr + route * sid)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < out_features
        wbase = words_ptr + p * swp + expert * swe + offs_n[:, None] * swn
        sbase = scale_ptr + p * ssp + expert * sse + offs_n[:, None] * ssn
        rbase = rx_ptr + p * srp + route * srr
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for w0 in range(0, words_per_row, words_per_chunk):
            offs_w = w0 + tl.arange(0, BLOCK_W)
            wmask = offs_w < tl.minimum(w0 + words_per_chunk, words_per_row)
            cur = tl.load(wbase + offs_w[None, :], mask=nmask[:, None] & wmask[None, :], other=0)
            prev = tl.load(wbase + offs_w[None, :] - 1,
                           mask=nmask[:, None] & wmask[None, :] & ((offs_w % WORDS_PER_GROUP) != 0), other=0)
            wsum = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.float32)
            for v in tl.static_range(VPW):
                lo = BITS * v - (L - BITS)
                if lo >= 0:
                    win = (cur >> lo) & ((1 << L) - 1)
                else:
                    top = -lo
                    win = (((prev >> (32 + lo)) & ((1 << top) - 1)) |
                           ((cur & ((1 << (L - top)) - 1)) << top))
                code = tl.load(table_ptr + win)
                k = offs_w * VPW + v
                value = tl.load(rbase + k * srk, mask=k < width, other=0.0)
                wsum += code * value[None, :]
            group = (offs_w * VPW) // GROUP_SIZE
            sc = tl.load(sbase + group[None, :] * ssg, mask=nmask[:, None] & wmask[None, :], other=0.0)
            acc += tl.sum(wsum * sc, axis=1)
        tl.store(out_ptr + (p * routes + route) * out_features + offs_n, acc, mask=nmask)

    grid = (triton.cdiv(out_features, 64), passes * routes)
    kernel[grid](rx, expert_ids, words, scale, table, parts, width, words_per_row, words_per_chunk, routes, out_features,
                 rx.stride(0), rx.stride(1), rx.stride(2), expert_ids.stride(0), words.stride(0), words.stride(1), words.stride(2),
                 scale.stride(0), scale.stride(1), scale.stride(2), scale.stride(3), GROUP_SIZE=group_size, WORDS_PER_GROUP=wpg,
                 BITS=bits, L=trellis_bits, VPW=vpw, BLOCK_N=64, BLOCK_W=64)
    return parts
