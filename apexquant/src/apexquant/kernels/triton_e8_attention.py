"""Fused single-query attention over paged E8 residual KV cache."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    E8_ATTENTION_AVAILABLE = True
except ImportError:
    E8_ATTENTION_AVAILABLE = False


_decode_tables = {}


def _get_decode_table(source: torch.Tensor, parity: torch.Tensor,
                      codec_scale: float) -> torch.Tensor:
    key = (source.device.index, source.data_ptr(), float(codec_scale))
    table = _decode_tables.get(key)
    if table is not None:
        return table
    codes = torch.arange(1 << 16, device=source.device, dtype=torch.int32)
    source_index = codes & 0xFF
    coord = torch.arange(8, device=source.device, dtype=torch.int32)
    bits = (codes[:, None] >> (8 + coord[None, :])) & 1
    last = (parity[source_index] - bits[:, :7].sum(1)) & 1
    negative = torch.cat((bits[:, :7], last[:, None]), dim=1).bool()
    magnitude = source[source_index]
    shift = torch.where((codes >> 15) & 1 == 0, 0.25, -0.25)
    table = (torch.where(negative, -magnitude, magnitude) + shift[:, None])
    table = (table * codec_scale).to(torch.float16).contiguous()
    _decode_tables[key] = table
    return table


if E8_ATTENTION_AVAILABLE:

    @triton.jit
    def _e8_attention_tile_kernel(
        q_ptr, cache_ptr, scales_ptr, blocks_ptr, seq_len_ptr, decode_ptr,
        max_ptr, norm_ptr, acc_ptr, scale,
        cache_s0, cache_s1, cache_s2, scales_s0, scales_s1, scales_s2,
        num_kv_heads,
        HEAD_SIZE: tl.constexpr, GROUPS: tl.constexpr, PASSES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, QUERY_GROUPS: tl.constexpr, TILE_SIZE: tl.constexpr,
        QUERY_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, TILES: tl.constexpr,
    ):
        pid = tl.program_id(0)
        kv_head = pid % NUM_KV_HEADS
        tile = pid // NUM_KV_HEADS
        d = tl.arange(0, HEAD_SIZE)
        group_ids = tl.arange(0, GROUPS)
        query_groups = tl.arange(0, QUERY_GROUPS)
        coord8 = tl.arange(0, 8)
        query = tl.load(
            q_ptr + (kv_head * QUERY_GROUPS + query_groups[:, None, None]) * HEAD_SIZE
            + group_ids[None, :, None] * 8 + coord8[None, None, :]
        ).to(tl.float32)
        max_score = tl.full((QUERY_GROUPS,), -float("inf"), tl.float32)
        norm = tl.zeros((QUERY_GROUPS,), tl.float32)
        acc = tl.zeros((QUERY_GROUPS, GROUPS, 8), tl.float32)
        code_words = PASSES * GROUPS
        seq_len = tl.load(seq_len_ptr).to(tl.int32)
        for offset in range(TILE_SIZE):
            pos = tile * TILE_SIZE + offset
            valid = pos < seq_len
            block = tl.load(blocks_ptr + pos // BLOCK_SIZE, mask=valid, other=0).to(tl.int32)
            in_block = pos % BLOCK_SIZE
            row = cache_ptr + block * cache_s0 + in_block * cache_s1 + kv_head * cache_s2
            key = tl.zeros((GROUPS, 8), tl.float32)
            value = tl.zeros((GROUPS, 8), tl.float32)
            for p in range(PASSES):
                for group in range(GROUPS):
                    in_group = group_ids == group
                    code_pos = p * GROUPS + group
                    kcode = tl.load(row + code_pos, mask=valid, other=0).to(tl.int32)
                    vcode = tl.load(row + code_words + code_pos, mask=valid, other=0).to(tl.int32)
                    kdecoded = tl.load(
                        decode_ptr + kcode * 8 + coord8, mask=valid, other=0.0,
                    ).to(tl.float32)
                    vdecoded = tl.load(
                        decode_ptr + vcode * 8 + coord8, mask=valid, other=0.0,
                    ).to(tl.float32)
                    scale_row = scales_ptr + block * scales_s0 + in_block * scales_s1 + kv_head * scales_s2
                    kscale = tl.load(scale_row + p * GROUPS + group, mask=valid, other=0.0).to(tl.float32)
                    vscale = tl.load(scale_row + PASSES * GROUPS + p * GROUPS + group, mask=valid, other=0.0).to(tl.float32)
                    key += tl.where(in_group[:, None], kdecoded[None, :] * kscale, 0.0)
                    value += tl.where(in_group[:, None], vdecoded[None, :] * vscale, 0.0)
            score = tl.sum((query * key[None, :, :]).reshape((QUERY_GROUPS, HEAD_SIZE)), axis=1) * scale
            score = tl.where(valid, score, -float("inf"))
            new_max = tl.maximum(max_score, score)
            old_weight = tl.where(
                new_max == -float("inf"),
                0.0,
                tl.exp2((max_score - new_max) * 1.4426950408889634),
            )
            new_weight = tl.where(
                valid,
                tl.exp2((score - new_max) * 1.4426950408889634),
                0.0,
            )
            norm = norm * old_weight + new_weight
            acc = acc * old_weight[:, None, None] + value[None, :, :] * new_weight[:, None, None]
            max_score = new_max
        base = tile * QUERY_HEADS + kv_head * QUERY_GROUPS
        tl.store(max_ptr + base + query_groups, max_score)
        tl.store(norm_ptr + base + query_groups, norm)
        acc_offsets = (
            (base + query_groups[:, None, None]) * HEAD_SIZE
            + group_ids[None, :, None] * 8 + coord8[None, None, :]
        )
        tl.store(acc_ptr + acc_offsets, acc)



    @triton.jit
    def _e8_decode_kv_kernel(
        cache_ptr, scales_ptr, blocks_ptr, seq_len_ptr, decode_ptr,
        key_ptr, value_ptr,
        cache_s0, cache_s1, cache_s2, scales_s0, scales_s1, scales_s2,
        num_kv_heads,
        HEAD_SIZE: tl.constexpr, GROUPS: tl.constexpr, PASSES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, MAX_TOKENS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        token = pid // num_kv_heads
        kv_head = pid % num_kv_heads
        d = tl.arange(0, HEAD_SIZE)
        group_ids = tl.arange(0, GROUPS)
        coord8 = tl.arange(0, 8)
        valid = token < tl.load(seq_len_ptr).to(tl.int32)
        block = tl.load(blocks_ptr + token // BLOCK_SIZE, mask=valid, other=0).to(tl.int32)
        in_block = token % BLOCK_SIZE
        row = cache_ptr + block * cache_s0 + in_block * cache_s1 + kv_head * cache_s2
        code_words = PASSES * GROUPS
        key = tl.zeros((GROUPS, 8), tl.float32)
        value = tl.zeros((GROUPS, 8), tl.float32)
        for p in range(PASSES):
            for group in range(GROUPS):
                in_group = group_ids == group
                code_pos = p * GROUPS + group
                kcode = tl.load(row + code_pos, mask=valid, other=0).to(tl.int32)
                vcode = tl.load(row + code_words + code_pos, mask=valid, other=0).to(tl.int32)
                kdecoded = tl.load(
                    decode_ptr + kcode * 8 + coord8, mask=valid, other=0.0,
                ).to(tl.float32)
                vdecoded = tl.load(
                    decode_ptr + vcode * 8 + coord8, mask=valid, other=0.0,
                ).to(tl.float32)
                scale_row = scales_ptr + block * scales_s0 + in_block * scales_s1 + kv_head * scales_s2
                kscale = tl.load(
                    scale_row + p * GROUPS + group, mask=valid, other=0.0,
                ).to(tl.float32)
                vscale = tl.load(
                    scale_row + PASSES * GROUPS + p * GROUPS + group,
                    mask=valid, other=0.0,
                ).to(tl.float32)
                key += tl.where(in_group[:, None], kdecoded[None, :] * kscale, 0.0)
                value += tl.where(in_group[:, None], vdecoded[None, :] * vscale, 0.0)
        offset = (kv_head * MAX_TOKENS + token) * HEAD_SIZE + d
        tl.store(key_ptr + offset, key.reshape((HEAD_SIZE,)), mask=valid)
        tl.store(value_ptr + offset, value.reshape((HEAD_SIZE,)), mask=valid)


    @triton.jit
    def _e8_attention_reduce_kernel(max_ptr, norm_ptr, acc_ptr, out_ptr,
                                    HEAD_SIZE: tl.constexpr, TILES: tl.constexpr,
                                    QUERY_HEADS: tl.constexpr):
        head = tl.program_id(0)
        d = tl.arange(0, HEAD_SIZE)
        max_score = tl.full((), -float("inf"), tl.float32)
        norm = tl.zeros((), tl.float32)
        acc = tl.zeros((HEAD_SIZE,), tl.float32)
        for tile in range(TILES):
            pid = tile * QUERY_HEADS + head
            tile_max = tl.load(max_ptr + pid)
            tile_norm = tl.load(norm_ptr + pid)
            tile_acc = tl.load(acc_ptr + pid * HEAD_SIZE + d).to(tl.float32)
            new_max = tl.maximum(max_score, tile_max)
            old_weight = tl.where(
                new_max == -float("inf"),
                0.0,
                tl.exp2((max_score - new_max) * 1.4426950408889634),
            )
            tile_weight = tl.where(
                tile_max == -float("inf"),
                0.0,
                tl.exp2((tile_max - new_max) * 1.4426950408889634),
            )
            norm = norm * old_weight + tile_norm * tile_weight
            acc = acc * old_weight + tile_acc * tile_weight
            max_score = new_max
        tl.store(out_ptr + head * HEAD_SIZE + d, acc / norm)


def e8_single_query_attention(
    query: torch.Tensor, cache: torch.Tensor, block_table: torch.Tensor,
    seq_len: torch.Tensor, source: torch.Tensor, parity: torch.Tensor,
    scale: float, codec_scale: float, num_kv_heads: int,
) -> torch.Tensor:
    """Run fused E8 attention for one query and one request."""
    if not E8_ATTENTION_AVAILABLE or not query.is_cuda:
        raise RuntimeError("fused E8 attention requires CUDA and Triton")
    if query.ndim != 2 or cache.ndim != 4:
        raise ValueError("query must be (heads, head_size), cache must be paged")
    heads, head_size = query.shape
    if head_size % 8 or heads % num_kv_heads:
        raise ValueError("head dimensions must be divisible by eight and GQA-aligned")
    groups = head_size // 8
    output = torch.empty_like(query, dtype=torch.float32)
    code_bytes = 2 * groups * 2
    cache_codes = cache.view(torch.uint16)
    scales = cache[..., code_bytes * 2:].view(torch.float8_e4m3fn)
    decode_table = _get_decode_table(source, parity, codec_scale)
    max_tokens = block_table.shape[0] * cache.shape[1]
    if os.environ.get("APEXQUANT_E8_MMA") == "1":
        cache_codes = cache.view(torch.uint16)
        keys = torch.empty(
            (num_kv_heads, max_tokens, head_size),
            device=query.device, dtype=query.dtype,
        )
        values = torch.empty_like(keys)
        _e8_decode_kv_kernel[(max_tokens * num_kv_heads,)](
            cache_codes, scales, block_table, seq_len, decode_table, keys, values,
            cache_codes.stride(0), cache_codes.stride(1), cache_codes.stride(2),
            scales.stride(0), scales.stride(1), scales.stride(2), num_kv_heads,
            HEAD_SIZE=head_size, GROUPS=groups, PASSES=2,
            BLOCK_SIZE=cache.shape[1], MAX_TOKENS=max_tokens,
            num_warps=int(os.environ.get("APEXQUANT_E8_MMA_WARPS", "1")),
        )
        attention = F.scaled_dot_product_attention(
            query.view(1, heads, 1, head_size),
            keys.unsqueeze(0), values.unsqueeze(0),
            dropout_p=0.0, scale=scale, enable_gqa=True,
        )
        return attention.reshape(heads, head_size).float()
    tile_size = int(os.environ.get("APEXQUANT_E8_TILE_SIZE", "128"))
    if tile_size not in (64, 128, 256, 512, 1024):
        raise ValueError("APEXQUANT_E8_TILE_SIZE must be 64, 128, 256, 512, or 1024")
    tiles = (max_tokens + tile_size - 1) // tile_size
    partial_max = torch.empty((tiles, heads), device=query.device, dtype=torch.float32)
    partial_norm = torch.empty_like(partial_max)
    partial_dtype = (
        torch.float16
        if os.environ.get("APEXQUANT_E8_PARTIAL_FP16", "1") == "1"
        else torch.float32
    )
    partial_acc = torch.empty(
        (tiles, heads, head_size), device=query.device, dtype=partial_dtype,
    )
    warps = int(os.environ.get("APEXQUANT_E8_WARPS", "1"))
    if warps not in (1, 2, 4, 8):
        raise ValueError("APEXQUANT_E8_WARPS must be 1, 2, 4, or 8")
    _e8_attention_tile_kernel[(num_kv_heads * tiles,)](
        query, cache_codes, scales, block_table, seq_len, decode_table,
        partial_max, partial_norm, partial_acc, scale,
        cache_codes.stride(0), cache_codes.stride(1), cache_codes.stride(2),
        scales.stride(0), scales.stride(1), scales.stride(2), num_kv_heads,
        HEAD_SIZE=head_size, GROUPS=groups, PASSES=2,
        BLOCK_SIZE=cache.shape[1], QUERY_GROUPS=heads // num_kv_heads,
        TILE_SIZE=tile_size, QUERY_HEADS=heads, NUM_KV_HEADS=num_kv_heads,
        TILES=tiles, num_warps=warps,
    )
    _e8_attention_reduce_kernel[(heads,)](
        partial_max, partial_norm, partial_acc, output,
        HEAD_SIZE=head_size, TILES=tiles, QUERY_HEADS=heads, num_warps=warps,
    )
    return output
