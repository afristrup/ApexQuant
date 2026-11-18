"""Residual E8 KV cache backend for vLLM."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from functools import lru_cache
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from ..vq.e8p_codec import E8PCodec
from ..kernels.triton_e8_attention import e8_single_query_attention

try:
    import triton
    import triton.language as tl
    E8_TRITON_AVAILABLE = True
except ImportError:
    E8_TRITON_AVAILABLE = False

E8_SCALE_MIN = float(torch.finfo(torch.float8_e4m3fn).tiny)


if E8_TRITON_AVAILABLE:

    @triton.jit
    def _e8_lookup_encode_kernel(x_ptr, scale_ptr, lookup_ptr, source_ptr, parity_ptr,
                                 out_ptr, rows, codec_scale, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        d = tl.arange(0, 8)
        x = tl.load(x_ptr + row * 8 + d).to(tl.float32)
        scale = tl.load(scale_ptr + row).to(tl.float32)
        x = x / scale / codec_scale
        best_cost = tl.full((), float("inf"), tl.float32)
        best_code = tl.zeros((), tl.int32)
        for cbit in range(2):
            shift = 0.25 if cbit == 0 else -0.25
            y = x - shift
            q = tl.floor(tl.abs(y) - 0.5 + 0.5).to(tl.int32)
            q = tl.maximum(0, tl.minimum(3, q))
            pattern = tl.sum(q << (2 * d), axis=0)
            source_index = tl.load(lookup_ptr + pattern).to(tl.int32)
            source = tl.load(source_ptr + source_index * 8 + d).to(tl.float32)
            negative = y < 0
            parity = tl.sum(negative.to(tl.int32), axis=0) & 1
            mismatch = parity != tl.load(parity_ptr + source_index).to(tl.int32)
            flip = tl.argmin(tl.abs(y) * source, axis=0)
            signs = negative ^ (mismatch & (d == flip))
            sign_bits = tl.sum(tl.where(d < 7, signs.to(tl.int32) << d, 0), axis=0)
            reconstructed = tl.where(signs, -source, source) + shift
            cost = tl.sum((x - reconstructed) * (x - reconstructed), axis=0)
            take = cost <= best_cost
            code = source_index | (sign_bits << 8) | (cbit << 15)
            best_cost = tl.where(take, cost, best_cost)
            best_code = tl.where(take, code, best_code)
        tl.store(out_ptr + row, best_code.to(tl.uint16))

    @triton.jit
    def _e8_decode_kernel(codes_ptr, scales_ptr, source_ptr, parity_ptr, out_ptr,
                          codec_scale, slots, heads, groups, total, PASSES: tl.constexpr,
                          HEAD_SIZE: tl.constexpr, BLOCK: tl.constexpr):
        e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = e < total
        d = e % HEAD_SIZE
        x = e // HEAD_SIZE
        kv = x % 2
        x = x // 2
        head = x % heads
        slot = x // heads
        group = d // 8
        coord = d % 8
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for p in tl.static_range(PASSES):
            code_idx = (((p * slots + slot) * heads + head) * 2 + kv) * groups + group
            code = tl.load(codes_ptr + code_idx, mask=mask, other=0).to(tl.int32)
            source = code & 0xFF
            bit = (code >> (8 + coord)) & 1
            explicit = coord < 7
            signs = tl.zeros((BLOCK,), dtype=tl.int32)
            for b in tl.static_range(7):
                signs += ((code >> (8 + b)) & 1)
            last = (tl.load(parity_ptr + source, mask=mask, other=0) - signs) & 1
            negative = tl.where(explicit, bit, last)
            sign = tl.where(negative == 1, -1.0, 1.0)
            shift = tl.where(((code >> 15) & 1) == 0, 0.25, -0.25)
            magnitude = tl.load(source_ptr + source * 8 + coord, mask=mask, other=0.0)
            scale_idx = code_idx
            scale = tl.load(scales_ptr + scale_idx, mask=mask, other=0.0).to(tl.float32)
            acc += (sign * magnitude + shift) * scale * codec_scale
        tl.store(out_ptr + e, acc, mask=mask)

from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import AttentionBackend, AttentionImpl, AttentionType, MultipleOf
from vllm.v1.attention.backends.turboquant_attn import TurboQuantMetadata, TurboQuantMetadataBuilder
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout


@lru_cache(maxsize=8)
def _tables(device: str) -> tuple[E8PCodec, torch.Tensor]:
    codec = E8PCodec(8, n_probe=20000, seed=0, device=device)
    levels = torch.tensor([0.5, 1.5, 2.5, 3.5], device=device)
    patterns = torch.arange(1 << 16, device=device, dtype=torch.int64)
    digits = torch.stack([(patterns >> (2 * i)) & 3 for i in range(8)], dim=1)
    magnitudes = levels[digits].float()
    nearest = []
    for chunk in magnitudes.split(4096):
        distance = (chunk[:, None, :] - codec.S[None, :, :]).square().sum(-1)
        nearest.append(distance.argmin(1))
    return codec, torch.cat(nearest).contiguous()


def _encode_reference(x: torch.Tensor, passes: int) -> tuple[torch.Tensor, torch.Tensor]:
    codec, lookup = _tables(str(x.device))
    groups = x.reshape(*x.shape[:-1], -1, 8).float()
    residual = groups
    codes = []
    scales = []
    dims = torch.arange(8, device=x.device)
    def branch(normalized: torch.Tensor, shift: float, cbit: int):
        y = normalized - shift
        quantized = y.abs().sub(0.5).round().clamp(0, 3).to(torch.int64)
        pattern = sum(quantized[..., i] << (2 * i) for i in range(8))
        source_index = lookup[pattern]
        source = codec.S[source_index]
        negative = y < 0
        parity = negative.sum(-1).to(torch.int32) & 1
        mismatch = parity != codec.par[source_index]
        flip_index = (y.abs() * source).argmin(-1)
        flip = mismatch.unsqueeze(-1) & (dims == flip_index.unsqueeze(-1))
        signs = negative ^ flip
        sign_bits = sum(signs[..., i].to(torch.int32) << i for i in range(7))
        code = (source_index.to(torch.int32) | (sign_bits << 8) |
                (cbit << 15)).to(torch.uint16)
        reconstructed = torch.where(signs, -source, source) + shift
        cost = (normalized - reconstructed).square().sum(-1)
        return code, cost
    for _ in range(passes):
        scale = residual.square().sum(-1).sqrt().clamp_min(1e-8)
        scale = scale.clamp_min(E8_SCALE_MIN).to(torch.float8_e4m3fn)
        normalized = residual / scale.float().unsqueeze(-1) / codec.scale
        code_pos, cost_pos = branch(normalized, 0.25, 0)
        code_neg, cost_neg = branch(normalized, -0.25, 1)
        code = torch.where((cost_pos <= cost_neg), code_pos, code_neg)
        codes.append(code)
        scales.append(scale)
        reconstruction = codec.decode(code).reshape_as(residual)
        residual = residual - reconstruction * scale.float().unsqueeze(-1)
    return torch.stack(codes), torch.stack(scales)


def _encode(x: torch.Tensor, passes: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not E8_TRITON_AVAILABLE or not x.is_cuda:
        return _encode_reference(x, passes)
    codec, lookup = _tables(str(x.device))
    residual = x.reshape(-1, 8).float().contiguous()
    rows = residual.shape[0]
    codes = []
    scales = []
    for _ in range(passes):
        scale = residual.square().sum(-1).sqrt().clamp_min(1e-8)
        scale = scale.clamp_min(E8_SCALE_MIN).to(torch.float8_e4m3fn)
        scale_f32 = scale.float()
        code = torch.empty(rows, device=x.device, dtype=torch.uint16)
        _e8_lookup_encode_kernel[(rows,)](
            residual, scale_f32, lookup, codec.S, codec.par, code, rows,
            codec.scale, BLOCK=8, num_warps=4,
        )
        codes.append(code)
        scales.append(scale)
        reconstruction = codec.decode(code).reshape_as(residual)
        residual = residual - reconstruction * scale_f32.unsqueeze(-1)
    shape = (*x.shape[:-1], x.shape[-1] // 8)
    return torch.stack(codes).reshape(passes, *shape), torch.stack(scales).reshape(passes, *shape)


class ApexE8ResidualAttentionBackend(AttentionBackend):
    """Paged two-pass E8 residual cache with a fused SDPA attention epilogue."""

    accept_output_buffer = True
    forward_includes_kv_cache_update = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["apexquant_e8p2"]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if spec.state_content_bytes is not None:
            return spec
        passes = int(os.environ.get("APEXQUANT_E8_PASSES", "2"))
        if passes != 2:
            raise ValueError("apexquant_e8p2 requires APEXQUANT_E8_PASSES=2")
        groups = math.ceil(spec.head_size / 8)
        code_bytes = passes * groups * 2
        slot_bytes = 2 * code_bytes + 2 * passes * groups
        return replace(spec, state_content_bytes=(slot_bytes + 15) // 16 * 16)

    @staticmethod
    def get_name() -> str:
        return "APEXQUANT_E8_RESIDUAL"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBNHC,)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "apexquant_e8p2"

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0 and head_size % 8 == 0

    @staticmethod
    def get_impl_cls() -> type["ApexE8ResidualAttentionImpl"]:
        return ApexE8ResidualAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TurboQuantMetadataBuilder]:
        return TurboQuantMetadataBuilder


class ApexE8ResidualAttentionImpl(AttentionImpl[TurboQuantMetadata]):
    """Decode two E8 residuals before the fused PyTorch SDPA epilogue."""

    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None, alibi_slopes=None,
                 sliding_window=None, kv_cache_dtype: str = "apexquant_e8p2",
                 logits_soft_cap=None, attn_type: str = AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, **_: Any) -> None:
        if alibi_slopes is not None:
            raise ValueError("apexquant_e8p2 does not support ALiBi")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.scale = scale
        self.sliding_window = sliding_window
        self.passes = 2
        self.groups = head_size // 8
        self.codec = None

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if torch.cuda.is_current_stream_capturing():
            self._update_fixed_shape(key, value, kv_cache, slot_mapping)
            return
        valid = slot_mapping >= 0
        key = key[valid].view(-1, self.num_kv_heads, self.head_size)
        value = value[valid].view(-1, self.num_kv_heads, self.head_size)
        if key.numel() == 0:
            return
        self._write_cache(key, value, kv_cache, slot_mapping[valid].long())

    def _update_fixed_shape(self, key, value, kv_cache, slot_mapping):
        valid = (slot_mapping >= 0).view(-1, 1, 1)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        key = torch.where(valid, key, torch.zeros_like(key))
        value = torch.where(valid, value, torch.zeros_like(value))
        slots = slot_mapping.clamp_min(0).long()
        self._write_cache(key, value, kv_cache, slots)

    def _write_cache(self, key, value, kv_cache, slots):
        codes, scales = _encode(torch.cat((key, value), dim=0), self.passes)
        kcodes, vcodes = codes[:, :key.shape[0]], codes[:, key.shape[0]:]
        kscales, vscales = scales[:, :key.shape[0]], scales[:, key.shape[0]:]
        cache = kv_cache.transpose(1, 2)
        blocks = slots // cache.shape[1]
        positions = slots % cache.shape[1]
        code_bytes = self.passes * self.groups * 2
        scale_offset = 2 * code_bytes
        cache[blocks, positions, :, :code_bytes] = kcodes.permute(1, 2, 0, 3).reshape(-1, self.num_kv_heads, self.passes * self.groups).view(torch.uint8)
        cache[blocks, positions, :, code_bytes:scale_offset] = vcodes.permute(1, 2, 0, 3).reshape(-1, self.num_kv_heads, self.passes * self.groups).view(torch.uint8)
        scale_bytes = self.passes * self.groups
        cache[blocks, positions, :, scale_offset:scale_offset + scale_bytes] = kscales.permute(1, 2, 0, 3).reshape(-1, self.num_kv_heads, scale_bytes).view(torch.uint8)
        value_offset = scale_offset + scale_bytes
        cache[blocks, positions, :, value_offset:value_offset + scale_bytes] = vscales.permute(1, 2, 0, 3).reshape(-1, self.num_kv_heads, scale_bytes).view(torch.uint8)

    def _decode(self, cache: torch.Tensor, slots: torch.Tensor):
        block_size = cache.shape[1]
        rows = cache[slots // block_size, slots % block_size]
        code_bytes = self.passes * self.groups * 2
        scale_offset = 2 * code_bytes
        kcodes = rows[..., :code_bytes].view(torch.uint16).view(-1, self.num_kv_heads, self.passes, self.groups).permute(2, 0, 1, 3)
        vcodes = rows[..., code_bytes:scale_offset].view(torch.uint16).view(-1, self.num_kv_heads, self.passes, self.groups).permute(2, 0, 1, 3)
        scale_bytes = self.passes * self.groups
        kscales = rows[..., scale_offset:scale_offset + scale_bytes].view(torch.float8_e4m3fn).float().view(-1, self.num_kv_heads, self.passes, self.groups).permute(2, 0, 1, 3)
        value_offset = scale_offset + scale_bytes
        vscales = rows[..., value_offset:value_offset + scale_bytes].view(torch.float8_e4m3fn).float().view(-1, self.num_kv_heads, self.passes, self.groups).permute(2, 0, 1, 3)
        codec, _ = _tables(str(rows.device))
        codes = torch.stack((kcodes, vcodes), dim=3).contiguous()
        scales = torch.stack((kscales, vscales), dim=3).contiguous()
        if E8_TRITON_AVAILABLE and rows.is_cuda:
            total = slots.numel() * self.num_kv_heads * 2 * self.head_size
            values = torch.empty((slots.numel(), self.num_kv_heads, 2, self.head_size),
                                 device=rows.device, dtype=torch.float32)
            _e8_decode_kernel[(triton.cdiv(total, 256),)](
                codes, scales, codec._S_f32, codec._par_i32, values,
                codec.scale, slots.numel(), self.num_kv_heads, self.groups, total,
                PASSES=self.passes, HEAD_SIZE=self.head_size, BLOCK=256,
            )
        else:
            decoded = codec.decode(codes.reshape(-1, self.groups))
            decoded = decoded.reshape(self.passes, -1, self.num_kv_heads, 2, self.head_size)
            expanded = scales.unsqueeze(-1).expand(
                -1, -1, -1, -1, -1, 8).reshape(
                    self.passes, -1, self.num_kv_heads, 2, self.head_size)
            values = (decoded * expanded).sum(0)
        return values[:, :, 0], values[:, :, 1]

    def _forward_capture_single(self, query, output, cache, attn_metadata):
        table = attn_metadata.block_table[0]
        if os.environ.get("APEXQUANT_E8_FUSED_DECODE", "1") == "1":
            codec, _ = _tables(str(query.device))
            fused = e8_single_query_attention(
                query[:1].view(self.num_heads, self.head_size), cache, table,
                attn_metadata.seq_lens[0], codec._S_f32, codec._par_i32,
                self.scale, codec.scale, self.num_kv_heads,
            )
            output[:1] = fused.to(output.dtype).view_as(output[:1])
            return output
        block_size = cache.shape[1]
        length = table.shape[0] * block_size
        positions = torch.arange(length, device=query.device)
        slots = (table[positions // block_size] * block_size
                 + positions % block_size)
        k, v = self._decode(cache, slots)
        groups = self.num_heads // self.num_kv_heads
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, 1).transpose(0, 1).to(query.dtype)
        v = v.repeat_interleave(groups, 1).transpose(0, 1).to(query.dtype)
        valid = positions < attn_metadata.seq_lens[0]
        attn = F.scaled_dot_product_attention(
            query[:1].view(1, self.num_heads, self.head_size).transpose(0, 1).unsqueeze(0),
            k.unsqueeze(0), v.unsqueeze(0),
            attn_mask=valid.view(1, 1, 1, -1),
            dropout_p=0.0, scale=self.scale,
        )
        output[:1] = attn.squeeze(0).transpose(0, 1).reshape_as(output[:1])
        return output

    def _forward_capture(self, query, output, cache, attn_metadata):
        table = attn_metadata.block_table
        block_size = cache.shape[1]
        requests = attn_metadata.query_start_loc.numel() - 1
        max_length = table.shape[1] * block_size
        positions = torch.arange(max_length, device=query.device)
        groups = self.num_heads // self.num_kv_heads
        q = query[:requests].view(requests, self.num_heads, self.head_size)
        for req in range(requests):
            slots = (table[req, positions // block_size] * block_size
                     + positions % block_size)
            k, v = self._decode(cache, slots)
            k = k.repeat_interleave(groups, 1).transpose(0, 1).to(q.dtype)
            v = v.repeat_interleave(groups, 1).transpose(0, 1).to(q.dtype)
            valid = positions < attn_metadata.seq_lens[req]
            attn = F.scaled_dot_product_attention(
                q[req:req + 1].transpose(0, 1).unsqueeze(0),
                k.unsqueeze(0), v.unsqueeze(0),
                attn_mask=valid.view(1, 1, 1, -1),
                dropout_p=0.0, scale=self.scale,
            )
            output[req:req + 1] = attn.squeeze(0).transpose(0, 1)
        return output

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, **_):
        output = torch.empty_like(query) if output is None else output
        if attn_metadata is None:
            return output.zero_()
        if torch.cuda.is_current_stream_capturing():
            cache = kv_cache.transpose(1, 2)
            requests = attn_metadata.query_start_loc.numel() - 1
            if requests == 1:
                return self._forward_capture_single(query, output, cache, attn_metadata)
            return self._forward_capture(query, output, cache, attn_metadata)
        n = attn_metadata.num_actual_tokens
        q = query[:n].view(n, self.num_heads, self.head_size)
        cache = kv_cache.transpose(1, 2)
        starts = attn_metadata.query_start_loc.tolist()
        lengths = attn_metadata.seq_lens.tolist()
        table = attn_metadata.block_table
        block_size = kv_cache.shape[2]
        groups = self.num_heads // self.num_kv_heads
        for req, (start, stop) in enumerate(zip(starts, starts[1:])):
            length = int(lengths[req])
            blocks = table[req, :math.ceil(length / block_size)]
            if stop - start == 1 and os.environ.get("APEXQUANT_E8_FUSED_DECODE", "1") == "1":
                codec, _ = _tables(str(q.device))
                fused = e8_single_query_attention(
                    q[start:stop].view(self.num_heads, self.head_size), cache,
                    blocks, torch.as_tensor(length, device=q.device),
                    codec._S_f32, codec._par_i32, self.scale, codec.scale,
                    self.num_kv_heads,
                )
                output[start:stop] = fused.to(output.dtype).view_as(output[start:stop])
                continue
            positions = torch.arange(length, device=q.device)
            slots = blocks[positions // block_size] * block_size + positions % block_size
            k, v = self._decode(cache, slots)
            k = k.repeat_interleave(groups, 1).transpose(0, 1).to(q.dtype)
            v = v.repeat_interleave(groups, 1).transpose(0, 1).to(q.dtype)
            qq = q[start:stop].transpose(0, 1).unsqueeze(0)
            if stop - start == 1:
                attn = F.scaled_dot_product_attention(
                    qq, k.unsqueeze(0), v.unsqueeze(0),
                    dropout_p=0.0, scale=self.scale,
                )
            else:
                qpos = torch.arange(length - (stop - start), length, device=q.device)
                allowed = positions.unsqueeze(0) <= qpos.unsqueeze(1)
                if self.sliding_window:
                    allowed &= positions.unsqueeze(0) >= qpos.unsqueeze(1) - self.sliding_window + 1
                attn = F.scaled_dot_product_attention(
                    qq, k.unsqueeze(0), v.unsqueeze(0),
                    attn_mask=allowed.unsqueeze(0).unsqueeze(0),
                    dropout_p=0.0, scale=self.scale,
                )
            output[start:stop] = attn.squeeze(0).transpose(0, 1)
        return output
