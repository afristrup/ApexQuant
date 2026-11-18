"""vLLM attention backend for packed ApexQuant Trellis KV cache."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from ..rotation.inverse_random_rotation import inverse_random_rotation
from ..rotation.rotation_matrix import rotation_matrix
from ..vq.progressive_artifact import quantize
from ..vq.trellis_codec import TrellisCodec

try:
    from vllm.config import get_current_vllm_config
    from vllm.config.cache import CacheDType
    from vllm.v1.attention.backend import AttentionBackend, AttentionImpl
    from vllm.v1.attention.backends.turboquant_attn import (
        TurboQuantMetadata,
        TurboQuantMetadataBuilder,
    )
    from vllm.v1.attention.backend import AttentionType, MultipleOf
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout
except ImportError:  # pragma: no cover
    AttentionBackend = object
    AttentionImpl = object
    TurboQuantMetadata = object
    TurboQuantMetadataBuilder = object
    AttentionSpec = object
    KVCacheLayout = object
    CacheDType = str


class ApexTrellisAttentionBackend(AttentionBackend):
    """Paged vLLM backend whose slots contain packed Trellis K and V."""

    accept_output_buffer = True
    forward_includes_kv_cache_update = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "apexquant_trellis"
    ]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if spec.state_content_bytes is not None:
            return spec
        bits = int(os.environ.get("APEXQUANT_KV_BITS", "2"))
        passes = int(os.environ.get("APEXQUANT_KV_PASSES", "1"))
        head_size = spec.head_size
        code_bytes = head_size * bits // 8
        slot_bytes = passes * (2 * code_bytes + 4)
        return replace(spec, state_content_bytes=((slot_bytes + 15) // 16) * 16)

    @staticmethod
    def get_name() -> str:
        return "APEXQUANT_TRELLIS"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBNHC,)

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec=None,
    ) -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type["ApexTrellisAttentionImpl"]:
        return ApexTrellisAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TurboQuantMetadataBuilder]:
        return TurboQuantMetadataBuilder

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "apexquant_trellis"

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0 and head_size % 8 == 0

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True


class ApexTrellisAttentionImpl(AttentionImpl[TurboQuantMetadata]):
    """Correctness-first paged Trellis attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype: str = "apexquant_trellis",
        logits_soft_cap=None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        **_: Any,
    ) -> None:
        if alibi_slopes is not None:
            raise ValueError("ApexQuant Trellis attention does not support ALiBi")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.scale = scale
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.passes = int(os.environ.get("APEXQUANT_KV_PASSES", "1"))
        self.bits = int(os.environ.get("APEXQUANT_KV_BITS", "2"))
        self.seed = int(os.environ.get("APEXQUANT_KV_SEED", "42"))
        if self.passes <= 0 or self.bits not in (2, 4, 8):
            raise ValueError("invalid ApexQuant KV configuration")
        self.codec = None
        self.rotations = None

    def _ensure_device(self, device: torch.device) -> None:
        if self.codec is not None and self.codec.device == device:
            return
        self.codec = TrellisCodec(
            self.head_size, bits=self.bits, trellis_bits=12,
            seed=self.seed, device=device,
        )
        if self.head_size <= 256:
            self.rotations = torch.stack([
                rotation_matrix(self.head_size, self.seed + p, str(device))
                for p in range(self.passes)
            ])

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if torch.cuda.is_current_stream_capturing():
            self._update_fixed_shape(key, value, kv_cache, slot_mapping)
            return
        valid = slot_mapping >= 0
        self._ensure_device(key.device)
        key = key[valid].view(-1, self.num_kv_heads, self.head_size)
        value = value[valid].view(-1, self.num_kv_heads, self.head_size)
        if key.numel() == 0:
            return
        self._write_cache(key, value, kv_cache, slot_mapping[valid].long())

    def _update_fixed_shape(self, key, value, kv_cache, slot_mapping):
        self._ensure_device(key.device)
        valid = (slot_mapping >= 0).view(-1, 1, 1)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        key = torch.where(valid, key, torch.zeros_like(key))
        value = torch.where(valid, value, torch.zeros_like(value))
        slots = slot_mapping.clamp_min(0).long()
        self._write_cache(key, value, kv_cache, slots)

    def _write_cache(self, key, value, kv_cache, slots):
        n_tokens = key.shape[0]
        flat = torch.cat((key, value), dim=0).reshape(-1, self.head_size).float()
        code_bytes = self.head_size * self.bits // 8
        cache = kv_cache.transpose(1, 2)
        blocks = slots // cache.shape[1]
        positions = slots % cache.shape[1]
        chunk = int(os.environ.get("APEXQUANT_KV_CHUNK", "256"))
        for begin in range(0, n_tokens, chunk):
            end = min(begin + chunk, n_tokens)
            rows = flat[begin * 2 * self.num_kv_heads:end * 2 * self.num_kv_heads]
            codes, scales = quantize(
                rows, self.codec, self.passes, self.head_size, self.seed
            )
            size = end - begin
            codes = codes.view(self.passes, 2, size, self.num_kv_heads, code_bytes)
            scales = scales.view(self.passes, 2, size, self.num_kv_heads)
            target_blocks = blocks[begin:end]
            target_positions = positions[begin:end]
            cache[target_blocks, target_positions, :, : self.passes * code_bytes] = codes[:, 0].permute(1, 2, 0, 3).reshape(size, self.num_kv_heads, -1)
            v0 = self.passes * code_bytes
            cache[target_blocks, target_positions, :, v0 : v0 + self.passes * code_bytes] = codes[:, 1].permute(1, 2, 0, 3).reshape(size, self.num_kv_heads, -1)
            s0 = v0 + self.passes * code_bytes
            cache[target_blocks, target_positions, :, s0 : s0 + 4 * self.passes] = torch.cat(
                (scales[:, 0].permute(1, 2, 0), scales[:, 1].permute(1, 2, 0)), dim=-1
            ).view(size, self.num_kv_heads, -1).view(torch.uint8)

    def _decode(self, cache: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_device(cache.device)
        block_size = cache.shape[1]
        rows = cache[slots // block_size, slots % block_size]
        code_bytes = self.head_size * self.bits // 8
        k0 = rows[..., : self.passes * code_bytes]
        v0 = self.passes * code_bytes
        v1 = rows[..., v0 : v0 + self.passes * code_bytes]
        s0 = v0 + self.passes * code_bytes
        scales = rows[..., s0 : s0 + 4 * self.passes].contiguous().view(torch.float16).view(
            -1, self.num_kv_heads, 2 * self.passes
        )
        codes = torch.stack((k0, v1), dim=1)
        codes = codes.view(-1, 2, self.num_kv_heads, self.passes, code_bytes)
        codes = codes.permute(3, 0, 2, 1, 4)
        vals = self.codec.decode(codes.reshape(self.passes, -1, code_bytes))
        vals = vals.view(self.passes, -1, self.num_kv_heads, 2, self.head_size)
        scales = scales.view(-1, self.num_kv_heads, 2, self.passes)
        scales = scales.permute(3, 0, 1, 2).unsqueeze(-1)
        vals = vals * scales
        if self.rotations is not None:
            vals = torch.einsum("pskhd,pfd->pskhf", vals, self.rotations)
        else:
            vals = torch.stack([
                inverse_random_rotation(vals[p], self.seed + p)
                for p in range(self.passes)
            ])
        vals = vals.sum(0)
        return vals[:, :, 0], vals[:, :, 1]

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

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, **_):
        if output is None:
            output = torch.empty_like(query)
        if attn_metadata is None:
            return output.zero_()
        if torch.cuda.is_current_stream_capturing():
            return self._forward_capture(query, output, kv_cache.transpose(1, 2),
                                         attn_metadata)
        n = attn_metadata.num_actual_tokens
        cache = kv_cache.transpose(1, 2)
        q = query[:n].view(n, self.num_heads, self.head_size)
        qsl = attn_metadata.query_start_loc.tolist()
        seq_lens = attn_metadata.seq_lens.tolist()
        block_table = attn_metadata.block_table
        block_size = kv_cache.shape[2]
        for req, (start, stop) in enumerate(zip(qsl, qsl[1:])):
            seq_len = int(seq_lens[req])
            positions = torch.arange(seq_len, device=q.device)
            blocks = block_table[req, : math.ceil(seq_len / block_size)]
            slots = blocks[positions // block_size] * block_size + positions % block_size
            k, v = self._decode(cache, slots)
            groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(groups, 1).transpose(0, 1).to(q.dtype)
            v = v.repeat_interleave(groups, 1).transpose(0, 1).to(q.dtype)
            query = q[start:stop].transpose(0, 1).unsqueeze(0)
            if stop - start == 1:
                attn = F.scaled_dot_product_attention(
                    query, k.unsqueeze(0), v.unsqueeze(0),
                    dropout_p=0.0, scale=self.scale,
                )
            else:
                key_positions = torch.arange(seq_len, device=q.device)
                query_positions = torch.arange(
                    seq_len - (stop - start), seq_len, device=q.device
                )
                allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                if self.sliding_window:
                    allowed &= key_positions.unsqueeze(0) >= (
                        query_positions.unsqueeze(1) - self.sliding_window + 1
                    )
                attn = F.scaled_dot_product_attention(
                    query, k.unsqueeze(0), v.unsqueeze(0),
                    attn_mask=allowed.unsqueeze(0).unsqueeze(0),
                    dropout_p=0.0, scale=self.scale,
                )
            output[start:stop] = attn.squeeze(0).transpose(0, 1)
        return output
