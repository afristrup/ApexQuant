"""Trellis-compressed KV storage and attention logits."""

from __future__ import annotations

import torch

from ..kernels.triton_gemv_trellis import triton_gemv_trellis, trellis_window_table
from ..rotation.random_rotation import random_rotation
from ..rotation.rotation_matrix import rotation_matrix
from ..vq.progressive_artifact import quantize
from ..rotation.inverse_random_rotation import inverse_random_rotation
from ..vq.trellis_codec import TrellisCodec


class TrellisKVCache:
    """Store K/V rows as packed progressive trellis codes."""

    def __init__(self, layers, batch_size, kv_heads, max_sequence, head_dim,
                 codec: TrellisCodec, *, passes=1, group_size=None, seed=42,
                 device=None):
        if passes <= 0:
            raise ValueError("passes must be positive")
        if group_size is None:
            group_size = head_dim
        if group_size != head_dim:
            raise ValueError("group_size must equal head_dim")
        self.layers, self.batch_size, self.kv_heads = layers, batch_size, kv_heads
        self.max_sequence, self.head_dim = max_sequence, head_dim
        self.codec, self.passes, self.group_size, self.seed = codec, passes, group_size, seed
        shape = (layers, batch_size, kv_heads, passes, max_sequence, head_dim // (8 // codec.bits))
        self.key = torch.empty(shape, dtype=torch.uint8, device=device)
        self.value = torch.empty_like(self.key)
        self.key_scale = torch.empty((layers, batch_size, kv_heads, passes, max_sequence, 1), dtype=torch.float16, device=device)
        self.value_scale = torch.empty_like(self.key_scale)
        self._rotations = (
            torch.stack([
                rotation_matrix(self.head_dim, self.seed + p, str(self.key.device))
                for p in range(self.passes)
            ])
            if self.head_dim <= 256
            else None
        )
        self.length = 0

    @property
    def device(self):
        return self.key.device

    def reset(self):
        self.length = 0

    def update(self, layer, key, value, start=None):
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("key and value must share (batch, heads, tokens, dim)")
        begin = self.length if start is None else start
        end = begin + key.shape[2]
        if key.shape[:2] != (self.batch_size, self.kv_heads) or key.shape[3] != self.head_dim:
            raise ValueError("KV shape does not match cache")
        if begin < 0 or end > self.max_sequence:
            raise ValueError("KV update exceeds cache capacity")
        flat = torch.cat((key, value), dim=0).reshape(-1, self.head_dim).float()
        codes, scales = quantize(flat, self.codec, self.passes, self.group_size, self.seed)
        tokens = key.shape[2]
        packed = codes[:, :, 0].view(
            self.passes, 2, self.batch_size, self.kv_heads, tokens, -1
        ).permute(1, 2, 3, 0, 4, 5)
        factors = scales[:, :, 0].view(
            self.passes, 2, self.batch_size, self.kv_heads, tokens
        ).permute(1, 2, 3, 0, 4)
        self.key[layer, :, :, :, begin:end].copy_(packed[0])
        self.value[layer, :, :, :, begin:end].copy_(packed[1])
        self.key_scale[layer, :, :, :, begin:end, 0].copy_(factors[0])
        self.value_scale[layer, :, :, :, begin:end, 0].copy_(factors[1])
        return self.key[layer, :, :, :, begin:end], self.value[layer, :, :, :, begin:end]

    def advance(self, tokens):
        if tokens < 0 or self.length + tokens > self.max_sequence:
            raise ValueError("cache advance exceeds capacity")
        self.length += tokens

    def layer(self, layer, length=None):
        end = self.length if length is None else length
        if not 0 <= end <= self.length:
            raise ValueError("requested length is outside valid cache")
        return self.key[layer, :, :, :, :end], self.value[layer, :, :, :, :end]

    def attention_logits(self, query, layer, batch=0, head=0, length=None):
        if query.ndim != 1 or query.numel() != self.head_dim:
            raise ValueError("query must have shape (head_dim,)")
        end = self.length if length is None else length
        if end <= 0 or end > self.length:
            raise ValueError("length must be within the committed cache")
        words = self.key[layer, batch, head, :, :end].contiguous().view(self.passes, end, -1).view(torch.int32)
        scales = self.key_scale[layer, batch, head, :, :end, 0].unsqueeze(-1)
        if self._rotations is not None:
            rotated = torch.einsum("d,pdf->pf", query.float(), self._rotations)
        else:
            rotated = torch.stack([
                random_rotation(query.float(), self.seed + p)
                for p in range(self.passes)
            ])
        return triton_gemv_trellis(rotated, words, scales, trellis_window_table(self.codec), self.group_size, self.codec.bits, self.codec.trellis_bits).squeeze(0)

    def attention(self, query, layer, batch=0, head=0, length=None):
        """Return scaled attention output with fused trellis QK logits."""
        end = self.length if length is None else length
        logits = self.attention_logits(query, layer, batch, head, end)
        weights = torch.softmax(logits / self.head_dim ** 0.5, dim=-1)
        codes = self.value[layer, batch, head, :, :end]
        scales = self.value_scale[layer, batch, head, :, :end]
        values = self.codec.decode(codes) * scales.to(torch.float32)
        if self._rotations is not None:
            values = torch.einsum("ped,pfd->pef", values, self._rotations)
        else:
            values = torch.stack([
                inverse_random_rotation(values[p], self.seed + p)
                for p in range(self.passes)
            ])
        values = values.sum(dim=0)
        return weights.to(values.dtype) @ values
