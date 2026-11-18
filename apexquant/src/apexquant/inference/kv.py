"""Preallocated KV storage for static-shape inference."""

from __future__ import annotations

import torch


class KVCache:
    """Layer-major, preallocated key/value cache.

    Keys and values use ``(batch, kv_heads, sequence, head_dim)`` layout.
    ``update`` is allocation-free after construction, which makes the cache
    suitable for CUDA graph replay and continuous decode loops.
    """

    def __init__(self, layers: int, batch_size: int, kv_heads: int,
                 max_sequence: int, head_dim: int, *, dtype=torch.float16,
                 device=None):
        if min(layers, batch_size, kv_heads, max_sequence, head_dim) <= 0:
            raise ValueError("cache dimensions must be positive")
        shape = (layers, batch_size, kv_heads, max_sequence, head_dim)
        self.key = torch.empty(shape, dtype=dtype, device=device)
        self.value = torch.empty_like(self.key)
        self.length = 0

    @property
    def device(self):
        return self.key.device

    @property
    def max_sequence(self):
        return self.key.shape[3]

    def reset(self):
        """Forget all tokens without clearing the backing allocation."""
        self.length = 0

    def update(self, layer: int, key: torch.Tensor, value: torch.Tensor,
               start: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Write one layer and return its valid cache view.

        The sequence length is advanced separately with :meth:`advance`, after
        all layers have written the new token span.
        """
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("key and value must share shape (batch, heads, tokens, dim)")
        begin = self.length if start is None else start
        end = begin + key.shape[2]
        if begin < 0 or end > self.max_sequence:
            raise ValueError("KV update exceeds the cache capacity")
        expected = self.key.shape[:3] + self.key.shape[4:]
        if key.shape[:2] + key.shape[3:] != expected[1:]:
            raise ValueError("KV tensor shape does not match the cache")
        self.key[layer, :, :, begin:end].copy_(key)
        self.value[layer, :, :, begin:end].copy_(value)
        return self.key[layer, :, :, :end], self.value[layer, :, :, :end]

    def advance(self, tokens: int):
        """Commit tokens after every layer has written that span."""
        end = self.length + tokens
        if tokens < 0 or end > self.max_sequence:
            raise ValueError("cache advance exceeds the cache capacity")
        self.length = end

    def layer(self, layer: int, length: int | None = None):
        """Return valid ``(key, value)`` views for one layer."""
        end = self.length if length is None else length
        if not 0 <= end <= self.length:
            raise ValueError("requested length is outside the valid cache")
        return self.key[layer, :, :, :end], self.value[layer, :, :, :end]

    def reorder(self, indices: torch.Tensor):
        """Reorder batch rows in place, for beam or speculative branches."""
        if indices.ndim != 1 or indices.numel() != self.key.shape[1]:
            raise ValueError("indices must contain one entry per cache batch row")
        self.key.copy_(self.key[:, indices])
        self.value.copy_(self.value[:, indices])
