"""Small allocation-conscious token selection helpers."""

from __future__ import annotations

import torch


def sample_next_token(logits: torch.Tensor, *, temperature: float = 0.0,
                      top_k: int | None = None) -> torch.Tensor:
    """Select one token per batch row using greedy or temperature sampling."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocabulary)")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and not 0 < top_k <= logits.shape[-1]:
        raise ValueError("top_k must be within the vocabulary size")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    scores = logits / temperature
    if top_k is not None:
        values, _ = torch.topk(scores, top_k, dim=-1)
        scores = scores.masked_fill(scores < values[:, -1:], float("-inf"))
    return torch.multinomial(scores.softmax(dim=-1), 1)
