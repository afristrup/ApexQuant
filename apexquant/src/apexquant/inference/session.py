"""Stateful prompt-prefill and autoregressive decode session."""

from __future__ import annotations

import torch

from .prefill import decode_step, prefill
from .sampling import sample_next_token


class InferenceSession:
    """Reuse model KV state while generating a single batch."""

    def __init__(self, model):
        self.model = model.eval()
        self.past_key_values = None
        self.logits = None
        self.tokens = 0

    def reset(self):
        """Drop the logical sequence state without modifying the model."""
        self.past_key_values = None
        self.logits = None
        self.tokens = 0

    def prefill(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        result = prefill(self.model, input_ids, **kwargs)
        self.past_key_values, self.logits = result.past_key_values, result.logits
        self.tokens = input_ids.shape[1]
        return self.logits

    def decode(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        result = decode_step(self.model, input_ids,
                             past_key_values=self.past_key_values, **kwargs)
        self.past_key_values, self.logits = result.past_key_values, result.logits
        self.tokens += input_ids.shape[1]
        return self.logits

    def generate(self, input_ids: torch.Tensor, *, steps: int,
                 temperature: float = 0.0, top_k: int | None = None, **kwargs):
        """Prefill a prompt and append ``steps`` sampled tokens."""
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self.prefill(input_ids, **kwargs)
        generated = []
        for _ in range(steps):
            token = sample_next_token(self.logits[:, -1], temperature=temperature,
                                      top_k=top_k)
            generated.append(token)
            self.decode(token, **kwargs)
        return torch.cat([input_ids, *generated], dim=1) if generated else input_ids
