"""Model-agnostic prefill and decode calls."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InferenceOutput:
    logits: torch.Tensor
    past_key_values: object | None


def _logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError("model output must expose logits or be a tuple")


def _past(output):
    if hasattr(output, "past_key_values"):
        return output.past_key_values
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[1]
    return None


def prefill(model, input_ids: torch.Tensor, *, past_key_values=None,
            attention_mask=None, position_ids=None, **kwargs) -> InferenceOutput:
    """Run the prompt once and return logits plus the model's KV state."""
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be a non-empty (batch, sequence) tensor")
    call = dict(input_ids=input_ids, use_cache=True, **kwargs)
    if past_key_values is not None:
        call["past_key_values"] = past_key_values
    if attention_mask is not None:
        call["attention_mask"] = attention_mask
    if position_ids is not None:
        call["position_ids"] = position_ids
    with torch.inference_mode():
        output = model(**call)
    return InferenceOutput(_logits(output), _past(output))


def decode_step(model, input_ids: torch.Tensor, *, past_key_values=None,
                attention_mask=None, position_ids=None, **kwargs) -> InferenceOutput:
    """Run one or more already-tokenized decode steps with KV reuse."""
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be a non-empty (batch, sequence) tensor")
    return prefill(model, input_ids, past_key_values=past_key_values,
                   attention_mask=attention_mask, position_ids=position_ids,
                   **kwargs)
