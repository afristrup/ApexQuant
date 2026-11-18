"""Load one checkpoint for scoring."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM


def load_causal_lm(path: str, revision: str | None = None):
    """A bfloat16 model spread over the visible GPUs, in eval mode.

    Fails if any weight of the architecture was left at its initial value, which
    transformers otherwise only warns about.
    """
    model, info = AutoModelForCausalLM.from_pretrained(
        path, revision=revision, dtype=torch.bfloat16, device_map="auto",
        output_loading_info=True)
    if info["missing_keys"]:
        raise RuntimeError(f"{path}: weights not in the checkpoint {sorted(info['missing_keys'])[:3]}")
    return model.eval().requires_grad_(False)
