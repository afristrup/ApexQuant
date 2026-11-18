"""Build tensors under float32 even while vLLM has set a half default dtype."""

from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def float32_default():
    """vLLM sets the default dtype to the model's while it builds the model, so
    codec tables made inside ``process_weights_after_loading`` would come out half."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)
