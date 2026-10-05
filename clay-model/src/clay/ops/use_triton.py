"""Decide whether an op should take its Triton path for a given tensor."""

from __future__ import annotations

import importlib.util
import os

import torch

from ..constants import DISABLE_TRITON_ENV

TRITON_INSTALLED = importlib.util.find_spec("triton") is not None


def use_triton(x: torch.Tensor) -> bool:
    """Return whether the Triton kernels should serve ``x``.

    The kernels accumulate ``tl.dot`` in TF32, and that single choice is what
    makes them worth taking. At ``input_precision="ieee"`` they lose to torch on
    an RTX 4090 -- attention 4.1 ms against 1.3 ms, linear+GELU 1.4 ms against
    1.1 ms, because torch's fp32 SDPA and cuBLAS SGEMM are better tuned than a
    readable Triton kernel. In TF32 the same two land at 0.55 ms and 0.51 ms.

    What that costs was measured on the published Clay v1.5 encoder (large,
    patch 8, 311M parameters) over a batch of 4 Sentinel-2 chips at 256 px: the
    whole forward pass drops from 95.8 ms to 70.5 ms, a 1.36x speedup, and the
    chip embedding it returns has cosine similarity 0.9999986 with the torch
    one. Individual tokens move by up to 4e-3. For similarity search over
    embeddings that is not a meaningful difference; for a task that reads token
    values directly, set :data:`~clay.constants.DISABLE_TRITON_ENV`.

    Three things disqualify a tensor outright. It is not on CUDA, so there is no
    kernel to launch. Triton is not installed, since it is an optional
    dependency. Or it is part of a live autograd graph: the kernels here are
    forward-only, so taking them under ``requires_grad`` would silently detach
    the result. Training therefore always runs on the torch path.

    Args:
        x: The tensor the op is about to consume.

    Returns:
        True if the Triton path is available, wanted and safe.
    """
    if os.environ.get(DISABLE_TRITON_ENV):
        return False
    if not TRITON_INSTALLED or not x.is_cuda:
        return False
    return not (torch.is_grad_enabled() and x.requires_grad)
