"""CUDA graph decode for fixed-batch Transformers StaticCache."""

from __future__ import annotations

from collections.abc import Callable

import torch


def capture_static_cache_decode(
    model: torch.nn.Module,
    cache: object,
    token: torch.Tensor,
    cache_position: torch.Tensor,
    warmup: int = 3,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Capture one-token decode and return a launch-free replay callable."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if token.shape != (1, 1) or token.dtype != torch.long:
        raise ValueError("token must be a CUDA int64 tensor of shape (1, 1)")
    if cache_position.shape != (1,) or cache_position.dtype != torch.long:
        raise ValueError("cache_position must be a CUDA int64 tensor of shape (1,)")
    if token.device.type != "cuda" or cache_position.device.type != "cuda":
        raise ValueError("token and cache_position must be CUDA tensors")
    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor) and inv_freq.device.type != "cuda":
            module.inv_freq = inv_freq.to(token.device)
    static_token = token.clone()
    static_position = cache_position.clone()
    with torch.inference_mode():
        for _ in range(warmup):
            model(input_ids=static_token, past_key_values=cache,
                  use_cache=True, cache_position=static_position)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = model(
                input_ids=static_token, past_key_values=cache,
                use_cache=True, cache_position=static_position).logits

    def replay(next_token: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        if next_token.shape != (1, 1) or position.shape != (1,):
            raise ValueError("replay inputs have invalid shapes")
        static_token.copy_(next_token)
        static_position.copy_(position)
        graph.replay()
        return static_logits

    return replay
