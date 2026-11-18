"""Progressive codes of one vLLM layer, encoded once at full depth and read as prefixes."""

from __future__ import annotations

import os

import torch

from ..vq.progressive_artifact import quantize
from .float32_default import float32_default


def cached_codes(prefix: str, stage: str, weight: torch.Tensor, codec, passes: int,
                 group_size: int, seed: int, row_block: int | None,
                 tag: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes, scale)`` for the first ``passes`` passes of ``weight``.

    With ``APEXQUANT_CODE_CACHE`` set, the layer is encoded at
    ``APEXQUANT_ENCODE_PASSES`` (default three) passes the first time and saved;
    later loads read the file and keep a prefix, which is exactly the shallower
    encoding because the passes are nested. ``tag`` names everything else the
    codes depend on (model, codec parameters), so a stale file is never reused;
    each tensor-parallel rank keeps its own shard.
    """
    root = os.environ.get("APEXQUANT_CODE_CACHE")
    if root is None:
        with float32_default():
            return quantize(weight, codec, passes, group_size, seed, row_block=row_block)
    from vllm.distributed import (get_tensor_model_parallel_rank,
                                  get_tensor_model_parallel_world_size)

    depth = max(passes, int(os.environ.get("APEXQUANT_ENCODE_PASSES", "3")))
    shard = f"tp{get_tensor_model_parallel_world_size()}-rank{get_tensor_model_parallel_rank()}"
    path = os.path.join(root, f"{stage}-{tag}-gs{group_size}-seed{seed}", shard,
                        f"{prefix}.pt")
    if os.path.exists(path):
        codes, scale = torch.load(path, map_location=weight.device)
    else:
        with float32_default():
            codes, scale = quantize(weight, codec, depth, group_size, seed,
                                    row_block=row_block)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save((codes.cpu(), scale.cpu()), path + ".tmp")
        os.replace(path + ".tmp", path)
    if codes.shape[0] < passes:
        raise RuntimeError(f"{path} holds {codes.shape[0]} passes, {passes} requested")
    return codes[:passes].contiguous(), scale[:passes].contiguous()
