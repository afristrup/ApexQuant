"""One named tensor from a sharded safetensors checkpoint, without loading the rest."""

from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open


def read_tensor(path: str, name: str) -> torch.Tensor:
    """Tensor ``name`` from the checkpoint directory ``path``."""
    index = os.path.join(path, "model.safetensors.index.json")
    shard = (json.load(open(index))["weight_map"][name] if os.path.exists(index)
             else "model.safetensors")
    with safe_open(os.path.join(path, shard), framework="pt") as f:
        return f.get_tensor(name)
