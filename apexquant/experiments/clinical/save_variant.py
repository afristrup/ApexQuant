"""Write a model plus additive updates as a standalone checkpoint."""

from __future__ import annotations

import json
import os

import torch
from transformers import AutoTokenizer


@torch.no_grad()
def save_variant(model, updates: dict[str, torch.Tensor], tokenizer_dir: str, path: str,
                 meta: dict) -> None:
    """Add ``updates`` to the named parameters and save model, tokenizer and ``meta``."""
    params = dict(model.named_parameters())
    for name, update in updates.items():
        p = params[name]
        p.data = (p.data.float() + update.to(p.device)).to(p.dtype)
    model.save_pretrained(path, safe_serialization=True)
    AutoTokenizer.from_pretrained(tokenizer_dir).save_pretrained(path)
    json.dump(meta, open(os.path.join(path, "variant.json"), "w"), indent=1)
