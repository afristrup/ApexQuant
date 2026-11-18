"""Train and merge the constrained clinical LoRA on the base model."""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer

from .checkpoint_dir import checkpoint_dir
from .generalized_top import generalized_top
from .input_key import input_key
from .load_causal_lm import load_causal_lm
from .save_variant import save_variant
from .settings import CONTROLLED_LAYERS, CONTROLLED_MODULE, CONTROLLED_RANK, LORA_SPAN
from .train_constrained_lora import train_constrained_lora


def step_constrained(work: str, steps: int, batch: int, lr: float) -> None:
    """Input projection restricted to the top generalized eigenvectors of (dev, generic)."""
    if os.path.exists(os.path.join(work, "checkpoints", "constrained", "variant.json")):
        print("constrained: built already", flush=True)
        return
    base_path = checkpoint_dir("base", work)
    model = load_causal_lm(base_path)
    moments = os.path.join(work, "moments", "base")
    spans = {}
    for layer in CONTROLLED_LAYERS:
        name = f"model.layers.{layer}.{CONTROLLED_MODULE}"
        key = input_key(name)
        s_s = torch.load(os.path.join(moments, "generic", f"{key}.pt"))["sigma"].cuda()
        s_t = torch.load(os.path.join(moments, "pubmed_dev", f"{key}.pt"))["sigma"].cuda()
        spans[name] = generalized_top(s_t, s_s, LORA_SPAN)[1]
    updates = train_constrained_lora(model, AutoTokenizer.from_pretrained(base_path), spans,
                                     CONTROLLED_RANK, steps, batch, lr, seed=0)
    save_variant(model, updates, base_path, os.path.join(work, "checkpoints", "constrained"),
                 {"steps": steps, "batch": batch, "lr": lr, "span": LORA_SPAN,
                  "layers": list(CONTROLLED_LAYERS), "rank": CONTROLLED_RANK})
