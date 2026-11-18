"""Build the controlled variant ``W_0 + Delta_c``, sized on PubMed dev windows."""

from __future__ import annotations

import os

import torch

from .checkpoint_dir import checkpoint_dir
from .controlled_directions import controlled_directions
from .load_causal_lm import load_causal_lm
from .save_variant import save_variant
from .scale_to_kl import scale_to_kl
from .settings import CONTROLLED_KL, CONTROLLED_LAYERS, CONTROLLED_MODULE, CONTROLLED_RANK


def step_controlled(work: str) -> None:
    """Directions from base moments on generic text and PubMed dev; scale to the target KL."""
    if os.path.exists(os.path.join(work, "checkpoints", "controlled", "variant.json")):
        print("controlled: built already", flush=True)
        return
    base_path = checkpoint_dir("base", work)
    reference, candidate = load_causal_lm(base_path), load_causal_lm(base_path)
    moments = os.path.join(work, "moments", "base")
    updates = controlled_directions(reference, os.path.join(moments, "generic"),
                                    os.path.join(moments, "pubmed_dev"), CONTROLLED_LAYERS,
                                    CONTROLLED_MODULE, CONTROLLED_RANK, seed=0)
    dev = torch.load(os.path.join(work, "windows", "pubmed_dev.pt"))[:16]
    t = scale_to_kl(reference, candidate, updates, dev, CONTROLLED_KL)
    del candidate
    scaled = {n: t * u for n, u in updates.items()}
    save_variant(reference, scaled, base_path, os.path.join(work, "checkpoints", "controlled"),
                 {"scale": t, "target_kl": CONTROLLED_KL, "layers": list(CONTROLLED_LAYERS),
                  "module": CONTROLLED_MODULE, "rank": CONTROLLED_RANK})
