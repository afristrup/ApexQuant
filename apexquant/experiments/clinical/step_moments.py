"""Input moments of one variant on generic and clinical text, and kappa per layer."""

from __future__ import annotations

import json
import os

import torch

from .checkpoint_dir import checkpoint_dir
from .collect_moments import collect_moments
from .generalized_top import generalized_top
from .input_key import input_key
from .load_causal_lm import load_causal_lm
from .settings import CONTROLLED_LAYERS, CONTROLLED_MODULE, SAMPLE_TOKENS

SETS = {"generic": "calib_c4_0", "pubmed_eval": "eval_pubmed", "pubmed_dev": "pubmed_dev"}


def step_moments(work: str, variant: str) -> None:
    """All layers for the fine-tune; the controlled layers for every other variant.

    ``pubmed_dev`` is collected for the base model only, since it shapes the
    constructed updates and must stay disjoint from what is scored.
    """
    root = os.path.join(work, "moments", variant)
    if os.path.exists(os.path.join(root, "kappa.json")):
        print(f"{variant}: moments done already", flush=True)
        return
    model = load_causal_lm(checkpoint_dir(variant, work))
    only = None if variant == "ft" else {
        input_key(f"model.layers.{i}.{CONTROLLED_MODULE}") for i in CONTROLLED_LAYERS}
    sets = SETS if variant == "base" else {k: v for k, v in SETS.items() if k != "pubmed_dev"}
    for label, windows in sets.items():
        ids = torch.load(os.path.join(work, "windows", f"{windows}.pt"))
        collect_moments(model, ids, os.path.join(root, label), SAMPLE_TOKENS, only=only)
        print(f"{variant}/{label} done", flush=True)
    del model
    kappa = {}
    for name in sorted(os.listdir(os.path.join(root, "generic"))):
        s_s = torch.load(os.path.join(root, "generic", name))["sigma"].cuda()
        s_t = torch.load(os.path.join(root, "pubmed_eval", name))["sigma"].cuda()
        kappa[name.removesuffix(".pt")] = float(generalized_top(s_t, s_s, 1)[0][0])
    json.dump(kappa, open(os.path.join(root, "kappa.json"), "w"), indent=1)
