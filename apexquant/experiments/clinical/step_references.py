"""FP16 accuracy of one variant and its KL from the base model on every eval set."""

from __future__ import annotations

import json
import os

import torch

from .checkpoint_dir import checkpoint_dir
from .load_causal_lm import load_causal_lm
from .run_lm_eval import run_lm_eval
from .settings import CHAT_TEMPLATE, CLINICAL_TASKS, GENERIC_TASKS, KL_SETS, REFERENCES
from .token_kl import token_kl


def step_references(work: str, variant: str) -> None:
    """Write ``{REFERENCES}/{variant}/summary.json``; ``KL(variant || base)`` is the undo denominator."""
    out = os.path.join(work, REFERENCES, variant)
    if os.path.exists(os.path.join(out, "summary.json")):
        print(f"{variant}: references done already", flush=True)
        return
    path = checkpoint_dir(variant, work)
    tasks = CLINICAL_TASKS + (GENERIC_TASKS if variant in ("base", "ft") else ())
    result = {"accuracy": run_lm_eval(path, tasks, os.path.join(out, "lm_eval"), chat=CHAT_TEMPLATE)}
    if variant != "base":
        reference, base = load_causal_lm(path), load_causal_lm(checkpoint_dir("base", work))
        result["kl_to_base"] = {
            name: token_kl(reference, base, torch.load(
                os.path.join(work, "windows", f"eval_{name}.pt")))
            for name in KL_SETS}
    json.dump(result, open(os.path.join(out, "summary.json"), "w"), indent=1)
