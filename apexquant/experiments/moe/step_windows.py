"""WikiText-2 KL windows, a generic held-out set for the oracle, and usage windows."""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer

from clinical.load_texts import load_texts
from clinical.token_windows import token_windows
from ladder.model_dir import model_dir

from .settings import HELDOUT_WINDOWS, KL_WINDOWS, MODELS, SEQ_LEN, SMOKE, USAGE_WINDOWS


def step_windows(work: str) -> None:
    """Write ``windows/{model}/{wikitext,heldout,usage}.pt``; held-out and usage are disjoint C4 draws."""
    c4 = load_texts("c4", "validation", 400 if SMOKE else 20000)
    wiki = load_texts("wikitext", "test", 1)
    for key in MODELS:
        out = os.path.join(work, "windows", key)
        os.makedirs(out, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(model_dir(key, MODELS))
        torch.save(token_windows(tok, wiki, KL_WINDOWS, SEQ_LEN, seed=0), os.path.join(out, "wikitext.pt"))
        both = token_windows(tok, c4, HELDOUT_WINDOWS + USAGE_WINDOWS, SEQ_LEN, seed=50)
        torch.save(both[:HELDOUT_WINDOWS].clone(), os.path.join(out, "heldout.pt"))
        torch.save(both[HELDOUT_WINDOWS:].clone(), os.path.join(out, "usage.pt"))
