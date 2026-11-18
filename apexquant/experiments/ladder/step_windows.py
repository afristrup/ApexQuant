"""Generic calibration windows and WikiText-2 KL windows, one set per model."""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer

from clinical.load_texts import load_texts
from clinical.settings import LQ_SAMPLES
from clinical.token_windows import token_windows

from .model_dir import model_dir
from .settings import CALIB_WINDOWS, KL_WINDOWS, MODELS, MOE, SEQ_LEN, SMOKE


def step_windows(work: str) -> None:
    """Write ``windows/{model}/calib_c4.pt``, ``calib_c4_lq.pt`` and ``wikitext.pt``.

    Each model gets windows from its own tokenizer, from the same documents and
    seeds, so every model sees the same text. LiftQuant's set is skipped for the
    mixture-of-experts models, which do not run it.
    """
    c4 = load_texts("c4", "validation", 400 if SMOKE else 40000)
    wiki = load_texts("wikitext", "test", 1)
    for key in MODELS:
        out = os.path.join(work, "windows", key)
        os.makedirs(out, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(model_dir(key))
        torch.save(token_windows(tok, c4, CALIB_WINDOWS, SEQ_LEN, seed=0),
                   os.path.join(out, "calib_c4.pt"))
        if key not in MOE:
            torch.save(token_windows(tok, c4, LQ_SAMPLES, SEQ_LEN, seed=100),
                       os.path.join(out, "calib_c4_lq.pt"))
        torch.save(token_windows(tok, wiki, KL_WINDOWS, SEQ_LEN, seed=0),
                   os.path.join(out, "wikitext.pt"))
        print(f"{key}: windows written", flush=True)
