"""Tokenize every calibration, development and evaluation set once."""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer

from .checkpoint_dir import checkpoint_dir
from .load_texts import load_texts
from .settings import CALIB_DRAWS, CALIB_WINDOWS, EVAL_WINDOWS, LQ_SAMPLES, SEQ_LEN, SMOKE
from .token_windows import token_windows

SETS = {
    "calib_oracle": ("pubmed", "oracle", 8000, CALIB_WINDOWS),
    "calib_oracle_lq": ("pubmed", "oracle", 60000, LQ_SAMPLES),
    "pubmed_dev": ("pubmed", "dev", 4000, 32),
    "eval_pubmed": ("pubmed", "eval", 4000, EVAL_WINDOWS),
    "eval_wikitext": ("wikitext", "test", 1, EVAL_WINDOWS),
    "eval_code": ("code", "valid", 4000, EVAL_WINDOWS),
    "eval_danish": ("danish", "train", 6000, EVAL_WINDOWS),
}


def step_windows(work: str) -> None:
    """Write ``windows/{name}.pt``; generic calibration draws differ only in their seed.

    LiftQuant's recipe calibrates on 4096 sequences, so it gets its own generic
    and oracle sets of that size, ``calib_c4_lq`` and ``calib_oracle_lq``.
    """
    out = os.path.join(work, "windows")
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(checkpoint_dir("base", work))
    c4 = load_texts("c4", "validation", 400 if SMOKE else 40000)
    for draw in range(CALIB_DRAWS):
        torch.save(token_windows(tok, c4, CALIB_WINDOWS, SEQ_LEN, seed=draw),
                   os.path.join(out, f"calib_c4_{draw}.pt"))
    torch.save(token_windows(tok, c4, LQ_SAMPLES, SEQ_LEN, seed=100),
               os.path.join(out, "calib_c4_lq.pt"))
    for name, (source, split, docs, count) in SETS.items():
        texts = load_texts(source, split, min(docs, 300) if SMOKE else docs)
        torch.save(token_windows(tok, texts, count, SEQ_LEN, seed=0), os.path.join(out, f"{name}.pt"))
        print(f"{name}: {count} windows", flush=True)
