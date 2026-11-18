"""A seeded subset of EuroSAT multispectral chips in Clay's band order."""

from __future__ import annotations

import random

import numpy as np
import tifffile
import torch

from .settings import BANDS, CHIPS, EUROSAT, SEED


def load_eurosat() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """``(pixels (N, 10, 64, 64) float32 raw reflectance, labels (N,), classes)``."""
    classes = sorted(p.name for p in EUROSAT.iterdir() if p.is_dir())
    files = [(f, i) for i, c in enumerate(classes) for f in sorted((EUROSAT / c).glob("*.tif"))]
    if len(files) < CHIPS:
        raise RuntimeError(f"{len(files)} chips under {EUROSAT}, need {CHIPS}")
    random.Random(SEED).shuffle(files)
    files = files[:CHIPS]
    pixels = np.stack([tifffile.imread(f)[..., BANDS].transpose(2, 0, 1) for f, _ in files])
    labels = torch.tensor([i for _, i in files])
    return torch.from_numpy(pixels.astype(np.float32)), labels, classes
