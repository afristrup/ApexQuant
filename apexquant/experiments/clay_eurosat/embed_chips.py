"""Class-token embeddings of normalized chips, in batches."""

from __future__ import annotations

import torch
import yaml

from clay.inference.embed import embed
from clay.inference.load_sensor_metadata import load_sensor_metadata

from .settings import BATCH, METADATA, PLATFORM


@torch.no_grad()
def embed_chips(encoder, pixels: torch.Tensor) -> torch.Tensor:
    """Normalize with Clay's Sentinel-2 statistics; EuroSAT has no date or location."""
    bands = yaml.safe_load(METADATA.read_text())[PLATFORM]["bands"]
    order = yaml.safe_load(METADATA.read_text())[PLATFORM]["band_order"]
    mean = torch.tensor([bands["mean"][b] for b in order])[:, None, None]
    std = torch.tensor([bands["std"][b] for b in order])[:, None, None]
    waves, gsd = load_sensor_metadata(METADATA, PLATFORM)
    device = next(encoder.parameters()).device
    out = []
    for chunk in pixels.split(BATCH):
        x = ((chunk - mean) / std).to(device)
        cube = {"pixels": x, "time": torch.zeros(len(x), 4, device=device),
                "latlon": torch.zeros(len(x), 4, device=device), "gsd": gsd.to(device),
                "waves": waves.to(device)}
        out.append(embed(encoder, cube)[:, 0].float().cpu())
    return torch.cat(out)
