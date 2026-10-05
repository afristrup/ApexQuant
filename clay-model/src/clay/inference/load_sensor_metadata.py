"""Read the band wavelengths and ground sample distance of one sensor."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml


def load_sensor_metadata(
    path: str | Path, platform: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``waves`` and ``gsd`` tensors the model expects for ``platform``.

    These two values are what make a datacube self-describing: the wavelengths
    tell the dynamic embedding what the channels physically are, and the ground
    sample distance sets the spatial scale of the position encoding. They come
    from the same ``metadata.yaml`` the training pipeline used.

    Args:
        path: Path to ``metadata.yaml``.
        platform: Sensor key, e.g. ``sentinel-2-l2a``.

    Returns:
        Band centres ``(C,)`` in micrometres, in band order, and a scalar GSD.
    """
    metadata = yaml.safe_load(Path(path).read_text())
    if platform not in metadata:
        raise ValueError(f"unknown platform {platform!r} in {path}")

    sensor = metadata[platform]
    waves = torch.tensor(list(sensor["bands"]["wavelength"].values()))
    return waves, torch.tensor(sensor["gsd"])
