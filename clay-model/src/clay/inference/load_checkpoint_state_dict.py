"""Pull one submodule's weights out of a Lightning training checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

# Key prefix Lightning gives the encoder inside a ``ClayMAEModule`` checkpoint.
ENCODER_PREFIX = "model.encoder."
DECODER_PREFIX = "model.decoder."


def load_checkpoint_state_dict(
    path: str | Path, prefix: str = ENCODER_PREFIX
) -> dict[str, torch.Tensor]:
    """Return the weights stored under ``prefix``, with the prefix removed.

    Published Clay checkpoints are Lightning artefacts: the weights sit under a
    ``state_dict`` key and carry the wrapper's module path, alongside a frozen
    DINOv2 teacher and a projection head that exist only for the training
    objective. This drops all of that and hands back something
    ``Encoder.load_state_dict`` accepts directly.

    Args:
        path: Path to the ``.ckpt`` file.
        prefix: Module path to extract, e.g. ``model.encoder.``.

    Returns:
        A state dict with ``prefix`` stripped from every key.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("state_dict", checkpoint)

    extracted = {
        key[len(prefix) :]: value
        for key, value in weights.items()
        if key.startswith(prefix)
    }
    if not extracted:
        raise ValueError(f"no weights under prefix {prefix!r} in {path}")
    return extracted
