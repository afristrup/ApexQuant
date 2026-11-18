"""Local directory of one model variant."""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download

from .settings import BASE, BASE_REVISION, FINETUNE, FINETUNE_REVISION

HUB = {"base": (BASE, BASE_REVISION), "ft": (FINETUNE, FINETUNE_REVISION)}


def checkpoint_dir(variant: str, work: str) -> str:
    """The pinned hub snapshot for ``base`` and ``ft``; the built checkpoint otherwise."""
    if variant in HUB:
        repo, revision = HUB[variant]
        return snapshot_download(repo, revision=revision,
                                 allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    path = os.path.join(work, "checkpoints", variant)
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise FileNotFoundError(f"{path}: run the prepare step that builds {variant!r}")
    return path
