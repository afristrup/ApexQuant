"""Local snapshot of one pinned E1 model."""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download

from .settings import MODELS


def model_dir(key: str, models: dict | None = None) -> str:
    """The cached snapshot of ``models[key]`` (E1's ``MODELS`` by default) at its pinned revision."""
    repo, revision = (models or MODELS)[key]
    if os.path.isdir(repo):
        return repo
    return snapshot_download(repo, revision=revision,
                             allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
