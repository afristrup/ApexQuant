"""Keep the checkpoint's own architecture name in a re-saved config."""

from __future__ import annotations

import json
import os


def restore_architecture(source: str, saved: str) -> None:
    """Copy ``architectures`` from ``source`` into ``saved`` when they differ only in case.

    Transformers 5 re-saves PhiMoE as ``PhimoeForCausalLM`` while vLLM registers
    ``PhiMoEForCausalLM``; under the new name vLLM falls back to its Transformers
    backend, which fails. A name that differs in more than case (Qwen3.5's
    multimodal class against the text model saved here) is left alone.
    """
    src = json.load(open(os.path.join(source, "config.json"))).get("architectures") or []
    path = os.path.join(saved, "config.json")
    config = json.load(open(path))
    dst = config.get("architectures") or []
    if len(src) == len(dst) == 1 and src != dst and src[0].lower() == dst[0].lower():
        config["architectures"] = src
        json.dump(config, open(path, "w"), indent=2)
