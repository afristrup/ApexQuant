"""Linear layers that read the same activation share one moment matrix."""

from __future__ import annotations

SHARED = {"k_proj": "q_proj", "v_proj": "q_proj", "up_proj": "gate_proj"}


def input_key(name: str) -> str:
    """The name of the representative layer whose input ``name`` also reads."""
    head, _, leaf = name.rpartition(".")
    return f"{head}.{SHARED.get(leaf, leaf)}"
