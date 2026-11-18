"""The fused experts modules of a mixture-of-experts model, by layer."""

from __future__ import annotations


def experts_modules(model) -> dict[str, object]:
    """Every submodule holding 3D ``gate_up_proj`` and ``down_proj`` expert tensors."""
    found = {name: mod for name, mod in model.named_modules()
             if getattr(getattr(mod, "gate_up_proj", None), "dim", lambda: 0)() == 3
             and getattr(getattr(mod, "down_proj", None), "dim", lambda: 0)() == 3}
    if not found:
        raise RuntimeError("no fused experts modules found")
    return found
