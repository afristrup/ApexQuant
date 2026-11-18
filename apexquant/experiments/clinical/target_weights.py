"""Every weight the quantizers touch, dense linears and fused expert tensors alike."""

from __future__ import annotations

import torch

ROUTERS = ("router", "mlp.gate.", "shared_expert_gate")


def target_weights(model) -> dict[str, torch.nn.Parameter]:
    """Decoder linear weights, routers excluded, plus 3D ``experts.*`` tensors.

    A 3D tensor holds one ``(out, in)`` matrix per expert, as transformers stores
    PhiMoE and Qwen3.5-MoE experts; quantizers treat each slice as a matrix.
    """
    found = {}
    for name, p in model.named_parameters():
        if ".layers." not in name or any(r in name + "." for r in ROUTERS):
            continue
        if p.dim() == 2 and name.endswith(".weight"):
            mod = model.get_submodule(name.removesuffix(".weight"))
            if isinstance(mod, torch.nn.Linear):
                found[name] = p
        elif p.dim() == 3 and ".experts." in name:
            found[name] = p
    if not found:
        raise RuntimeError("no decoder weights found")
    return found
