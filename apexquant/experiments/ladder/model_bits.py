"""Stored bits per parameter, over the whole model and over the quantized weights only."""

from __future__ import annotations

from clinical.target_weights import target_weights

from .linear_bits import linear_bits


def model_bits(model, arm: str, level, group: int) -> dict[str, float]:
    """``whole_model`` counts unquantized tensors at 16 bits; ``quantized`` is the arm's own rate.

    Tied tensors are counted once; a 3D expert tensor is charged per slice. Arms
    ending in ``_l0`` keep decoder layer 0 in BF16, so its weights cost 16 bits.
    """
    linears = target_weights(model)
    total = bits = quantized_n = quantized_bits = 0.0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if name in linears:
            out_f, in_f = p.shape[-2:]
            b = n * (16.0 if arm.endswith("_l0") and ".layers.0." in f".{name}"
                     else linear_bits(arm, level, group, out_f, in_f))
            bits += b
            quantized_n += n
            quantized_bits += b
        else:
            bits += 16.0 * n
    return {"whole_model": bits / total, "quantized": quantized_bits / quantized_n}
