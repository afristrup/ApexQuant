"""Per-output-row projection of the quantization error onto its weight row."""

from __future__ import annotations

import torch

WEIGHT_NDIM = 2


def row_alpha(w: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
    """Return ``alpha_c = <W_c, dW_c> / ||W_c||^2`` for every output row ``c``.

    ``alpha_c`` is the coherent part of the quantization error: the component of
    ``dW_c = W_hat_c - W_c`` that lies along the row itself. For ApexQuant's
    density-matched codebook the centroid condition makes it systematically
    *negative* -- about ``-rho^{2P}`` -- so the layer output contracts rather
    than inflates, and the leading variance term of QBN is ``(2 alpha_c +
    alpha_c^2) v_c``. It is exact algebra on the realized error: no input model,
    no distributional assumption, no data.

    Padding columns do not matter. ``turboquantize_weight`` pads the input axis
    with structural zeros, and a zero column contributes nothing to either the
    inner product or the norm, so ``alpha_c`` is the same whether it is computed
    on the padded or the unpadded weight.

    Args:
        w: Dense fp32 weight ``(out_features, in_features)``, after the harness
            reshape of a conv to ``(C_out, C_in * kh * kw)``.
        dw: Quantization error ``W_hat - W``, same shape as ``w``.

    Returns:
        ``(out_features,)`` float32 tensor of per-row ``alpha_c``.
    """
    if w.ndim != WEIGHT_NDIM:
        raise ValueError(f"w must be {WEIGHT_NDIM}-D (out_features, in_features), got {tuple(w.shape)}")
    if dw.shape != w.shape:
        raise ValueError(f"dw must match w, got {tuple(dw.shape)} against {tuple(w.shape)}")

    w = w.to(torch.float32)
    dw = dw.to(torch.float32)
    row_energy = w.pow(2).sum(dim=1)
    if not bool(torch.all(row_energy > 0)):
        raise ValueError("w has a zero output row; alpha_c is undefined there")
    return (w * dw).sum(dim=1) / row_energy
