"""Per-group energy of the quantization error orthogonal to its weight row."""

from __future__ import annotations

import torch

GROUPED_NDIM = 3
MIN_GROUP_SIZE = 2


def perp_energy(w: torch.Tensor, dw: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Return ``||dWperp_{c,b}||^2`` for every (row, group) pair.

    ``dWperp_c = dW_c - alpha_c W_c`` is what is left of the error once the
    coherent contraction is removed, and it is the part that feeds the isotropic
    variance term ``sum_b ||dWperp_{c,b}||^2 tau_{l,b}``.

    The split is *per group*, not per row, and that is deliberate.
    ``turboquantize_weight`` pads the input axis and views the weight as
    ``(out_features, num_groups, group_size)``, so a group never spans two output
    rows and the isotropic term decomposes exactly over groups. Evaluating
    ``tau`` over each group's own columns is then strictly more informative than
    the whole-row form and costs nothing. Both inputs must therefore arrive in
    exactly that grouped view; the rank and the group axis are checked here
    rather than trusted, because a channel-last flattening or a grouping across
    output channels would make the decomposition ``dW_c = alpha_c W_c +
    dWperp_c`` ill posed per channel and silently invalidate the correction.

    Args:
        w: Grouped fp32 weight ``(out_features, num_groups, group_size)``, padded
            exactly as :func:`turboquantize_weight` pads it.
        dw: Quantization error ``W_hat - W`` in the same grouped view.
        alpha: ``(out_features,)`` per-row ``alpha_c`` from :func:`row_alpha`.

    Returns:
        ``(out_features, num_groups)`` float32 tensor of squared norms.
    """
    if w.ndim != GROUPED_NDIM:
        raise ValueError(
            f"w must be {GROUPED_NDIM}-D (out_features, num_groups, group_size), got {tuple(w.shape)}"
        )
    if dw.shape != w.shape:
        raise ValueError(f"dw must match w, got {tuple(dw.shape)} against {tuple(w.shape)}")
    group_size = w.shape[2]
    if group_size < MIN_GROUP_SIZE or (group_size & (group_size - 1)) != 0:
        raise ValueError(f"group axis must be a power of two >= {MIN_GROUP_SIZE}, got {group_size}")
    if alpha.shape != w.shape[:1]:
        raise ValueError(f"alpha must be ({w.shape[0]},), got {tuple(alpha.shape)}")

    w = w.to(torch.float32)
    dw = dw.to(torch.float32)
    perp = dw - alpha.to(torch.float32).view(-1, 1, 1) * w
    return perp.pow(2).sum(dim=2)
