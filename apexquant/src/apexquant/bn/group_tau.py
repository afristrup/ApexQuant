"""Mean input variance over the columns of each weight group."""

from __future__ import annotations

import torch

VARIANCE_RTOL = 1e-6


def group_tau(e: torch.Tensor, s: torch.Tensor, in_index: torch.Tensor, group_size: int) -> torch.Tensor:
    """Return ``tau_b = mean_j (s_{in(j)} - e_{in(j)}^2)`` over each group's columns.

    This is ``tr(Cov x) / d`` restricted to group ``b``, the trace aggregate the
    isotropy corollary needs and the only thing about the input covariance that
    BatchNorm actually stores.

    The two moments are taken separately and the square is subtracted here on
    purpose. The variance line needs ``tr(Cov x)/d`` and *not*
    ``tr(E[x x^T])/d``: using the second moment while the mean is also being
    corrected double counts the mean, and on ResNet-50 inflates the median
    ``tau`` from 0.610 to 0.955. A signature taking one pre-combined tensor would
    let a caller make that mistake silently, so it takes ``e`` and ``s``.

    Args:
        e: ``(C_in,)`` per-input-channel first moment ``E[x_i]``.
        s: ``(C_in,)`` per-input-channel second moment ``E[x_i^2]``, not a
            variance.
        in_index: ``(d_padded,)`` integer column-to-input-channel map,
            ``in(j) = j // (kh * kw)`` under spatial stationarity, covering the
            padded input axis so that ``d_padded = num_groups * group_size``.
        group_size: Coordinates per group, as resolved by
            :func:`resolve_group_size`.

    Returns:
        ``(num_groups,)`` float32 tensor of per-group mean input variance.
    """
    if e.ndim != 1 or s.shape != e.shape:
        raise ValueError(f"e and s must be 1-D and equal, got {tuple(e.shape)} and {tuple(s.shape)}")
    if in_index.ndim != 1:
        raise ValueError(f"in_index must be 1-D, got {tuple(in_index.shape)}")
    if not torch.is_floating_point(e) or not torch.is_floating_point(s):
        raise ValueError("e and s must be floating point")
    if torch.is_floating_point(in_index):
        raise ValueError("in_index must be an integer tensor of column-to-channel indices")
    if group_size < 1 or in_index.numel() % group_size != 0:
        raise ValueError(f"in_index length {in_index.numel()} is not a multiple of group_size {group_size}")
    if in_index.numel() and (int(in_index.min()) < 0 or int(in_index.max()) >= e.numel()):
        raise ValueError("in_index addresses a channel outside e")

    e = e.to(torch.float32)
    s = s.to(torch.float32)
    variance = s - e.pow(2)
    floor = -VARIANCE_RTOL * float(s.abs().amax()) if s.numel() else 0.0
    if variance.numel() and float(variance.amin()) < floor:
        raise ValueError("s - e^2 is negative; s must be the second moment E[x^2], not a variance")

    columns = variance.index_select(0, in_index.to(torch.long))
    return columns.view(-1, group_size).mean(dim=1)
