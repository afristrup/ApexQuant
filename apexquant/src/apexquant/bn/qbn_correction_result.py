"""The per-channel numbers one layer's QBN correction produces."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QbnCorrection:
    """What section 4.5 computes for one quantized layer, before it is applied.

    Attributes:
        alpha: ``(C_out,)`` realized ``alpha_c = <W_c, dW_c> / ||W_c||^2``. The
            realized value, never the closed form ``-rho^{2P}``: section 8.1 is
            explicit that a structurally padded group makes the closed form
            wrong while the realized value stays right.
        tau: ``(num_groups,)`` per-group mean input variance ``tr(Cov x)/d``.
        d_mean: ``(C_out,)`` raw ``<dW_c, e>``, before the mean-shift guard.
        d_var: ``(C_out,)`` raw ``(2 alpha + alpha^2) v + sum_b perp_b tau_b``,
            before the variance floor.
        running_mean: ``(C_out,)`` the guarded new ``running_mean``.
        running_var: ``(C_out,)`` the guarded new ``running_var``.
        mean_shift_hits: Channels whose ``|d_mean|`` exceeded the guard.
        var_floor_hits: Channels whose corrected variance hit the floor.
    """

    alpha: torch.Tensor
    tau: torch.Tensor
    d_mean: torch.Tensor
    d_var: torch.Tensor
    running_mean: torch.Tensor
    running_var: torch.Tensor
    mean_shift_hits: int
    var_floor_hits: int
