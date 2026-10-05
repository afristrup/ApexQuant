"""The per-channel BatchNorm correction of QBN section 4.5."""

from __future__ import annotations

import torch

from .group_tau import group_tau
from .perp_energy import perp_energy
from .qbn_correction_result import QbnCorrection
from .qbn_guards import MAX_MEAN_SHIFT, VAR_FLOOR_RATIO
from .row_alpha import row_alpha

WEIGHT_NDIM = 2


def qbn_correction(w: torch.Tensor, dw: torch.Tensor, e: torch.Tensor,
                   s: torch.Tensor, mu: torch.Tensor, v: torch.Tensor,
                   group_size: int) -> QbnCorrection:
    """Return the corrected BatchNorm statistics for one quantized layer.

    Section 4.5, per output channel ``c``, with ``dWperp_c = dW_c - alpha_c W_c``::

        alpha_c  = <W_c, dW_c> / ||W_c||^2
        d_mean_c = <dW_c, e>
        d_var_c  = (2 alpha_c + alpha_c^2) v_c + sum_b ||dWperp_{c,b}||^2 tau_b

    and then ``running_mean_c <- mu_c + d_mean_c`` and
    ``running_var_c <- max(v_c + d_var_c, VAR_FLOOR_RATIO v_c)``.

    The derivation is three lines. ``Y'_c = (W_c + dW_c).x + b_c`` gives
    ``E[Y'_c] = E[Y_c] + dW_c . E[x]`` with no distributional assumption beyond
    the value of ``E[x]``, and with ``C = Cov(x)``::

        Var[Y'_c] = (1 + alpha_c)^2 W_c^T C W_c
                  + 2 (1 + alpha_c) W_c^T C dWperp_c
                  + dWperp_c^T C dWperp_c

    The first term is ``(1 + alpha_c)^2 v_c`` exactly, because ``W_c^T C W_c``
    is the stored ``running_var``. The third has rotation-expectation
    ``||dWperp_c||^2 tau``, which depends on ``C`` only through its trace and so
    needs only what BatchNorm supplies.

    **The middle term is dropped, and that is the central limitation of the
    method.** It is mean-zero over the quantizer's random rotation, which is why
    it does not survive the expectation, but its per-channel standard deviation
    was measured at 2.9x to 4.1x the isotropic term on trained features, and it
    requires the off-diagonal input covariance. That information is not in
    ``(gamma, beta, running_mean, running_var)`` and cannot be recovered from
    them: inverting the stored variances as a non-negative least squares problem
    gives relative error 1.7 to 4.1, worse than predicting zero, because
    ``A_{cj} = W_{cj}^2`` is ill conditioned. Do not try to estimate it. The
    consequence for the claim is that QBN corrects the channel-averaged output
    variance, not the per-channel one. The mean correction carries no such
    caveat; it is linear in ``dW`` and needs only ``E[x]``.

    Two terms, never one. ``(2 alpha + alpha^2) v`` and ``perp . tau`` partly
    cancel: at ResNet-50's median ``kappa = 1.286`` the contraction alone is
    ``-2 rho^{2P} = -2.7%`` against a truth of ``-1.7%``. A variance-only arm is
    a control, not a product.

    ``tau`` is evaluated per group rather than per row. ``turboquantize_weight``
    views the weight as ``(out, num_groups, gs)``, so a group never spans two
    output rows and the isotropic term decomposes exactly over groups; per-group
    ``tau`` is strictly more informative and costs nothing. The layout is
    asserted here rather than trusted (section 8.3): a fan-in that is not a whole
    number of groups would mean the quantizer padded the row with structural
    zeros, and the document fixes no convention for what input channel a pad
    column names, so it is rejected instead of guessed. The harness's group rule
    picks a group size that divides the fan-in, so this never fires there.

    Args:
        w: ``(C_out, d)`` fp32 weight after the harness reshape of a convolution
            to ``(C_out, C_in kh kw)``.
        dw: ``(C_out, d)`` realized quantization error ``W_hat - W``.
        e: ``(C_in,)`` per-input-channel first moment ``E[x_i]``.
        s: ``(C_in,)`` per-input-channel second moment ``E[x_i^2]``, not a
            variance. The centring happens inside :func:`group_tau`, because
            using ``tr(E[x x^T])/d`` in the variance line while also correcting
            the mean double counts the mean and inflates ResNet-50's median
            ``tau`` from 0.610 to 0.955.
        mu: ``(C_out,)`` the layer's stored ``running_mean``.
        v: ``(C_out,)`` the layer's stored ``running_var``.
        group_size: Group size the quantizer actually used for this layer.

    Returns:
        A :class:`QbnCorrection` carrying the realized ``alpha``, ``tau``, the
        two raw correction terms, the guarded buffers, and the guard counts.

    Raises:
        ValueError: Shapes disagree, or the fan-in is not a whole number of
            groups of ``group_size`` columns.
    """
    if w.ndim != WEIGHT_NDIM:
        raise ValueError(f"w must be {WEIGHT_NDIM}-D (C_out, d), got {tuple(w.shape)}")
    if dw.shape != w.shape:
        raise ValueError(f"dw must match w, got {tuple(dw.shape)} against {tuple(w.shape)}")
    out_features, d = w.shape
    if mu.shape != (out_features,) or v.shape != (out_features,):
        raise ValueError(f"mu and v must be ({out_features},), got "
                         f"{tuple(mu.shape)} and {tuple(v.shape)}")
    channels = e.numel()
    if e.shape != s.shape or e.ndim != 1 or channels == 0 or d % channels != 0:
        raise ValueError(f"e and s must be (C_in,) with C_in dividing d = {d}, "
                         f"got {tuple(e.shape)} and {tuple(s.shape)}")
    if group_size < 1 or d % group_size != 0:
        raise ValueError(f"fan-in {d} is not a whole number of groups of "
                         f"{group_size}; a padded row has no fixed convention "
                         "for what channel its pad columns name")

    w = w.to(torch.float32)
    dw = dw.to(torch.float32)
    v = v.to(torch.float32)
    spatial = d // channels
    in_index = torch.arange(d, device=w.device) // spatial

    alpha = row_alpha(w, dw)
    groups = d // group_size
    perp = perp_energy(w.reshape(out_features, groups, group_size),
                       dw.reshape(out_features, groups, group_size), alpha)
    tau = group_tau(e, s, in_index, group_size)

    d_mean = (dw * e.to(torch.float32).index_select(0, in_index)).sum(dim=1)
    d_var = (2.0 * alpha + alpha * alpha) * v + (perp * tau).sum(dim=1)

    limit = MAX_MEAN_SHIFT * v.clamp_min(0.0).sqrt()
    guarded_mean = torch.clamp(d_mean, min=-limit, max=limit)
    floor = VAR_FLOOR_RATIO * v
    guarded_var = torch.maximum(v + d_var, floor)
    return QbnCorrection(
        alpha=alpha,
        tau=tau,
        d_mean=d_mean,
        d_var=d_var,
        running_mean=mu.to(torch.float32) + guarded_mean,
        running_var=guarded_var,
        mean_shift_hits=int((d_mean.abs() > limit).sum()),
        var_floor_hits=int((v + d_var < floor).sum()),
    )
