"""Re-derive a quantized model's BatchNorm statistics, data-free, in one pass."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .case_letters import CASE_C, CORRECTABLE_CASES
from .conv_bn_site import ConvBnSite
from .conv_bn_sites import conv_bn_sites
from .input_moments import input_moments
from .layer_residual import LayerResidual
from .qbn_correction import qbn_correction
from .qbn_guards import (SKIP_LAYOUT, SKIP_NO_BATCHNORM, SKIP_NO_RESIDUAL,
                         SKIP_NONE, SKIP_UNCORRECTABLE_CASE)
from .qbn_modes import MODE_MEAN, MODE_MEAN_VAR, MODE_VAR, QBN_MODES
from .qbn_report import QbnReport

UNGROUPED_CONV = 1


def apply_qbn(model: nn.Module, residuals: Mapping[str, LayerResidual],
              mode: str = MODE_MEAN_VAR) -> dict[str, QbnReport]:
    """Edit every quantized layer's following BatchNorm to describe the quantized net.

    Call this **after** the weights are quantized and **before** anything is
    folded, on the unfolded model, in eval mode. Both halves of that order are
    forced: the correction needs ``dW``, which only exists after quantization,
    and it needs the unfolded ``(gamma, beta)`` of the *preceding* BatchNorm,
    which folding destroys. Folding afterwards is exact and free.

    ``gamma`` and ``beta`` are never touched. Only ``running_mean`` and
    ``running_var`` change.

    **One pass, deliberately, and it must stay one pass.** The obvious worry is
    that correcting layer ``l`` changes layer ``l+1``'s input moments, so the
    whole thing should be a forward recursion. It should not. Case A reads only
    ``(gamma_p, beta_p, v_p)`` of the preceding BatchNorm, and those are exactly
    what QBN leaves alone; setting ``mu' = E[Y']`` and ``v' = Var[Y']`` makes the
    corrected BatchNorm's output mean ``beta`` and variance
    ``gamma^2 v'/(v' + eps)``, which is to say it *restores* the fp32 output
    distribution. Under its own model the corrected network therefore has zero
    upstream drift, and the correction is purely local. Nagel et al.'s ordering
    constraint, "bias-correct a layer only after all layers feeding into it have
    been corrected", applies to their empirical variant, which measures the
    shift; it does not apply to the analytic one. Every layer's moments are
    re-seeded from its own stored BatchNorm parameters, so modelling error never
    propagates more than one layer, and that is the whole reason this is not the
    network-wide analytic moment propagation that Shekhovtsov and Flach found
    wanting. Do not turn this loop into a recursion.

    The single forward ordering that *is* needed is over sites, not over
    corrections: a case-B identity branch reuses the recorded output moments of
    an earlier node. :func:`input_moments` does that in one pass and hands back
    the finished moments before any buffer is written.

    Cases C and D are not corrected. Case C is the image, and section 4.4 says to
    assert rather than implement it because both falsification models leave their
    stem dense under the harness group rule; a case-C layer that turns up with a
    residual therefore raises, because that assertion has failed. Case D is
    pooling and squeeze-excitation gates, where a data-dependent sigmoid's
    correlation with ``x`` is unavailable from the checkpoint; those layers are
    left uncorrected and counted, and section 8.7 asks for the count to be
    published next to the accuracy.

    The fp32 weight is recovered as ``W = W_hat - dW`` rather than read off the
    model, because by the time this runs the model holds ``W_hat``. That costs
    one float32 rounding, which is five orders of magnitude below the quantity it
    feeds. Section 4.5's
    ``alpha_c`` is defined against the fp32 row, and reading ``W_hat`` instead
    makes its numerator ``alpha ||W||^2 + ||dW||^2``, which is near zero at every
    ``P``: the contraction term would quietly disappear and the variance half of
    the method would read as a null result rather than as a bug.

    Args:
        model: The quantized model, unfolded, in eval mode.
        residuals: Module name to :class:`LayerResidual`, from
            :func:`quantize_with_residuals`. Layers not present were left dense.
        mode: One of :data:`QBN_MODES`. ``"mean"`` is arm 3 of section 7.3,
            ``"mean+var"`` arm 4, ``"var"`` arm 5's control. The three arms are
            one code path with a flag so they cannot drift apart.

    Returns:
        One :class:`QbnReport` per convolution, keyed by its dotted name,
        including the layers that were left alone and why.

    Raises:
        ValueError: ``mode`` is not a known mode, a residual's shape disagrees
            with its weight, or a case-C layer was quantized.
    """
    if mode not in QBN_MODES:
        raise ValueError(f"mode must be one of {QBN_MODES}, got {mode!r}")
    sites = conv_bn_sites(model)
    moments = input_moments(model, sites)

    reports: dict[str, QbnReport] = {}
    with torch.no_grad():
        for site in sites:
            conv = model.get_submodule(site.conv)
            residual = residuals.get(site.conv)
            reason = _skip_reason(model, site, residual, conv)
            if reason != SKIP_NONE:
                reports[site.conv] = QbnReport(
                    conv=site.conv, bn=site.bn, case=site.case, mode=mode,
                    corrected=False, skipped=reason,
                    channels=conv.out_channels, correction=None)
                continue
            bn = model.get_submodule(site.bn)
            e, s = moments[site.conv]
            rows = conv.weight.shape[0]
            # The model's weight is already W_hat by the time this runs, and
            # section 4.5's alpha_c is <W_c, dW_c> / ||W_c||^2 against the *fp32*
            # row. Recovering W = W_hat - dW is exact, and it matters: reading
            # W_hat instead makes the numerator alpha ||W||^2 + ||dW||^2, which
            # is very nearly zero at every P, so the contraction term silently
            # vanishes and the whole variance half of the method reads as a
            # null result.
            # The residuals are captured while the weights are quantized, which
            # may happen before the model is moved to the accelerator, so dw can
            # trail behind on the host. Follow the weight rather than assume.
            dw = residual.dw.to(conv.weight.device, torch.float32)
            fp32_weight = conv.weight.to(torch.float32) - dw
            correction = qbn_correction(
                w=fp32_weight.reshape(rows, -1),
                dw=dw.reshape(rows, -1),
                e=e, s=s, mu=bn.running_mean, v=bn.running_var,
                group_size=residual.group_size)
            if mode in (MODE_MEAN, MODE_MEAN_VAR):
                bn.running_mean.copy_(correction.running_mean.to(bn.running_mean.dtype))
            if mode in (MODE_VAR, MODE_MEAN_VAR):
                bn.running_var.copy_(correction.running_var.to(bn.running_var.dtype))
            reports[site.conv] = QbnReport(
                conv=site.conv, bn=site.bn, case=site.case, mode=mode,
                corrected=True, skipped=SKIP_NONE,
                channels=conv.out_channels, correction=correction)
    return reports


def _skip_reason(model: nn.Module, site: ConvBnSite,
                 residual: LayerResidual | None, conv: nn.Module) -> str:
    """Return why this site is left alone, or ``SKIP_NONE`` to correct it."""
    if residual is None:
        return SKIP_NO_RESIDUAL
    if residual.dw.shape != conv.weight.shape:
        raise ValueError(f"{site.conv}: residual is {tuple(residual.dw.shape)} but "
                         f"the weight is {tuple(conv.weight.shape)}")
    if site.case == CASE_C:
        raise ValueError(
            f"{site.conv} reads the image and was quantized anyway; section 4.4 "
            "asserts this cannot happen, because both falsification models leave "
            "their stem dense under the harness group rule")
    if site.case not in CORRECTABLE_CASES:
        return SKIP_UNCORRECTABLE_CASE
    if site.bn is None:
        return SKIP_NO_BATCHNORM
    bn = model.get_submodule(site.bn)
    fan_in = conv.weight.shape[1] * conv.weight.shape[2] * conv.weight.shape[3]
    if (conv.groups != UNGROUPED_CONV or fan_in % residual.group_size != 0
            or getattr(bn, "running_var", None) is None
            or bn.running_var.shape != (conv.out_channels,)):
        return SKIP_LAYOUT
    return SKIP_NONE
