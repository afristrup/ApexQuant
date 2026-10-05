"""Allocate extra residual passes to the layers a calibration pass says matter."""

from __future__ import annotations

import copy
from typing import Callable, Mapping, Optional

import torch
from torch import nn

from ..config import ApexQuantConfig
from ..nn.linear_is_fusable import linear_is_fusable
from ..nn.apex_linear import ApexQuantLinear


def _set_submodule(model: nn.Module, qualified: str, new: nn.Module) -> None:
    parent = model
    parts = qualified.split(".")
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    setattr(parent, parts[-1], new)


def _fusable_layers(
    model: nn.Module,
    config: ApexQuantConfig,
    fuse_narrow: bool,
    keep_dense: Optional[Callable[[str, nn.Linear], bool]],
) -> list[tuple[str, nn.Linear]]:
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and linear_is_fusable(m.in_features, config, fuse_narrow)
            and not (keep_dense is not None and keep_dense(n, m))]


def measure_layer_sensitivity(
    model: nn.Module,
    run: Callable[[nn.Module], torch.Tensor],
    config: ApexQuantConfig | None = None,
    *,
    fuse_narrow: bool = True,
    keep_dense: Optional[Callable[[str, nn.Linear], bool]] = None,
) -> dict[str, float]:
    """Measure each fusable layer's share of the output error, leave-one-out.

    ``sensitivity[name] = ||run(quantized only at name) - run(dense)|| / ||run(dense)||``,
    one forward pass per candidate layer plus one reference forward. Measured on
    deep copies, so ``model`` is left untouched.

    The result depends on ``config`` -- including ``config.passes``, since each
    trial layer is quantized at the base pass count -- but not on how the passes
    are subsequently allocated. Sweeping ``extra_passes`` or ``upgrade_fraction``
    at a fixed base therefore needs this only once; pass the result to
    :func:`sensitivity_residual_allocation` as ``sensitivity=``.

    Args:
        model: The dense model (not mutated).
        run: Calibration forward: takes a model, returns its output tensor.
        config: Quantization settings used for the trial layers.
        fuse_narrow: Consider narrow layers fusable too (full coverage).
        keep_dense: Optional ``(qualified_name, linear) -> bool``; matching layers
            are neither measured nor returned.

    Returns:
        ``{qualified_name: sensitivity}`` for every fusable, non-protected layer.
    """
    config = config or ApexQuantConfig()
    with torch.no_grad():
        reference = run(model).detach()
    denom = reference.norm().clamp_min(1e-12).item()

    sensitivity: dict[str, float] = {}
    for name, _ in _fusable_layers(model, config, fuse_narrow, keep_dense):
        trial = copy.deepcopy(model)
        target = dict(trial.named_modules())[name]
        _set_submodule(trial, name, ApexQuantLinear.from_linear(target, config))
        with torch.no_grad():
            out = run(trial).detach()
        sensitivity[name] = (out - reference).norm().item() / denom
    return sensitivity


def sensitivity_residual_allocation(
    model: nn.Module,
    run: Optional[Callable[[nn.Module], torch.Tensor]] = None,
    config: ApexQuantConfig | None = None,
    *,
    sensitivity: Optional[Mapping[str, float]] = None,
    extra_passes: int = 1,
    upgrade_fraction: float = 0.25,
    graded: bool = False,
    fuse_narrow: bool = True,
    keep_dense: Optional[Callable[[str, nn.Linear], bool]] = None,
) -> dict[str, int]:
    """Return a per-layer residual-pass map, spending extra passes where it pays.

    At coarse bit-widths a *few* layers dominate the output error (for a
    Clay-style encoder, the patch embedding). A uniform pass count wastes bits on
    the robust layers; this measures each fusable layer's output sensitivity on a
    calibration input and hands the extra passes to the most sensitive ones per
    parameter, so the average bit budget barely moves. Feed the result to
    ``fuse_earth_observation_model(..., passes_for=alloc.get)``.

    Sensitivity is ``||run(one-layer-quantized) - run(dense)|| / ||run(dense)||``,
    measured on a deep copy so ``model`` is left untouched
    (:func:`measure_layer_sensitivity`).

    Args:
        model: The dense model (not mutated).
        run: Calibration forward: takes a model, returns its output tensor.
            Unused, and may be ``None``, when ``sensitivity`` is supplied.
        config: Base quantization settings (its ``passes`` is the floor).
        sensitivity: Optional precomputed ``{name: sensitivity}`` from
            :func:`measure_layer_sensitivity`, which skips the measurement
            sweep. Sweeping ``extra_passes``/``upgrade_fraction`` at a fixed base
            config reuses one measurement instead of repeating it per point; it
            must have been measured with the same ``config``, ``fuse_narrow`` and
            ``keep_dense``, and must cover every layer selected here.
        extra_passes: Passes added on top of ``config.passes`` for upgraded layers
            (the *maximum* extra when ``graded``).
        upgrade_fraction: Fraction of *parameters* (most sensitive first) to
            upgrade. Small values (the patch embedding is tiny) keep the average
            bit budget near the base.
        graded: Scale each upgraded layer's extra passes by its sensitivity
            relative to the most sensitive layer, ``round(extra_passes * s/s_max)``
            (at least 1). The sensitivity spread is ~20:1, so grading spends bits
            more efficiently than a flat bump -- lower error at equal average bits.
        fuse_narrow: Consider narrow layers fusable too (full coverage).
        keep_dense: Optional ``(qualified_name, linear) -> bool`` matching the
            fuser's ``keep_dense``. Protected layers are neither measured nor
            allocated (they stay dense), so the sensitivity pass never quantizes a
            layer the fuse will skip -- important when a submodule reads its
            ``.weight`` directly (e.g. ``nn.MultiheadAttention``).

    Returns:
        ``{qualified_name: passes}`` for every fusable, non-protected layer.
    """
    config = config or ApexQuantConfig()
    layers = _fusable_layers(model, config, fuse_narrow, keep_dense)
    params = {n: m.in_features * m.out_features for n, m in layers}

    if sensitivity is None:
        if run is None:
            raise ValueError("pass either run= (to measure) or sensitivity= (precomputed)")
        sensitivity = measure_layer_sensitivity(model, run, config, fuse_narrow=fuse_narrow,
                                                keep_dense=keep_dense)
    else:
        # A precomputed map from a different config/keep_dense would silently
        # allocate against the wrong layer set; require it to cover this one.
        missing = sorted(set(params) - set(sensitivity))
        if missing:
            raise ValueError(f"sensitivity= is missing {len(missing)} selected layer(s), "
                             f"e.g. {missing[:3]}; measure it with the same config, "
                             "fuse_narrow and keep_dense")

    # Greedy by bang-for-buck (sensitivity per parameter); upgrade until the
    # parameter budget is spent.
    budget = upgrade_fraction * sum(params.values())
    order = sorted(params, key=lambda n: sensitivity[n] / params[n], reverse=True)
    upgraded: set[str] = set()
    spent = 0
    for name in order:
        upgraded.add(name)
        spent += params[name]
        if spent >= budget:
            break

    base = config.passes
    if not graded:
        return {name: (base + extra_passes if name in upgraded else base) for name in params}

    # Grade the extra passes by sensitivity relative to the most sensitive layer,
    # so a 20:1 sensitivity spread doesn't get a flat bump.
    s_max = max(sensitivity[n] for n in upgraded)
    result = {}
    for name in params:
        if name in upgraded:
            extra = max(1, round(extra_passes * sensitivity[name] / s_max))
            result[name] = base + extra
        else:
            result[name] = base
    return result
