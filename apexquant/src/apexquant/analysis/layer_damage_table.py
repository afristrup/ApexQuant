"""Per-layer, per-pass-count damage measured with the real quantizer."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from ..config import ApexQuantConfig
from ..quantize.dequantize_weight import dequantize_weight
from ..quantize.turboquantize_weight import turboquantize_weight


def layer_damage_table(
    model: nn.Module,
    measure: Callable[[nn.Module], float],
    layers: Sequence[str],
    config: ApexQuantConfig,
    pass_counts: Sequence[int],
    *,
    baseline_passes: int | None = None,
    progress: Callable[[str, Mapping[int, float]], None] | None = None,
) -> dict[str, dict[int, float]]:
    """Divergence caused by quantizing one layer at ``P`` passes, for each ``P``.

    This replaces the two-step estimate that goes through the linearity theorem
    (fit ``alpha_l`` from Gaussian noise, then extrapolate to ``alpha_l *
    rho**(2P)``) with a direct measurement of the quantity the allocator
    actually needs. Both approximations in that route are removed at once: the
    transfer from an unbiased Gaussian perturbation to a structured quantization
    residual, and the extrapolation from the probe levels out to the operating
    point.

    Measuring is affordable here because of a fact that is specific to our
    scheme and worth naming. Proposition~2's decay law is accurate enough on real
    weights that the relative error ``t_l = rho**P`` carries essentially no
    per-layer information: across the decoder linears of Phi-1.5 the measured
    ``t_l`` spans a factor of ``1.01`` at every pass count. So the entire ranking
    lives in the coefficient, and a proxy for it that ranks at Spearman ``0.7``
    is the whole error budget of the allocator. One forward pass per layer per
    candidate pass count buys the ranking exactly instead.

    The cost is ``len(layers) * len(pass_counts)`` calls to ``measure``, against
    ``len(layers) * len(levels)`` for the noise route, so this is not more
    expensive than what it replaces; it is the same sweep pointed at the real
    quantizer.

    ``baseline_passes`` decides *where the expansion is taken*, and on real models
    it matters more than any of the above. With it unset, each layer is perturbed
    against an otherwise dense model, which is the linearization the linearity
    theorem states and is correct when every layer's error is small. It is not
    the regime we deploy in: at two bits with every layer quantized at once the
    total perturbation is large, errors compound with depth, and a table built
    against a dense model systematically understates what a shallow layer costs
    once the layers feeding it are also quantized. Setting ``baseline_passes``
    quantizes every layer to that depth first and reports each entry as a
    *marginal* change against that configuration, which is the linearization
    around the point an allocator actually moves away from. Entries are then
    signed: negative means deeper than baseline and better, positive means
    shallower and worse.

    Args:
        model: The unquantized model. Restored exactly on return.
        measure: Maps a (partly quantized) model to a scalar divergence from the
            unperturbed one, returning ``0`` for no perturbation. Data-freeness
            is entirely this function's business; pass a random-token KL probe
            and nothing about the training distribution is read.
        layers: Qualified names of the ``nn.Linear`` modules to characterize.
        config: Quantization settings. Its ``passes`` field is ignored, since
            the sweep supplies the pass count.
        pass_counts: Candidate pass counts, e.g. ``(1, 2, 3, 4)``.
        progress: Optional callback ``(layer_name, {P: divergence})``.

    Returns:
        ``{layer: {P: divergence}}``. With ``baseline_passes`` unset these are
        absolute divergences clamped at zero; with it set they are signed
        marginals against the baseline. Neither is guaranteed monotone in ``P``,
        because a layer already at the noise floor of ``measure`` can come back
        slightly out of order, and the allocator handles that correctly since it
        minimizes over an explicit table rather than assuming a shape.
    """
    modules = dict(model.named_modules())
    missing = [n for n in layers if n not in modules]
    if missing:
        raise ValueError(f"unknown layers: {missing[:3]}")
    if not pass_counts:
        raise ValueError("pass_counts must be non-empty")

    def rebuilt(weight: torch.Tensor, passes: int) -> torch.Tensor:
        cfg = ApexQuantConfig(
            group_size=config.group_size, passes=int(passes), seed=config.seed,
            bits=config.bits, codebook=config.codebook,
            lp_norm=config.lp_norm, sinkhorn_iters=config.sinkhorn_iters,
        )
        return dequantize_weight(turboquantize_weight(weight.float(), cfg)).to(weight.dtype)

    # Held on the host: this is a second copy of every probed weight, so on a 7B
    # it is another 14 GiB and the probe will not fit beside the model on one
    # 24 GiB card. Each is moved back to the layer's device only while that layer
    # is being probed. Pure device placement; the arithmetic is unchanged.
    originals = {n: modules[n].weight.detach().to("cpu", copy=True) for n in layers}
    try:
        base_div = 0.0
        if baseline_passes is not None:
            with torch.no_grad():
                for n in layers:
                    modules[n].weight.copy_(
                        rebuilt(originals[n].to(modules[n].weight.device), baseline_passes))
            base_div = float(measure(model))

        table: dict[str, dict[int, float]] = {}
        for name in layers:
            layer = modules[name]
            original = originals[name].to(layer.weight.device, non_blocking=True)
            row: dict[int, float] = {}
            for p in pass_counts:
                with torch.no_grad():
                    layer.weight.copy_(rebuilt(original, p))
                d = float(measure(model))
                row[int(p)] = d - base_div if baseline_passes is not None \
                    else max(d, 0.0)
            with torch.no_grad():
                # Back to whatever the surrounding configuration is, so the next
                # layer is measured against the same context as this one.
                layer.weight.copy_(rebuilt(original, baseline_passes)
                                   if baseline_passes is not None else original)
            del original
            table[name] = row
            if progress is not None:
                progress(name, row)
    finally:
        with torch.no_grad():
            for n in layers:
                modules[n].weight.copy_(originals[n].to(modules[n].weight.device))

    return table
