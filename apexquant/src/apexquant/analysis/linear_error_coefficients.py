"""Per-layer scaling coefficients of the linearity theorem, by noise insertion."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch
from torch import nn


def linear_error_coefficients(
    model: nn.Module,
    measure: Callable[[nn.Module], float],
    layers: Sequence[str] | None = None,
    *,
    levels: Sequence[float] = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16),
    seed: int = 0,
    progress: Callable[[str, float], None] | None = None,
) -> dict[str, float]:
    """Estimate the coefficients ``alpha_l`` linking layer error to model error.

    HIGGS's linearity theorem states that for small enough per-layer errors the
    model's output error is linear in the *relative squared* layer errors,

        ``E[loss(W_hat)] ~= loss(W*) + sum_l alpha_l * t_l**2``,
        ``t_l**2 = E||W_hat_l - W*_l||_F**2 / ||W*_l||_F**2``,

    with coefficients ``alpha_l`` that depend only on the pretrained weights and
    *not* on the quantizer, which is what makes them worth estimating once and
    reusing across every operating point. That is exactly the quantity our
    geometric decay law produces in closed form: after ``P`` passes,
    ``t_l = rho_l**P``. Composing the two turns the pass count into a prediction
    of end-to-end damage, ``sum_l alpha_l * rho_l**(2P)``, computed before any
    weight is quantized.

    We estimate ``alpha_l`` the way HIGGS does, by *noise insertion* rather than
    by quantizing: perturb one layer at a time with Gaussian noise scaled to a
    known relative error ``t_j``, measure the resulting model divergence, and fit
    a line through the origin in ``t**2``. Two things make this the right
    procedure rather than a convenience. Noise lets us dial ``t`` to whatever
    value we want, including values no real bit-width produces, so the fit is not
    hostage to the grid of available quantizers. And because the theorem says the
    coefficients are quantizer-independent, an estimate obtained from noise is a
    valid estimate for our recursion.

    Whether this is data-free depends entirely on ``measure``. Pass a divergence
    against the unperturbed model on randomly sampled inputs (see
    :func:`random_token_divergence`) and nothing about the training distribution
    is used; pass a validation loss and it is calibration-based. The theorem is
    indifferent, and so is this function.

    Args:
        model: The unquantized model. Restored exactly on return; every
            perturbation is undone before the next layer is touched.
        measure: Maps a (perturbed) model to a scalar divergence from the
            unperturbed one, and must return ``0`` for no perturbation. Called
            ``len(layers) * len(levels)`` times, so its cost sets the cost of the
            whole procedure.
        layers: Qualified names of ``nn.Linear`` modules to characterize.
            Defaults to every ``nn.Linear`` in the model.
        levels: Relative error levels ``t_j`` to probe. Keep them inside the
            regime where the theorem holds: HIGGS report the linear prediction
            tracking above roughly 3 bits per weight and diverging below, so
            probing at error levels far larger than the operating point buys a
            worse fit rather than a better one.
        seed: Base RNG seed for the inserted noise.
        progress: Optional callback ``(layer_name, alpha)`` after each layer.

    Returns:
        ``{layer_name: alpha}``. Coefficients are non-negative by construction:
        the theorem's ``alpha_l`` measure how much a given relative error at that
        layer costs the model, so a negative least-squares fit means the
        measurement was noise-dominated and is clamped to zero.
    """
    names = list(layers) if layers is not None else [
        n for n, m in model.named_modules() if isinstance(m, nn.Linear)
    ]
    if not names:
        raise ValueError("no linear layers to characterize")
    if any(t <= 0 for t in levels):
        raise ValueError("levels must be positive relative errors")

    modules = dict(model.named_modules())
    denom = sum(t ** 4 for t in levels)
    coefficients: dict[str, float] = {}

    for i, name in enumerate(names):
        layer = modules[name]
        weight = layer.weight
        original = weight.detach().clone()
        # Noise scaled so the *relative* Frobenius error is exactly t_j in
        # expectation, which is the quantity the theorem is stated in.
        unit = original.norm() / max(original.numel(), 1) ** 0.5

        numer = 0.0
        for j, t in enumerate(levels):
            g = torch.Generator(device="cpu").manual_seed(seed + 1000 * i + j)
            noise = torch.randn(original.shape, generator=g, dtype=torch.float32)
            noise = noise.to(original.device, original.dtype) * (t * unit)
            with torch.no_grad():
                weight.copy_(original + noise)
            numer += float(measure(model)) * t ** 2

        with torch.no_grad():
            weight.copy_(original)

        alpha = max(numer / denom, 0.0)
        coefficients[name] = alpha
        if progress is not None:
            progress(name, alpha)

    return coefficients
