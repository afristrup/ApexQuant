"""Top generalized eigenpairs of a target and a source moment matrix."""

from __future__ import annotations

import torch


def generalized_top(sigma_t: torch.Tensor, sigma_s: torch.Tensor, k: int,
                    ridge: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``k`` largest ``e^T S_t e / e^T S_s e`` and their unit directions ``e``.

    ``S_s`` is whitened through its eigendecomposition with eigenvalues clamped at
    ``ridge`` times their mean, so directions generic text never visits give a
    large finite ratio. A Cholesky factor would fail on the rank-deficient or
    badly conditioned moments that large activations produce. Returns
    eigenvalues in descending order and a ``(k, d)`` matrix of unit rows.
    """
    s = sigma_s.double()
    values, vectors = torch.linalg.eigh(0.5 * (s + s.T))
    values = values.clamp_min(ridge * values.clamp_min(0).mean())
    whiten = vectors / values.sqrt()
    target = whiten.T @ sigma_t.double() @ whiten
    ratios, basis = torch.linalg.eigh(0.5 * (target + target.T))
    ratios, basis = ratios.flip(0)[:k], basis.flip(1)[:, :k]
    directions = (whiten @ basis).T
    return ratios.float(), torch.nn.functional.normalize(directions, dim=1).float()
