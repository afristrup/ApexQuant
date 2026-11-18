"""Distribution-shift metrics for reference versus quantized models."""

from __future__ import annotations

import torch


def distribution_shift(reference: torch.Tensor, candidate: torch.Tensor, *,
                       logits: bool = False, eps: float = 1e-8) -> dict[str, float]:
    """Measure output or activation drift without fitting on the inputs.

    ``reference`` and ``candidate`` are evaluated on the same held-out inputs.
    If ``logits`` is true, rows are treated as categorical distributions and
    the result includes Jensen-Shannon divergence, total variation, and top-1
    agreement. Otherwise rows are treated as activation samples and the result
    reports first-, second-, and vector-moment drift.
    """
    if reference.shape != candidate.shape or reference.ndim < 2:
        raise ValueError("reference and candidate must have the same rank >= 2")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("reference and candidate must be finite")
    ref = reference.float().reshape(-1, reference.shape[-1])
    cand = candidate.float().reshape_as(ref)
    delta = cand - ref
    result = {
        "mean_abs_delta": float(delta.abs().mean()),
        "rms_delta": float(delta.square().mean().sqrt()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(ref, cand, dim=-1).mean()),
    }
    if not logits:
        ref_mean, cand_mean = ref.mean(0), cand.mean(0)
        ref_var, cand_var = ref.var(0, unbiased=False), cand.var(0, unbiased=False)
        result.update({
            "relative_mean_shift": float((cand_mean - ref_mean).norm() / ref_mean.norm().clamp_min(eps)),
            "relative_std_shift": float((cand_var.sqrt() - ref_var.sqrt()).norm() / ref_var.sqrt().norm().clamp_min(eps)),
        })
        return result

    ref_prob = torch.softmax(ref, dim=-1).clamp_min(eps)
    cand_prob = torch.softmax(cand, dim=-1).clamp_min(eps)
    midpoint = (ref_prob + cand_prob).mul(0.5)
    ref_kl = (ref_prob * (ref_prob.log() - midpoint.log())).sum(-1)
    cand_kl = (cand_prob * (cand_prob.log() - midpoint.log())).sum(-1)
    result.update({
        "js_divergence": float((0.5 * (ref_kl + cand_kl)).mean()),
        "total_variation": float((0.5 * (ref_prob - cand_prob).abs().sum(-1)).mean()),
        "top1_agreement": float((ref_prob.argmax(-1) == cand_prob.argmax(-1)).float().mean()),
    })
    return result
