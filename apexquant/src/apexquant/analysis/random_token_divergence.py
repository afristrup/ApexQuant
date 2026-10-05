"""Data-free divergence probe: KL against the original model on random inputs."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def random_token_divergence(
    reference: nn.Module | torch.Tensor,
    candidate: nn.Module,
    vocab_size: int,
    *,
    batch: int = 4,
    seq_len: int = 256,
    seed: int = 0,
    forward: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> float:
    """Mean ``KL(reference || candidate)`` per position on uniformly random tokens.

    This is what makes the coefficient estimate of
    :func:`linear_error_coefficients` data-free. HIGGS make the same move: their
    calibrated variant fits the coefficients against perplexity on a calibration
    set, and their data-free variant replaces that with KL divergence between the
    pretrained and compressed models on randomly sampled token sequences,
    reporting that the method is "robust to being made completely data-free via
    random sampling".

    Two reasons the substitution is sound, and one reason to be careful. It is
    sound because the coefficients being fitted are properties of the pretrained
    weights rather than of any input distribution, and because KL against the
    original model cancels the shared difficulty that makes perplexity such a
    blunt instrument at these effect sizes. It needs care because random tokens
    are off-distribution, so the *absolute* divergence they produce is not a
    model-quality number and should never be reported as one; what is being used
    here is only the relative ordering across layers and noise levels, which is
    all the least-squares fit consumes.

    Args:
        reference: The unperturbed model, or its precomputed log-probabilities
            ``(batch, seq_len, vocab)`` for the same ``seed``/``batch``/
            ``seq_len``. Pass the tensor when sweeping many perturbations of one
            model: it holds one set of logits resident instead of a second copy
            of the weights, which is the difference between fitting a 12B
            reference on one card and not.
        candidate: The perturbed or quantized model. May be ``reference`` itself,
            in which case the result is ``0`` up to floating point.
        vocab_size: Token id range to sample from.
        batch: Sequences per probe.
        seq_len: Tokens per sequence.
        seed: RNG seed, so repeated calls are comparable rather than noisy. Fix
            it across a coefficient sweep or the fit measures sampling jitter.
        forward: Optional ``(model, input_ids) -> logits``. Defaults to calling
            the model and unwrapping a ``.logits`` attribute if present, which
            covers both plain modules and HuggingFace causal LMs.

    Returns:
        Mean KL in nats per position, ``>= 0``.
    """
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")

    def default_forward(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
        out = model(ids)
        return getattr(out, "logits", out)

    run = forward or default_forward

    device = next(candidate.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, vocab_size, (batch, seq_len), generator=g).to(device)

    with torch.no_grad():
        if isinstance(reference, torch.Tensor):
            ref_log = reference.to(device)
        else:
            ref_log = torch.log_softmax(run(reference, ids).float(), dim=-1)
        cand_log = torch.log_softmax(run(candidate, ids).float(), dim=-1)
        kl = (ref_log.exp() * (ref_log - cand_log)).sum(dim=-1)

    return float(kl.mean().clamp_min(0.0))
