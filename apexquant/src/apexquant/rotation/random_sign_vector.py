"""Deterministic random +/-1 diagonal used to randomize the Hadamard rotation."""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=256)
def _cached_signs(dim: int, seed: int, device_str: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bits = torch.randint(0, 2, (dim,), generator=generator, dtype=torch.int64)
    signs = (bits * 2 - 1).to(torch.float32)
    return signs.to(device_str)


def random_sign_vector(dim: int, seed: int, device=None) -> torch.Tensor:
    """Return a reproducible vector of +/-1 of length ``dim``.

    The randomized Hadamard rotation is ``R = WHT @ diag(s)`` with ``s`` a random
    sign vector. Making ``s`` a pure function of ``(dim, seed)`` means the encoder
    and the fused kernel derive the *same* rotation without storing it. We always
    generate on CPU (so the stream is device-independent) and then move.

    The vector is a pure function of ``(dim, seed, device)`` and is used on every
    forward (once per layer per pass), so it is cached -- regenerating it on CPU
    and copying to the GPU each call was ~11% of the rotation cost. The cached
    tensor is read-only (``random_rotation`` multiplies, never mutates).

    Args:
        dim: Vector length (the group size).
        seed: RNG seed; different passes pass different seeds.
        device: Optional device for the returned tensor.

    Returns:
        Float32 tensor of +/-1 with shape ``(dim,)``.
    """
    return _cached_signs(int(dim), int(seed), str(device) if device is not None else "cpu")
