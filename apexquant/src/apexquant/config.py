"""Quantization configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ApexQuantConfig:
    """Settings for turboquantizing a weight matrix.

    The defaults reproduce the headline TurboQuant recipe: 4-bit Lloyd-Max
    scalar quantization applied to randomly-rotated, group-normalized weights.

    Attributes:
        group_size: Weights are split along the input dimension into contiguous
            groups of this many coordinates; each group gets its own rotation
            and scale. Must be a power of two (the fast Walsh-Hadamard rotation
            requires it). 128 is the value used in the TurboQuant paper.
        passes: Number of residual quantization passes. ``1`` is single-pass
            4-bit. ``2`` is the near-lossless "4+4" residual recipe: the second
            pass quantizes the error left by the first with a fresh rotation.
        seed: Base RNG seed for the random rotations. Pass ``i`` uses
            ``seed + i`` so every pass rotates into a different frame.
        bits: Codebook bit-width, one of 2, 4, 8. Indices pack ``8 // bits`` per
            byte; ``bits=2`` is the 16x-compression regime where the density-matched
            codebook matters most, ``bits=4`` the near-lossless default.
        codebook: ``"gaussian"`` uses the unit-variance Lloyd-Max codebook with
            per-group std normalization (the TurboQuant large-fan-in limit).
            ``"apex"`` uses ApexQuant's density-matched Lloyd-Max codebook for the
            exact post-rotation marginal ``f_d`` with per-group L2 normalization
            (``d = group_size``) -- strictly better at coarse bit-widths.
            ``"lp"`` fits the same density-matched codebook under HQQ's ``l_p``
            objective instead of MSE (see :func:`lp_levels`); it is worse in
            squared error by construction, so ``rho`` rises, and exists to test
            whether end-to-end KL falls anyway.
        lp_norm: Norm exponent used when ``codebook="lp"``. HQQ uses ``0.7``;
            ``2.0`` reproduces the MSE codebook.
        sinkhorn_iters: Sinkhorn-Knopp sweeps run on the weight *before* the
            first rotation, equalizing per-row and per-column variance as SINQ
            does. ``0`` disables it. The rotation mixes columns, so this is the
            only order in which a per-column scale means anything; applied first
            it leaves the theory intact, since the two diagonals are
            deterministic in the weight and so independent of the rotations.
    """

    group_size: int = 128
    passes: int = 1
    seed: int = 42
    bits: int = 4
    codebook: str = "gaussian"
    lp_norm: float = 0.7
    sinkhorn_iters: int = 0

    def __post_init__(self) -> None:
        if self.bits not in (2, 4, 8):
            raise ValueError("bits must be 2, 4, or 8")
        if self.codebook not in ("gaussian", "apex", "uniform", "lp"):
            raise ValueError("codebook must be 'gaussian', 'apex', 'uniform', or 'lp'")
        if self.lp_norm <= 0:
            raise ValueError("lp_norm must be > 0")
        if self.sinkhorn_iters < 0:
            raise ValueError("sinkhorn_iters must be >= 0")
        if self.group_size < 2 or (self.group_size & (self.group_size - 1)) != 0:
            raise ValueError("group_size must be a power of two >= 2")
        if self.group_size % (8 // self.bits) != 0:
            raise ValueError(f"group_size must be divisible by {8 // self.bits} to pack {self.bits}-bit")
        if self.passes < 1:
            raise ValueError("passes must be >= 1")
