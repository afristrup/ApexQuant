"""Turboquantize a dense weight matrix into packed 4-bit residual passes."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..codebook.apex_beta_levels import apex_beta_levels
from ..codebook.lloyd_max_levels import lloyd_max_levels
from ..codebook.lp_levels import lp_levels
from ..codebook.uniform_levels import uniform_levels
from ..config import ApexQuantConfig
from ..preprocess.sinkhorn_scales import sinkhorn_scales
from ..rotation.guarded_rotation import guarded_rotation
from ..rotation.inverse_guarded_rotation import inverse_guarded_rotation
from ..rotation.inverse_random_rotation import inverse_random_rotation
from ..rotation.random_rotation import random_rotation
from .pack_bits import pack_bits
from .quantize_to_indices import quantize_to_indices
from .quantized_weight import QuantizedWeight
from .resolve_group_size import resolve_group_size


def turboquantize_weight(weight: torch.Tensor, config: ApexQuantConfig) -> QuantizedWeight:
    """Quantize ``weight`` ``(out, in)`` with the ApexQuant recipe.

    Optionally Sinkhorn-normalize the whole matrix first
    (``config.sinkhorn_iters``), then per group of ``group_size`` input
    coordinates, and per residual pass:

        1. rotate the (current residual of the) group into a fresh random frame,
           where the coordinates are near-Gaussian;
        2. normalize by the group's std so it matches the unit-variance codebook;
        3. assign each coordinate to its nearest Lloyd-Max level;
        4. reconstruct and subtract, leaving a smaller residual for the next pass.

    The stored ``scale`` folds together the group std and the rotation frame so
    the fused kernel can work entirely in rotated codebook space and never
    materialize the float weight (see :func:`dequantize_weight`).

    Args:
        weight: Dense weight ``(out_features, in_features)``.
        config: Quantization settings.

    Returns:
        A :class:`QuantizedWeight`.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be 2-D (out_features, in_features)")

    out_features, in_features = weight.shape
    # Shrink the group to the weight when it is narrower than group_size: padding
    # a d=64 row up to 128 would otherwise double its stored bits. The shared
    # resolver keeps this a power of two so the fast Hadamard rotation applies.
    gs = resolve_group_size(in_features, config.group_size)

    # The codebook and its matching per-group normalizer are chosen together:
    #   gaussian -> unit-variance Lloyd-Max + std normalization (large-d limit);
    #   apex     -> density-matched Lloyd-Max for f_gs + L2 normalization (exact),
    #               with coordinates living on the sphere in [-1, 1].
    if config.codebook in ("apex", "lp"):
        if config.codebook == "apex":
            levels = apex_beta_levels(config.bits, gs, device=weight.device)
        else:
            levels = lp_levels(config.bits, gs, p=config.lp_norm, device=weight.device)

        def normalizer(v: torch.Tensor) -> torch.Tensor:  # L2 -> unit sphere
            return v.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
    elif config.codebook == "uniform":
        levels = uniform_levels(config.bits, device=weight.device)

        def normalizer(v: torch.Tensor) -> torch.Tensor:  # absmax -> [-1, 1]
            return v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    else:
        levels = lloyd_max_levels(config.bits, device=weight.device)

        def normalizer(v: torch.Tensor) -> torch.Tensor:  # std -> unit variance
            return v.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)

    # SINQ preprocessing, before the rotation and therefore before the grouping.
    # Order matters: the Hadamard mixes columns, so a per-column scale applied
    # afterwards would be meaningless. Both diagonals are deterministic in the
    # weight, hence independent of every pass's rotation, so re-isotropization
    # and the geometric decay law carry over unchanged into the normalized frame.
    row_scale = col_scale = None
    source = weight
    if config.sinkhorn_iters > 0:
        row_scale, col_scale = sinkhorn_scales(weight, config.sinkhorn_iters)
        source = weight.to(torch.float32) / (row_scale[:, None] * col_scale)

    # Pad the input dim up to a whole number of groups, then view as groups.
    pad = (-in_features) % gs
    w = F.pad(source, (0, pad)).to(torch.float32)
    num_groups = w.shape[1] // gs
    w = w.view(out_features, num_groups, gs)

    residual = w
    packed_passes: list[torch.Tensor] = []
    scale_passes: list[torch.Tensor] = []
    guard_passes: list[torch.Tensor] = []

    for p in range(config.passes):
        seed = config.seed + p
        if config.guard_threshold is None:
            rotated = random_rotation(residual, seed)  # (out, groups, gs)
        else:
            rotated, flags = guarded_rotation(residual, seed, config.guard_threshold)
            guard_passes.append(flags)
        # Normalize each group onto the codebook's domain (unit variance for
        # gaussian, unit sphere for apex), then assign nearest levels.
        scale = normalizer(rotated)
        indices = quantize_to_indices(rotated / scale, levels)  # uint8 (out, groups, gs)

        packed_passes.append(pack_bits(indices, config.bits))
        scale_passes.append(scale.squeeze(-1))  # (out, groups)

        # Reconstruct this pass and peel it off the residual.
        recon_rot = scale * levels[indices.long()]
        if config.guard_threshold is None:
            residual = residual - inverse_random_rotation(recon_rot, seed)
        else:
            residual = residual - inverse_guarded_rotation(recon_rot, seed, flags)

    guard = torch.stack(guard_passes, dim=0) if guard_passes else None
    if guard is not None and not guard.any():
        guard = None

    return QuantizedWeight(
        packed=torch.stack(packed_passes, dim=0),
        scale=torch.stack(scale_passes, dim=0),
        levels=levels,
        in_features=in_features,
        out_features=out_features,
        group_size=gs,
        seed=config.seed,
        bits=config.bits,
        row_scale=row_scale,
        col_scale=col_scale,
        guard=guard,
    )
