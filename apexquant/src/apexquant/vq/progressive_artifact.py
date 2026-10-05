"""Progressive quantisation of a weight matrix, keeping the codes.

`experiments/progressive_vector_stage_llm.reconstruct_stage` runs the recursion
and returns a dense reconstruction, which is all a perplexity measurement needs.
A checkpoint needs the artifact instead: the per-pass codes and scales that the
reconstruction was built from. This module produces those, and :func:`dequantize`
turns them back into a dense weight.

The recursion is the published one, unchanged: a fresh independent rotation per
pass, per-group L2 normalisation onto the sphere, quantise, reconstruct and
peel. Only the bookkeeping differs. The stage quantiser is any object with
``encode``/``decode``, so scalar and E8P share this path exactly, which is the
whole point of Proposition 2 being quantiser-agnostic.

Storage per weight is the `P (b + 16/g)` accounting the paper charges:

    codes   P x out x groups x C   -> b bits per weight per pass
    scale   P x out x groups       fp16   -> 16/g bits per weight per pass

The scale is stored fp16 deliberately. The reference implementation carries it
in fp32 in memory, which is a slightly richer artifact than the paper costs;
rounding to fp16 changes relative reconstruction error by under 0.003%, so the
accounting is honest either way, but the shipped artifact should be the one that
was costed. The peel uses the rounded scale, so the residual each pass sees is
the residual a decoder will see.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..rotation.inverse_random_rotation import inverse_random_rotation
from ..rotation.random_rotation import random_rotation

DEFAULT_GROUP_SIZE = 512
DEFAULT_SEED = 42


def group_size_for(in_features: int, group_size: int = DEFAULT_GROUP_SIZE) -> int:
    """The group size actually used for a given input width.

    Narrow layers fall back to the largest power of two that fits, matching the
    published pipeline.
    """
    if in_features < group_size:
        return 1 << (in_features.bit_length() - 1)
    return group_size


def quantize(weight: torch.Tensor, codec, passes: int,
             group_size: int = DEFAULT_GROUP_SIZE,
             seed: int = DEFAULT_SEED,
             row_block: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Progressively quantise ``weight``, returning its codes and scales.

    Args:
        weight: Dense ``(out_features, in_features)`` weight.
        codec: Stage quantiser exposing ``encode``/``decode``.
        passes: Number of residual passes ``P``.
        group_size: Coordinates per group.
        seed: Base rotation seed; pass ``p`` uses ``seed + p``.
        row_block: Quantise this many output rows at a time, which makes peak
            memory a tunable instead of a property of the widest weight. Output
            rows never interact mathematically -- the rotation acts along the
            group axis and the scale is per (row, group) -- but blocking is
            *not* bit-identical, and this docstring used to claim that it was.
            The rotation is a matmul, so its last bits move with the batch shape
            (9.3e-10 on Gemma-3-12B's layer-0 ``q_proj``), and a trellis Viterbi
            near-tie then falls the other way. Measured on that layer: scalar and
            E8P are unmoved, and the trellis leaves pass 1 identical while moving
            6 of 30720 pass-2 codes and 61 pass-3 codes, worth 1.2e-4 of relative
            reconstruction error. The scales are identical in every case, because
            a scale is the norm of the residual and the rotation is orthogonal.
            Prefix nesting is a separate property and is exact; see
            ``experiments/prefix_identity_gemma.py``, which measures both.
            ``None`` means the whole matrix.

    Returns:
        ``(codes, scale)``: codes ``(P, out, groups, C)`` in the codec's own
        dtype, and scale fp16 ``(P, out, groups)``.
    """
    if row_block is not None and row_block < weight.shape[0]:
        parts = [quantize(weight[i:i + row_block], codec, passes, group_size, seed)
                 for i in range(0, weight.shape[0], row_block)]
        return (torch.cat([c for c, _ in parts], dim=1),
                torch.cat([s for _, s in parts], dim=1))

    out_f, in_f = weight.shape
    gs = group_size_for(in_f, group_size)
    pad = (-in_f) % gs
    w = F.pad(weight, (0, pad)).to(torch.float32)
    ngroups = w.shape[1] // gs
    w = w.view(out_f, ngroups, gs)

    residual = w
    all_codes, all_scale = [], []
    for p in range(passes):
        s = seed + p
        rotated = random_rotation(residual, s)
        scale = rotated.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        scale = scale.to(torch.float16).to(torch.float32)
        codes = codec.encode((rotated / scale).reshape(-1, gs))
        codes = codes.reshape(out_f, ngroups, codes.shape[-1])
        recon_rot = scale * codec.decode(codes).reshape(out_f, ngroups, gs)
        residual = residual - inverse_random_rotation(recon_rot, s)

        all_codes.append(codes)
        all_scale.append(scale.squeeze(-1).to(torch.float16))

    return torch.stack(all_codes), torch.stack(all_scale)


def dequantize(codes: torch.Tensor, scale: torch.Tensor, codec,
               in_features: int, group_size: int = DEFAULT_GROUP_SIZE,
               seed: int = DEFAULT_SEED, passes: int | None = None,
               row_block: int | None = None) -> torch.Tensor:
    """Rebuild a dense weight from codes and scales.

    Taking ``passes`` below the stored depth decodes a *prefix*, which is the
    paper's nested-decoding property: the result is bit-identical to having
    quantised at that lower depth in the first place.

    Args:
        codes: ``(P, out, groups, C)`` codes.
        scale: Fp16 ``(P, out, groups)``.
        codec: The same stage quantiser used to encode.
        in_features: True input width, for slicing off the pad.
        group_size: Coordinates per group.
        seed: Base rotation seed.
        passes: Decode only the first ``passes`` passes; defaults to all.
        row_block: Rebuild this many output rows at a time, the mirror of
            :func:`quantize`'s argument. It bounds a *transient*: reconstructing
            a 3840x15360 layer costs 226 MiB in one go, which is what puts a 12B
            model over a 24 GB card during a dense load even though the model
            itself fits. Output rows never interact, but the inverse rotation is
            a matmul whose last bits move with the batch shape, so this is not
            bit-identical either: 1.0e-8 to 2.4e-8 absolute across the three
            stages on a layer whose largest weight is 5.3e-2, which is 5e-7 of
            full scale and far below the fp16 the scales are stored in.

    Returns:
        Float32 ``(out_features, in_features)`` reconstruction.
    """
    if row_block is not None and row_block < codes.shape[1]:
        return torch.cat([
            dequantize(codes[:, i:i + row_block], scale[:, i:i + row_block],
                       codec, in_features, group_size, seed, passes)
            for i in range(0, codes.shape[1], row_block)
        ], dim=0)
    P = codes.shape[0] if passes is None else passes
    out_f, ngroups = codes.shape[1], codes.shape[2]
    gs = group_size_for(in_features, group_size)

    acc = torch.zeros(out_f, ngroups, gs, device=codes.device, dtype=torch.float32)
    for p in range(P):
        s = seed + p
        sc = scale[p].to(torch.float32).unsqueeze(-1)
        recon_rot = sc * codec.decode(codes[p]).reshape(out_f, ngroups, gs)
        acc = acc + inverse_random_rotation(recon_rot, s)
    return acc.reshape(out_f, ngroups * gs)[:, :in_features].contiguous()


def effective_bits(passes: int, group_size: int = DEFAULT_GROUP_SIZE,
                   bits: int = 2) -> float:
    """Stored bits per weight: ``P (b + 16/g)``."""
    return passes * (bits + 16.0 / group_size)


def stored_bits_per_weight(out_features: int, in_features: int, passes: int,
                           group_size: int = DEFAULT_GROUP_SIZE,
                           bits: int = 2) -> float:
    """Stored bits per *real* weight, charging the group padding to the model.

    :func:`effective_bits` costs one stored coordinate, which is the right number
    whenever the width is a multiple of the group -- as it is for every language
    model in the paper. It is the wrong number when it is not. ``quantize`` pads
    a row up to a whole number of groups and stores codes for the pad, so a width
    of 768 at ``group_size=512`` pads to 1024 and a third of the artifact
    describes zeros: 4.06 bits per stored coordinate, but 5.42 per weight that
    exists. The two agree exactly when ``group_size`` divides ``in_features``.

    Report this one for any model whose widths are not the usual powers of two,
    and prefer a group size that divides the width when there is a choice -- 768
    at ``group_size=256`` costs 4.13, not 5.42, for the same recursion.

    Args:
        out_features: Rows of the dense weight.
        in_features: True input width, before padding.
        passes: Number of residual passes ``P``.
        group_size: Requested group size; the resolved one is used.
        bits: Rate of one stage, in bits per coordinate.

    Returns:
        Bits of stored artifact per weight actually present in the layer.
    """
    gs = group_size_for(in_features, group_size)
    ngroups = (in_features + gs - 1) // gs
    stored = passes * out_features * ngroups * (gs * bits + 16)
    return stored / (out_features * in_features)
