"""Scalar progressive checkpoint, executed through the packed scalar kernels.

The scalar stage needs almost nothing new. A progressive artifact already stores
what :class:`~apexquant.quantize.quantized_weight.QuantizedWeight`
describes -- densely packed indices ``(P, out, groups, gs*bits/8)``, scales
``(P, out, groups)``, and a codebook shared across passes -- and
``ScalarCodec.decode`` is a bare ``levels[idx]`` gather with no codec-side scale
to fold, so the two vocabularies line up exactly. The whole stage-specific part
of this file is the constructor.

What it does add is a dtype guard, and it is not cosmetic.
:func:`fused_dequant_matmul` selects a Triton backend only for float32 input and
otherwise falls back to the torch reference, which reconstructs the weight. Hand
it a bf16 activation -- which is what Gemma must run in, since its hidden states
pass 6e5 and fp16 does not reach that -- and every layer silently takes the slow
path that this class exists to avoid. Casting on the way in and back on the way
out keeps the packed kernels engaged; it also makes this marginally *more*
accurate than the dense path, which rounds each reconstructed weight to the
model dtype first.

The trellis equivalent is
:class:`~apexquant.vq.progressive_trellis_linear.ProgressiveTrellisLinear`,
which has to carry its own kernels. This one does not: the scalar mat-vec, block
mat-vec and mid-M GEMM already exist, already fold every residual pass into one
launch, and are already tuned, so the dispatch in :func:`fused_dequant_matmul`
is left to do its job untouched.
"""

from __future__ import annotations

import torch

from ..kernels.fused_dequant_matmul import fused_dequant_matmul
from ..quantize.quantized_weight import QuantizedWeight
from .apex_linear import ApexQuantLinear


class ProgressiveScalarLinear(ApexQuantLinear):
    """``ApexQuantLinear`` that keeps the packed path under a non-fp32 model."""

    @classmethod
    def from_artifact(cls, codes: torch.Tensor, scale: torch.Tensor,
                      levels: torch.Tensor, in_features: int, group_size: int,
                      seed: int, bits: int,
                      bias: torch.Tensor | None) -> "ProgressiveScalarLinear":
        """Build from the tensors a progressive artifact stores.

        Args:
            codes: uint8 ``(P, out, groups, gs*bits/8)`` packed indices.
            scale: ``(P, out, groups)`` per-group scales; cast to float32.
            levels: The stage codebook, ``ScalarCodec.levels``.
            in_features: True input width, before the group padding.
            group_size: Coordinates per group.
            seed: Base rotation seed; pass ``p`` used ``seed + p``.
            bits: Index width.
            bias: Full-precision bias, or None.
        """
        return cls(QuantizedWeight(
            packed=codes, scale=scale.float(), levels=levels.float(),
            in_features=in_features, out_features=codes.shape[1],
            group_size=group_size, seed=seed, bits=bits,
        ), bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.col_scale is not None:
            x = x * self.col_scale.to(x.dtype)
        out = fused_dequant_matmul(x.float(), self._quantized_weight()).to(x.dtype)
        if self.row_scale is not None:
            out = out * self.row_scale.to(out.dtype)
        return out if self.bias is None else out + self.bias
