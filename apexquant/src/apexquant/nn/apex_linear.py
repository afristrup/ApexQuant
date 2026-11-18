"""Drop-in fused 4-bit replacement for ``torch.nn.Linear``."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ApexQuantConfig
from ..kernels.fused_dequant_matmul import fused_dequant_matmul
from ..quantize.quantized_weight import QuantizedWeight
from ..quantize.require_unguarded import require_unguarded
from ..quantize.turboquantize_weight import turboquantize_weight


class ApexQuantLinear(nn.Module):
    """A linear layer whose weight lives as packed 4-bit indices.

    The forward pass calls the fused dequant+matmul kernel, so the float weight
    is never reconstructed. Bias (kept in full precision) is added afterwards.
    Swap it in anywhere ``nn.Linear`` is used; shapes and semantics match.

    Attributes are registered as buffers so ``.to(device)``, ``state_dict`` and
    ``load_state_dict`` all work out of the box.
    """

    def __init__(self, qw: QuantizedWeight, bias: torch.Tensor | None) -> None:
        super().__init__()
        require_unguarded(qw)
        self.in_features = qw.in_features
        self.out_features = qw.out_features
        self.group_size = qw.group_size
        self.seed = qw.seed
        self.bits = qw.bits

        self.register_buffer("packed", qw.packed)
        self.register_buffer("scale", qw.scale)
        self.register_buffer("levels", qw.levels)
        self.register_buffer("bias", bias if bias is not None else None)
        # SINQ's two diagonals. They never enter the packed kernel: the column
        # scale multiplies the activation and the row scale the output, which is
        # the same algebra as folding them into the neighbouring layers.
        self.register_buffer("row_scale", qw.row_scale)
        self.register_buffer("col_scale", qw.col_scale)

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: ApexQuantConfig) -> "ApexQuantLinear":
        """Build a fused layer from an existing dense ``nn.Linear``."""
        qw = turboquantize_weight(linear.weight.detach(), config)
        bias = None if linear.bias is None else linear.bias.detach().clone()
        return cls(qw, bias)

    def _quantized_weight(self) -> QuantizedWeight:
        return QuantizedWeight(
            packed=self.packed, scale=self.scale, levels=self.levels,
            in_features=self.in_features, out_features=self.out_features,
            group_size=self.group_size, seed=self.seed, bits=self.bits,
            row_scale=self.row_scale, col_scale=self.col_scale,
        )

    def get_extra_state(self) -> dict:
        """Non-tensor quant metadata, so a saved ``state_dict`` is self-describing.

        ``packed``/``scale``/``levels``/``bias`` are buffers and travel already;
        the scalar geometry (group size, seed, bit-width, shapes) is not, so we
        route it through torch's extra-state hook. It is then embedded in every
        ``torch.save(model.state_dict())`` and restored by ``load_state_dict`` --
        no need to remember the original :class:`ApexQuantConfig` to reload.
        """
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "group_size": self.group_size,
            "seed": self.seed,
            "bits": self.bits,
        }

    def set_extra_state(self, state: dict) -> None:
        """Restore the metadata saved by :meth:`get_extra_state`."""
        self.in_features = state["in_features"]
        self.out_features = state["out_features"]
        self.group_size = state["group_size"]
        self.seed = state["seed"]
        self.bits = state["bits"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = row * (W_normalized @ (col * x)), so the Sinkhorn diagonals cost two
        # elementwise passes and the packed kernel is untouched. In a fused
        # deployment the column scale is absorbed by the preceding layer, exactly
        # as SmoothQuant absorbs its own input-side diagonal.
        if self.col_scale is not None:
            x = x * self.col_scale.to(x.dtype)
        out = fused_dequant_matmul(x, self._quantized_weight())
        if self.row_scale is not None:
            out = out * self.row_scale.to(out.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"passes={self.packed.shape[0]}, group_size={self.group_size}, bits={self.bits}")
