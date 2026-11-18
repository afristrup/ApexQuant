"""Container for a turboquantized weight matrix."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantizedWeight:
    """A weight matrix stored as packed 4-bit indices plus per-group scales.

    The original weight ``W`` has shape ``(out_features, in_features)``. Along the
    input axis it is padded to a multiple of ``group_size`` and split into
    ``num_groups`` groups. Each residual pass stores one index per coordinate and
    one scale per (output row, group).

    Attributes:
        packed: uint8 tensor ``(passes, out_features, num_groups, group_size // 2)``
            of nibble-packed codebook indices.
        scale: float32 tensor ``(passes, out_features, num_groups)``; the weight
            energy carried by each group in each pass.
        levels: 1-D codebook shared by all passes.
        in_features: True input dim before padding (for slicing off the pad).
        out_features: Output dim.
        group_size: Coordinates per group.
        seed: Base rotation seed; pass ``p`` uses ``seed + p``.
        row_scale: Optional ``(out_features,)`` output-side Sinkhorn scale.
        col_scale: Optional ``(in_features,)`` input-side Sinkhorn scale. The
            reconstruction is ``row_scale[:, None] * normalized * col_scale``,
            so at inference the column diagonal scales the *activation* and the
            row diagonal scales the output, and the packed kernels never see
            either. Both are ``None`` unless the artifact was built with
            ``ApexQuantConfig(sinkhorn_iters > 0)``.
        guard: Optional bool tensor ``(passes, out_features, num_groups)`` marking
            groups rotated twice by the Hadamard guard. ``None`` when the guard
            is off or never fired, so such artifacts equal unguarded ones.
    """

    packed: torch.Tensor
    scale: torch.Tensor
    levels: torch.Tensor
    in_features: int
    out_features: int
    group_size: int
    seed: int
    bits: int = 4
    row_scale: torch.Tensor | None = None
    col_scale: torch.Tensor | None = None
    rotations: torch.Tensor | None = None
    guard: torch.Tensor | None = None
    graph_safe: bool = False

    @property
    def passes(self) -> int:
        return self.packed.shape[0]

    @property
    def num_groups(self) -> int:
        return self.packed.shape[2]

    @property
    def device(self) -> torch.device:
        return self.packed.device

    def to(self, device) -> "QuantizedWeight":
        """Move every stored tensor to ``device`` (returns a new container)."""
        return QuantizedWeight(
            packed=self.packed.to(device),
            scale=self.scale.to(device),
            levels=self.levels.to(device),
            in_features=self.in_features,
            out_features=self.out_features,
            group_size=self.group_size,
            seed=self.seed,
            bits=self.bits,
            row_scale=None if self.row_scale is None else self.row_scale.to(device),
            col_scale=None if self.col_scale is None else self.col_scale.to(device),
            rotations=None if self.rotations is None else self.rotations.to(device),
            guard=None if self.guard is None else self.guard.to(device),
            graph_safe=self.graph_safe,
        )

    def nbytes(self) -> int:
        """Total bytes of the stored payload (indices + scales + codebook)."""
        total = (
            self.packed.numel() * self.packed.element_size()
            + self.scale.numel() * self.scale.element_size()
            + self.levels.numel() * self.levels.element_size()
        )
        # Two vectors per matrix, so this is O(out + in) against O(out * in)
        # indices: SINQ's "zero to negligible" overhead, counted rather than
        # asserted, and it does show up in average_bits() on narrow layers.
        for extra in (self.row_scale, self.col_scale):
            if extra is not None:
                total += extra.numel() * extra.element_size()
        if self.guard is not None:
            total += (self.guard.numel() + 7) // 8
        return total

    def decode_views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Kernel-ready views ``(words, scale, levels)``, built once and cached.

        ``words`` is the packed payload flattened to int32 rows
        ``(passes, out_features, K*bits/32)``; ``scale`` and ``levels`` are the
        contiguous tensors the Triton kernels take. Rebuilding these per forward
        is pure host overhead that dominates batch-one decode on small layers.
        """
        views = getattr(self, "_decode_views", None)
        if views is None or views[0].device != self.packed.device:
            words = (self.packed.reshape(self.passes, self.out_features, -1)
                     .contiguous().view(torch.int32))
            views = (words, self.scale.contiguous(), self.levels.contiguous())
            object.__setattr__(self, "_decode_views", views)
        return views
