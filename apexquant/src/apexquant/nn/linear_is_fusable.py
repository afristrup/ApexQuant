"""Decide whether an ``nn.Linear`` of a given width can be turboquant-fused."""

from __future__ import annotations

from ..config import ApexQuantConfig
from ..quantize.resolve_group_size import resolve_group_size


def linear_is_fusable(in_features: int, config: ApexQuantConfig,
                      fuse_narrow: bool = False) -> bool:
    """Return whether a linear with ``in_features`` columns should be fused.

    Two policies, selected by ``fuse_narrow``:

    * ``False`` (default): only layers at least ``group_size`` wide are fused.
      Narrower layers -- e.g. a Clay-style patch projection (``in=64``) -- are
      left dense. This is the conservative default: the patch embedding is the
      model's input interface and the least redundant place to spend bits.
    * ``True`` (full coverage): every layer the pipeline can represent is fused.
      The group shrinks to the largest power of two ``<= in_features`` (see
      :func:`resolve_group_size`), so the only truly unfusable layers are those
      whose resolved group cannot pack ``8 // bits`` indices per byte (e.g. a
      2-coordinate row at 2-bit). This is what "fuse every kernel" needs.

    Args:
        in_features: Input dimension of the layer.
        config: Quantization settings (``group_size``, ``bits``).
        fuse_narrow: Fuse layers narrower than ``group_size`` too.

    Returns:
        True if the layer is a fusion target under the chosen policy.
    """
    if not fuse_narrow:
        return in_features >= config.group_size
    effective = resolve_group_size(in_features, config.group_size)
    per_byte = 8 // config.bits
    return effective >= 2 and effective % per_byte == 0
