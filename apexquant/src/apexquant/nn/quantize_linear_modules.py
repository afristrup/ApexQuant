"""Recursively replace ``nn.Linear`` layers with fused ApexQuant layers."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from torch import nn

from ..config import ApexQuantConfig
from .linear_is_fusable import linear_is_fusable
from .apex_linear import ApexQuantLinear


def quantize_linear_modules(
    model: nn.Module,
    config: ApexQuantConfig,
    should_quantize: Callable[[str, nn.Linear], bool] | None = None,
    fuse_narrow: bool = False,
    passes_for: Callable[[str, nn.Linear], int] | None = None,
) -> int:
    """In-place swap every eligible ``nn.Linear`` in ``model`` for a fused layer.

    Walks the module tree and replaces children; nested modules are handled by
    recursion. A layer is skipped when it is not a fusion target under
    :func:`linear_is_fusable` (by default, narrower than the group size) or when
    ``should_quantize`` returns False.

    Args:
        model: The module to convert (mutated in place).
        config: Quantization settings.
        should_quantize: Optional predicate ``(qualified_name, linear) -> bool``
            to keep sensitive layers dense.
        fuse_narrow: Also fuse layers narrower than ``group_size`` (full
            coverage). The group shrinks to fit; only layers too small to pack
            are left dense. See :func:`linear_is_fusable`.
        passes_for: Optional ``(qualified_name, linear) -> int`` giving a
            per-layer residual-pass count (mixed precision). Defaults to
            ``config.passes`` for every layer. Spending extra passes only on the
            most sensitive layers beats a uniform count at equal average bits --
            see :func:`sensitivity_residual_allocation`.

    Returns:
        The number of layers replaced.
    """
    replaced = 0

    def recurse(module: nn.Module, prefix: str) -> None:
        nonlocal replaced
        for name, child in list(module.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if not linear_is_fusable(child.in_features, config, fuse_narrow):
                    continue
                if should_quantize is not None and not should_quantize(qualified, child):
                    continue
                layer_config = (config if passes_for is None
                                else replace(config, passes=passes_for(qualified, child)))
                setattr(module, name, ApexQuantLinear.from_linear(child, layer_config))
                replaced += 1
            else:
                recurse(child, qualified)

    recurse(model, "")
    return replaced
