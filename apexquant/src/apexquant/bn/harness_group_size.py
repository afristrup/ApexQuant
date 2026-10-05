"""The group size the vision harness gives a convolution of a given fan-in."""

from __future__ import annotations

#: Mirrors ``experiments/progressive_vision_imagenet.py``: the largest power of
#: two at most ``MAX_GROUP`` that *divides* the fan-in, because the pass reshapes
#: a row into whole groups. ``MIN_FANIN_GROUP`` is the smallest 2-bit group that
#: packs ``8 // 2 = 4`` indices into a byte.
MAX_GROUP = 256
MIN_FANIN_GROUP = 4


def harness_group_size(fan_in: int) -> int | None:
    """Return the group size the vision harness uses for ``fan_in``, or None.

    None means the layer is left dense, which is what happens to ResNet-50's
    7x7 stem (fan-in 147, no power-of-two divisor above one), to all sixteen of
    EfficientNet-B0's depthwise convolutions (fan-in 9 or 25) and to its five
    squeeze-excitation ``fc2`` layers (fan-in 6 or 10). That is 21 layers and
    3.62% of EfficientNet-B0, and it is why the depthwise pathology that
    dominates the published BatchNorm-staleness literature does not arise in this
    harness at all (section 8.1). A layer left dense has no residual and gets no
    correction.

    This duplicates the harness's ``group_for_fanin`` deliberately: the harness
    is the script the paper's numbers come from and is not to be edited.
    ``tests/test_qbn_apply.py`` asserts the two agree on every fan-in in range,
    so the duplication cannot drift unnoticed.

    Args:
        fan_in: Columns of the reshaped weight, ``C_in * kh * kw``.

    Returns:
        The group size, or None when no admissible group divides ``fan_in``.
    """
    gs = MAX_GROUP
    while gs >= MIN_FANIN_GROUP:
        if fan_in % gs == 0:
            return gs
        gs //= 2
    return None
