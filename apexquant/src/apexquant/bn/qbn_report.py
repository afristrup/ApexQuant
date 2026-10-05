"""What QBN did, or declined to do, to one convolution's BatchNorm."""

from __future__ import annotations

from dataclasses import dataclass

from .qbn_correction_result import QbnCorrection


@dataclass(frozen=True)
class QbnReport:
    """One layer's row of the diagnostics table of section 4.3.

    These diagnostics are the publishable part of the method even if the
    accuracy result is null, so every layer gets a row, including the ones left
    alone.

    Attributes:
        conv: Dotted name of the convolution.
        bn: Dotted name of the BatchNorm whose buffers were edited, or None.
        case: Input-moment case letter of section 4.4.
        mode: Which buffers this pass was allowed to write.
        corrected: Whether the buffers were edited at all.
        skipped: Why they were not, one of the ``SKIP_`` reasons of
            :mod:`qbn_guards`, or ``SKIP_NONE``.
        channels: Output channels of the convolution.
        correction: The per-channel numbers, or None for a skipped layer.
    """

    conv: str
    bn: str | None
    case: str
    mode: str
    corrected: bool
    skipped: str
    channels: int
    correction: QbnCorrection | None
