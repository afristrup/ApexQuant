"""One cell of the section 7.3 table: a model, a pass count and an arm."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmResult:
    """What one arm scored on one model at one ``P``.

    Every field is a plain Python scalar so a finished cell can be written to a
    JSONL file and read back on a resumed run. The accuracy experiment is long
    enough that it will be interrupted, and a cell is the unit that survives.

    Attributes:
        model: timm model name.
        passes: Residual passes ``P``, or :data:`FP32_PASSES` for the fp32 arm,
            which has no ``P``.
        arm: Arm number, one of :data:`ARM_NUMBERS`.
        top1: Top-1 accuracy in **percent**, so that the success criteria of
            section 7.3, which are stated in points, subtract directly.
        kl: KL of this arm's logits against the same model at fp32, nats per
            image, zero for the fp32 arm itself.
        identical_to_quantized: Whether this arm's logits are bit-identical to
            arm 2's on the same model and ``P``, or None when arm 2 was not run
            in the same process. Section 7.3 requires True for ConvNeXt-Tiny's
            arms 3, 4 and 5: any difference there is a bug touching LayerNorm,
            and it is the cheapest bug detector available.
    """

    model: str
    passes: int
    arm: int
    top1: float
    kl: float
    identical_to_quantized: bool | None = None
