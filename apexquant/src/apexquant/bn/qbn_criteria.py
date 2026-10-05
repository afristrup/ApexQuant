"""Judge a run of the seven arms against the criteria fixed before it ran."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .arm_result import ArmResult
from .qbn_arms import (ARM_FOLDED, ARM_QBN_MEAN, ARM_QBN_MEAN_VAR, ARM_QBN_VAR,
                       ARM_QUANTIZED, ARM_REESTIMATED, FP32_PASSES)

#: Section 7.3, criterion 1, on the headline model at the headline pass count.
HEADLINE_MODEL = "resnet50"
HEADLINE_PASSES = 2
GAP_CLOSED_MIN = 0.50

#: Section 7.3, criterion 2.
ARM4_MAX_LOSS = 0.3

#: Section 7.3, criterion 3 and section 7.4's arm-7 check both turn on the word
#: "materially", for which the document fixes no number. This reuses criterion
#: 2's 0.3 points, which is the smallest available choice; it is the one
#: threshold here that was not fixed in advance, and the evidence string prints
#: the raw difference so a reader can apply their own.
MATERIAL_POINTS = 0.3

#: Section 7.3, the negative control: ConvNeXt-Tiny has no BatchNorm, so the
#: three QBN arms must come out bit-identical to arm 2.
CONTROL_MODEL = "convnext_tiny"
CONTROL_ARMS = (ARM_QBN_MEAN, ARM_QBN_MEAN_VAR, ARM_QBN_VAR)

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "n/a"
PERCENT = 100.0
#: Comparison slack, so that a difference of exactly the threshold passes
#: whatever way the subtraction rounds. Not a change to any threshold.
POINT_EPS = 1e-6

Cell = tuple[str, int]


def qbn_criteria(results: Sequence[ArmResult]) -> list[str]:
    """Return one verdict line per success criterion of sections 7.3 and 7.4.

    The criteria were fixed before the experiment ran, which is the only reason
    they are worth anything, so they are printed whether or not the run can
    decide them: calling this with no results at all gives the same five lines
    with every verdict ``n/a``, which is what the header of a run prints.

    Args:
        results: The finished cells, in any order. Cells from a previous run
            that were read back from disk count.

    Returns:
        Five lines, each ``[verdict] criterion: evidence``.
    """
    top1 = {(r.model, r.passes, r.arm): r.top1 for r in results}
    same = {(r.model, r.passes, r.arm): r.identical_to_quantized for r in results}
    cells = sorted({(r.model, r.passes) for r in results
                    if r.passes != FP32_PASSES})
    return [
        _gap_closed(top1),
        _arm4_not_much_worse(top1, cells),
        _arm5_is_only_a_control(top1, cells),
        _control_model_is_untouched(same),
        _folding_is_a_no_op(top1, cells),
    ]


def _line(verdict: str, criterion: str, evidence: str) -> str:
    """Format one criterion's verdict."""
    return f"[{verdict:^4}] {criterion}: {evidence}"


def _gap_closed(top1: Mapping[tuple[str, int, int], float]) -> str:
    """Criterion 1: arm 3 recovers half the ceiling's headroom on the headline."""
    criterion = (f"arm 3 closes >= {GAP_CLOSED_MIN * PERCENT:.0f}% of the arm 2 to "
                 f"arm 6 gap ({HEADLINE_MODEL}, P={HEADLINE_PASSES})")
    key = (HEADLINE_MODEL, HEADLINE_PASSES)
    quantized = top1.get((*key, ARM_QUANTIZED))
    corrected = top1.get((*key, ARM_QBN_MEAN))
    ceiling = top1.get((*key, ARM_REESTIMATED))
    if quantized is None or corrected is None or ceiling is None:
        return _line(NOT_RUN, criterion, "arms 2, 3 and 6 not all run")
    gap = ceiling - quantized
    if gap <= 0.0:
        return _line(NOT_RUN, criterion,
                     f"arm 6 ({ceiling:.2f}) did not beat arm 2 ({quantized:.2f}), so "
                     "there is no gap to close and section 7.4 refutes the mechanism "
                     "rather than the method")
    closed = (corrected - quantized) / gap
    verdict = PASS if closed >= GAP_CLOSED_MIN else FAIL
    return _line(verdict, criterion,
                 f"closed {closed * PERCENT:.1f}% of {gap:.2f} points "
                 f"(arm 2 {quantized:.2f}, arm 3 {corrected:.2f}, arm 6 {ceiling:.2f})")


def _arm4_not_much_worse(top1: Mapping[tuple[str, int, int], float],
                         cells: Sequence[Cell]) -> str:
    """Criterion 2: the variance half may cost a little, not a lot."""
    criterion = f"arm 4 is not worse than arm 3 by more than {ARM4_MAX_LOSS} points"
    losses = [(top1[(m, p, ARM_QBN_MEAN)] - top1[(m, p, ARM_QBN_MEAN_VAR)], m, p)
              for m, p in cells
              if (m, p, ARM_QBN_MEAN) in top1 and (m, p, ARM_QBN_MEAN_VAR) in top1]
    if not losses:
        return _line(NOT_RUN, criterion, "no model has both arm 3 and arm 4")
    loss, model, passes = max(losses)
    verdict = PASS if loss <= ARM4_MAX_LOSS + POINT_EPS else FAIL
    tail = ("" if verdict == PASS else
            "; ship arm 3 and report the variance term as a measured negative result")
    return _line(verdict, criterion,
                 f"worst loss {loss:+.2f} points at {model} P={passes}{tail}")


def _arm5_is_only_a_control(top1: Mapping[tuple[str, int, int], float],
                            cells: Sequence[Cell]) -> str:
    """Criterion 3: the variance term alone must not be the mechanism."""
    criterion = "arm 5 is worse than arm 2 or close to it"
    gains = [(top1[(m, p, ARM_QBN_VAR)] - top1[(m, p, ARM_QUANTIZED)], m, p)
             for m, p in cells
             if (m, p, ARM_QBN_VAR) in top1 and (m, p, ARM_QUANTIZED) in top1]
    if not gains:
        return _line(NOT_RUN, criterion, "no model has both arm 2 and arm 5")
    gain, model, passes = max(gains)
    verdict = PASS if gain <= MATERIAL_POINTS + POINT_EPS else FAIL
    tail = ("" if verdict == PASS else
            "; the mechanism is then the coherent contraction rather than the mean "
            "shift, and the paper should be rewritten around alpha_c = -rho^{2P}")
    return _line(verdict, criterion,
                 f"best gain {gain:+.2f} points at {model} P={passes}{tail}")


def _control_model_is_untouched(same: Mapping[tuple[str, int, int], bool | None]) -> str:
    """Criterion 4: the model with no BatchNorm must not move at all."""
    criterion = (f"{CONTROL_MODEL} arms {CONTROL_ARMS} are bit-identical to arm 2")
    flags = {key: value for key, value in same.items()
             if key[0] == CONTROL_MODEL and key[2] in CONTROL_ARMS
             and value is not None}
    if not flags:
        return _line(NOT_RUN, criterion,
                     "arm 2 and the QBN arms were not run in the same process")
    broken = sorted(key for key, value in flags.items() if not value)
    if broken:
        return _line(FAIL, criterion,
                     f"{len(broken)} of {len(flags)} arms moved, first at "
                     f"{broken[0]}; an implementation bug is touching LayerNorm")
    return _line(PASS, criterion, f"{len(flags)} arms compared, all identical")


def _folding_is_a_no_op(top1: Mapping[tuple[str, int, int], float],
                        cells: Sequence[Cell]) -> str:
    """Section 7.4: arm 7 must equal arm 2, because section 2.1 proves it does."""
    criterion = f"arm 7 is within {MATERIAL_POINTS} points of arm 2 (predicted no-op)"
    gaps = [(abs(top1[(m, p, ARM_FOLDED)] - top1[(m, p, ARM_QUANTIZED)]), m, p)
            for m, p in cells
            if (m, p, ARM_FOLDED) in top1 and (m, p, ARM_QUANTIZED) in top1]
    if not gaps:
        return _line(NOT_RUN, criterion, "no model has both arm 2 and arm 7")
    gap, model, passes = max(gaps)
    verdict = PASS if gap <= MATERIAL_POINTS + POINT_EPS else FAIL
    tail = ("" if verdict == PASS else
            "; the harness is doing something other than what section 2.1 proves it "
            "does, and no other result can be trusted until that is explained")
    return _line(verdict, criterion,
                 f"worst difference {gap:.2f} points at {model} P={passes}{tail}")
