"""The success criteria of sections 7.3 and 7.4, judged against made-up numbers."""

from __future__ import annotations

import pytest

from apexquant.bn.arm_result import ArmResult
from apexquant.bn.qbn_arms import (ARM_FOLDED, ARM_FP32, ARM_QBN_MEAN,
                                          ARM_QBN_MEAN_VAR, ARM_QBN_VAR,
                                          ARM_QUANTIZED, ARM_REESTIMATED,
                                          FP32_PASSES)
from apexquant.bn.qbn_criteria import (ARM4_MAX_LOSS, CONTROL_MODEL, FAIL,
                                              HEADLINE_MODEL, HEADLINE_PASSES,
                                              MATERIAL_POINTS, NOT_RUN, PASS,
                                              qbn_criteria)

CRITERIA = 5
GAP = "gap"
ARM4 = "arm 4"
ARM5 = "arm 5"
CONTROL = "bit-identical"
FOLD = "arm 7"


def _cell(model: str, passes: int, arm: int, top1: float,
          identical: bool | None = None) -> ArmResult:
    return ArmResult(model=model, passes=passes, arm=arm, top1=top1, kl=0.0,
                     identical_to_quantized=identical)


def _verdict(lines: list[str], needle: str) -> str:
    matching = [line for line in lines if needle in line]
    assert len(matching) == 1, f"{needle} matched {len(matching)} lines"
    return matching[0][1:5].strip()


def _headline(quantized: float, mean: float, ceiling: float) -> list[ArmResult]:
    return [_cell(HEADLINE_MODEL, HEADLINE_PASSES, ARM_QUANTIZED, quantized),
            _cell(HEADLINE_MODEL, HEADLINE_PASSES, ARM_QBN_MEAN, mean),
            _cell(HEADLINE_MODEL, HEADLINE_PASSES, ARM_REESTIMATED, ceiling)]


def test_an_empty_run_still_prints_every_criterion():
    lines = qbn_criteria([])
    assert len(lines) == CRITERIA
    assert all(line[1:5].strip() == NOT_RUN for line in lines)


def test_the_criteria_are_the_same_five_whatever_the_numbers():
    assert len(qbn_criteria(_headline(22.23, 50.0, 60.0))) == CRITERIA


def test_an_fp32_row_alone_decides_nothing_and_does_not_crash():
    lines = qbn_criteria([_cell(HEADLINE_MODEL, FP32_PASSES, ARM_FP32, 80.39)])
    assert all(line[1:5].strip() == NOT_RUN for line in lines)


def test_closing_exactly_half_the_gap_passes():
    lines = qbn_criteria(_headline(20.0, 50.0, 80.0))
    assert _verdict(lines, GAP) == PASS
    assert "50.0%" in [line for line in lines if GAP in line][0]


def test_closing_just_under_half_the_gap_fails():
    assert _verdict(qbn_criteria(_headline(20.0, 49.9, 80.0)), GAP) == FAIL


def test_a_ceiling_that_does_not_beat_the_quantized_arm_is_undecidable():
    lines = qbn_criteria(_headline(22.23, 30.0, 22.0))
    line = [entry for entry in lines if GAP in entry][0]
    assert line[1:5].strip() == NOT_RUN
    assert "7.4" in line


def test_the_headline_criterion_ignores_other_models_and_other_passes():
    other = [_cell("efficientnet_b0", HEADLINE_PASSES, ARM_QUANTIZED, 10.0),
             _cell("efficientnet_b0", HEADLINE_PASSES, ARM_QBN_MEAN, 60.0),
             _cell("efficientnet_b0", HEADLINE_PASSES, ARM_REESTIMATED, 70.0),
             _cell(HEADLINE_MODEL, HEADLINE_PASSES + 1, ARM_QUANTIZED, 10.0),
             _cell(HEADLINE_MODEL, HEADLINE_PASSES + 1, ARM_QBN_MEAN, 60.0),
             _cell(HEADLINE_MODEL, HEADLINE_PASSES + 1, ARM_REESTIMATED, 70.0)]
    assert _verdict(qbn_criteria(other), GAP) == NOT_RUN


def test_arm_4_may_lose_exactly_the_stated_points():
    results = [_cell("m", 2, ARM_QBN_MEAN, 50.0),
               _cell("m", 2, ARM_QBN_MEAN_VAR, 50.0 - ARM4_MAX_LOSS)]
    assert _verdict(qbn_criteria(results), ARM4) == PASS


def test_arm_4_losing_more_than_that_fails_and_says_what_to_ship():
    results = [_cell("m", 2, ARM_QBN_MEAN, 50.0),
               _cell("m", 2, ARM_QBN_MEAN_VAR, 50.0 - ARM4_MAX_LOSS - 0.05)]
    lines = qbn_criteria(results)
    assert _verdict(lines, ARM4) == FAIL
    assert "ship arm 3" in [line for line in lines if ARM4 in line][0]


def test_the_worst_cell_decides_arm_4_not_the_first_one():
    results = [_cell("m", 2, ARM_QBN_MEAN, 50.0), _cell("m", 2, ARM_QBN_MEAN_VAR, 50.0),
               _cell("m", 3, ARM_QBN_MEAN, 50.0), _cell("m", 3, ARM_QBN_MEAN_VAR, 40.0)]
    assert _verdict(qbn_criteria(results), ARM4) == FAIL


def test_a_variance_only_arm_that_helps_materially_rewrites_the_paper():
    results = [_cell("m", 2, ARM_QUANTIZED, 50.0),
               _cell("m", 2, ARM_QBN_VAR, 50.0 + MATERIAL_POINTS + 0.05)]
    lines = qbn_criteria(results)
    assert _verdict(lines, ARM5) == FAIL
    assert "alpha_c" in [line for line in lines if ARM5 in line][0]


def test_a_variance_only_arm_that_hurts_is_the_expected_control():
    results = [_cell("m", 2, ARM_QUANTIZED, 50.0), _cell("m", 2, ARM_QBN_VAR, 48.0)]
    assert _verdict(qbn_criteria(results), ARM5) == PASS


@pytest.mark.parametrize("arm", [ARM_QBN_MEAN, ARM_QBN_MEAN_VAR, ARM_QBN_VAR])
def test_a_control_model_arm_that_moved_is_an_implementation_bug(arm):
    results = [_cell(CONTROL_MODEL, 2, arm, 70.0, identical=False)]
    lines = qbn_criteria(results)
    assert _verdict(lines, CONTROL) == FAIL
    assert "LayerNorm" in [line for line in lines if CONTROL in line][0]


def test_a_control_model_that_did_not_move_passes():
    results = [_cell(CONTROL_MODEL, 2, arm, 70.0, identical=True)
               for arm in (ARM_QBN_MEAN, ARM_QBN_MEAN_VAR, ARM_QBN_VAR)]
    assert _verdict(qbn_criteria(results), CONTROL) == PASS


def test_the_control_criterion_ignores_the_headline_model():
    results = [_cell(HEADLINE_MODEL, 2, ARM_QBN_MEAN, 70.0, identical=False)]
    assert _verdict(qbn_criteria(results), CONTROL) == NOT_RUN


def test_a_comparison_that_was_never_made_is_not_a_pass():
    results = [_cell(CONTROL_MODEL, 2, ARM_QBN_MEAN, 70.0, identical=None)]
    assert _verdict(qbn_criteria(results), CONTROL) == NOT_RUN


def test_folding_that_changes_the_accuracy_materially_fails():
    results = [_cell("m", 2, ARM_QUANTIZED, 50.0),
               _cell("m", 2, ARM_FOLDED, 50.0 + MATERIAL_POINTS + 0.05)]
    lines = qbn_criteria(results)
    assert _verdict(lines, FOLD) == FAIL
    assert "2.1" in [line for line in lines if FOLD in line][0]


def test_folding_that_changes_nothing_is_the_predicted_no_op():
    results = [_cell("m", 2, ARM_QUANTIZED, 50.0), _cell("m", 2, ARM_FOLDED, 49.95)]
    assert _verdict(qbn_criteria(results), FOLD) == PASS
