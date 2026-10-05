"""The seven-arm accuracy script's command line, table and resume file."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

from apexquant.bn.arm_result import ArmResult
from apexquant.bn.qbn_arms import (ARM_FOLDED, ARM_FP32, ARM_NUMBERS,
                                          ARM_QUANTIZED, FP32_PASSES)

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" \
    / "qbn_accuracy_imagenet.py"
MISSING_ARGUMENT = 2
CONTROL_MODEL = "convnext_tiny"
HEADLINE_MODEL = "resnet50"
CRITERIA = 5
IMAGES = 4
SIDE = 64
CLASSES = 1000


def _script():
    """Import the experiment script without running it."""
    experiments = str(SCRIPT.parent)
    sys.path.insert(0, experiments)
    try:
        spec = importlib.util.spec_from_file_location("qbn_accuracy_imagenet", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(experiments)


SCRIPT_MODULE = _script()


def test_help_works_without_imagenet_or_a_gpu():
    done = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          capture_output=True, text=True, check=True)
    for flag in ("--arms", "--models", "--passes", "--calib-dir", "--out"):
        assert flag in done.stdout


def test_the_validation_directory_is_required():
    done = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert done.returncode == MISSING_ARGUMENT


def test_the_three_models_are_the_two_falsification_models_and_the_control():
    assert SCRIPT_MODULE.MODELS == (HEADLINE_MODEL, "efficientnet_b0", CONTROL_MODEL)
    assert SCRIPT_MODULE.PASSES == (2, 3)


def test_arm_6_gets_the_2048_images_the_document_budgets_for_it():
    assert SCRIPT_MODULE.CALIB_IMAGES == 2048


def test_a_selection_comes_back_in_the_canonical_order():
    assert SCRIPT_MODULE._selected("7,2,3", ARM_NUMBERS, int, "arm") == (2, 3, 7)


def test_a_repeated_selection_is_taken_once():
    assert SCRIPT_MODULE._selected("2,2", ARM_NUMBERS, int, "arm") == (2,)


@pytest.mark.parametrize("text", ["", "8", "0", "2,9"])
def test_an_unknown_or_empty_selection_fails_fast(text):
    with pytest.raises(ValueError, match="arm selection"):
        SCRIPT_MODULE._selected(text, ARM_NUMBERS, int, "arm")


def test_every_arm_is_selectable_on_its_own():
    for arm in ARM_NUMBERS:
        assert SCRIPT_MODULE._selected(str(arm), ARM_NUMBERS, int, "arm") == (arm,)


def test_the_fp32_cell_comes_first_and_carries_no_pass_count():
    cells = SCRIPT_MODULE._cells(ARM_NUMBERS, (2, 3))
    assert cells[0] == (FP32_PASSES, ARM_FP32)
    assert cells[1:] == [(p, arm) for p in (2, 3)
                         for arm in ARM_NUMBERS if arm != ARM_FP32]


def test_dropping_the_fp32_arm_drops_only_that_cell():
    arms = tuple(arm for arm in ARM_NUMBERS if arm != ARM_FP32)
    cells = SCRIPT_MODULE._cells(arms, (2,))
    assert cells == [(2, arm) for arm in arms]


def test_one_arm_and_one_pass_is_one_cell():
    assert SCRIPT_MODULE._cells((ARM_FOLDED,), (3,)) == [(3, ARM_FOLDED)]


def test_a_missing_resume_file_is_not_an_error(tmp_path):
    assert SCRIPT_MODULE._load_done(str(tmp_path / "absent.jsonl")) == {}
    assert SCRIPT_MODULE._load_done(None) == {}


def test_a_finished_cell_survives_the_round_trip(tmp_path):
    path = str(tmp_path / "cells.jsonl")
    written = ArmResult(model=HEADLINE_MODEL, passes=2, arm=ARM_QUANTIZED,
                        top1=22.23, kl=3.5, identical_to_quantized=None)
    SCRIPT_MODULE._append(path, written)
    assert SCRIPT_MODULE._load_done(path) == {(HEADLINE_MODEL, 2, ARM_QUANTIZED): written}


def test_a_completed_run_leaves_every_cell_resumable(tmp_path):
    path = str(tmp_path / "cells.jsonl")
    cells = SCRIPT_MODULE._cells(ARM_NUMBERS, (2, 3))
    for passes, arm in cells:
        SCRIPT_MODULE._append(path, ArmResult(model=CONTROL_MODEL, passes=passes,
                                              arm=arm, top1=1.0, kl=0.0,
                                              identical_to_quantized=True))
    done = SCRIPT_MODULE._load_done(path)
    assert all((CONTROL_MODEL, passes, arm) in done for passes, arm in cells)


def test_a_resume_file_from_a_different_schema_fails_fast(tmp_path):
    path = tmp_path / "cells.jsonl"
    path.write_text('{"model": "resnet50", "passes": 2, "arm": 2, "top-1": 22.2}\n')
    with pytest.raises(TypeError):
        SCRIPT_MODULE._load_done(str(path))


def test_the_row_of_a_quantized_cell_says_which_tier_and_whether_it_is_data_free(capsys):
    SCRIPT_MODULE._print_row(ArmResult(model=HEADLINE_MODEL, passes=2,
                                       arm=ARM_QUANTIZED, top1=22.23, kl=3.5))
    row = capsys.readouterr().out
    assert "| resnet50 | P=2 | 2 quantized | yes | 22.23% |" in row


def test_the_fp32_row_has_no_pass_count(capsys):
    SCRIPT_MODULE._print_row(ArmResult(model=HEADLINE_MODEL, passes=FP32_PASSES,
                                       arm=ARM_FP32, top1=80.39, kl=0.0))
    assert "| resnet50 | fp32 | 1 fp32 | n/a |" in capsys.readouterr().out


@pytest.mark.parametrize("identical,shown", [(True, "yes"), (False, "no"), (None, "")])
def test_the_control_column_reports_the_comparison_with_arm_2(capsys, identical, shown):
    SCRIPT_MODULE._print_row(ArmResult(model=CONTROL_MODEL, passes=2, arm=3,
                                       top1=70.0, kl=0.1,
                                       identical_to_quantized=identical))
    assert capsys.readouterr().out.rstrip().endswith(f"| {shown} |")


@pytest.fixture
def cpu_run(monkeypatch):
    """Run ``main`` end to end on CPU with a small stand-in model and no images.

    The GPU belongs to another job, so every path that would reach it is
    redirected here: ``DEV``, the forward pass, and the re-estimator. What is
    left is the whole of ``main``, which is otherwise the only part of this
    script that no test touches.
    """
    import timm
    import torch

    create_model = timm.create_model
    built: list[str] = []
    calibrated: list[int] = []

    def stand_in(name, pretrained=False, **kwargs):
        built.append(name)
        torch.manual_seed(0)
        return create_model("resnet18", pretrained=False, **kwargs).eval()

    def images(root, limit, transform, skip=0):
        torch.manual_seed(limit + skip)
        return torch.randn(IMAGES, 3, SIDE, SIDE), torch.randint(0, CLASSES, (IMAGES,))

    monkeypatch.setattr(timm, "create_model", stand_in)
    monkeypatch.setattr(SCRIPT_MODULE, "DEV", "cpu")
    monkeypatch.setattr(SCRIPT_MODULE, "load_val", images)
    monkeypatch.setattr(SCRIPT_MODULE, "logits_of",
                        lambda model, x: model(x).detach())
    monkeypatch.setattr(SCRIPT_MODULE, "reestimate_batchnorm",
                        lambda model, x: calibrated.append(1))
    return built, calibrated


def test_main_runs_every_cell_and_records_each_one(cpu_run, monkeypatch, tmp_path, capsys):
    built, calibrated = cpu_run
    path = tmp_path / "cells.jsonl"
    monkeypatch.setattr(sys, "argv", ["qbn", str(tmp_path), "--models", HEADLINE_MODEL,
                                      "--passes", "2", "--out", str(path)])
    SCRIPT_MODULE.main()

    out = capsys.readouterr().out
    cells = SCRIPT_MODULE._cells(ARM_NUMBERS, (2,))
    assert len(SCRIPT_MODULE._load_done(str(path))) == len(cells)
    assert len(built) == len(cells) + 1        # one fresh model per cell, plus fp32
    assert calibrated == [1]                   # arm 6 re-estimates, once
    assert out.count(f"| {HEADLINE_MODEL} |") == len(cells)
    assert "input-moment cases over" in out and "A " in out
    assert out.count("[") >= 2 * CRITERIA      # criteria before and after the table


def test_a_second_run_reuses_the_recorded_cells_instead_of_recomputing(
        cpu_run, monkeypatch, tmp_path, capsys):
    built, _ = cpu_run
    path = tmp_path / "cells.jsonl"
    argv = ["qbn", str(tmp_path), "--models", HEADLINE_MODEL, "--passes", "2",
            "--out", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    SCRIPT_MODULE.main()
    first = path.read_text()
    built.clear()
    capsys.readouterr()

    SCRIPT_MODULE.main()
    out = capsys.readouterr().out
    assert path.read_text() == first            # nothing recomputed, nothing appended
    assert len(built) == 1                      # only the model the case split needs
    assert out.count(f"| {HEADLINE_MODEL} |") == len(SCRIPT_MODULE._cells(ARM_NUMBERS, (2,)))
