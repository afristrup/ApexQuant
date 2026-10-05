"""ImageNet-1K: the seven-arm accuracy experiment for QBN, design document 7.3.

QBN re-derives a BatchNorm's two stored statistics so that they describe the
quantized network rather than the fp32 one, from the weights and the stored
BatchNorm parameters alone. This script is the accuracy test of that claim, and
it is deliberately a *seven*-arm table rather than a three-arm one, because the
interesting comparisons are between arms rather than against fp32:

    1 fp32                                          n/a
    2 quantized                                     data-free
    3 + QBN mean                                    data-free
    4 + QBN mean + variance                         data-free
    5 + QBN variance only, a control                data-free
    6 + BN re-estimated on 2048 unlabelled images   NOT data-free, the ceiling
    7 + BN folded before quantizing                 data-free, predicted no-op

Arm 6 is the ceiling: whatever re-estimation recovers is an upper bound on what
any intervention that only edits BatchNorm's two buffers can recover, QBN
included, and it is the only arm that reads images. It belongs in its own
column and must never be folded into the data-free claim. Arm 5 is a control,
not a product: if the variance term alone helps materially then the mechanism is
the coherent contraction rather than the mean shift, and the paper should be
rewritten around ``alpha_c``. Arm 7 is a provable no-op, and is here as a check
on the harness rather than as a method.

ConvNeXt-Tiny is the negative control. It has no BatchNorm, so arms 3, 4 and 5
must come out bit-identical to arm 2; the ``= arm 2`` column reports that
comparison, and it is the cheapest bug detector available.

The success criteria were fixed before the run and are printed twice, once
before the first number is computed and once with verdicts at the end.

Every cell of the table is independent: one fresh model, quantized for its own
arm, so ``--models``, ``--passes`` and ``--arms`` select any subset and
``--out`` makes a killed run resumable by skipping the cells already on disk.

    uv run python experiments/qbn_accuracy_imagenet.py /path/to/ImageNet1K-val \
        --out results/qbn_accuracy.jsonl

"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bn_stale_statistics_imagenet import (BN_BATCH, BN_BATCHES, fold_batchnorm,
                                          reestimate_batchnorm)
from progressive_vision_imagenet import DEV, N_IMAGES, load_val, logits_of

from apexquant.bn.arm_model import arm_model
from apexquant.bn.arm_result import ArmResult
from apexquant.bn.case_fractions import case_fractions
from apexquant.bn.conv_bn_sites import conv_bn_sites
from apexquant.bn.qbn_arms import (ARM_DATA_FREE, ARM_FP32, ARM_NAMES,
                                          ARM_NUMBERS, ARM_QUANTIZED,
                                          ARM_REESTIMATED, FP32_PASSES)
from apexquant.bn.qbn_criteria import qbn_criteria

MODELS = ("resnet50", "efficientnet_b0", "convnext_tiny")
PASSES = (2, 3)
PERCENT = 100.0

#: Arm 6's budget, the same 2048 images the three-arm script re-estimates on.
CALIB_IMAGES = BN_BATCHES * BN_BATCH

#: Validation shards, so that a calibration slice can be taken past the
#: evaluation window. ``load_val`` draws one permutation per shard from a seeded
#: generator, so ``[0, take)`` and ``[skip, skip + take)`` are disjoint by
#: construction. This mirrors ``bn_stale_statistics_imagenet.py`` exactly.
VAL_SHARDS = 14
HELD_OUT_SKIP = -(-N_IMAGES // VAL_SHARDS)


def main() -> None:
    args = _parse_args()
    import timm

    done = _load_done(args.out)
    results = list(done.values())
    source = "the training shards" if args.calib_dir else "a held-out validation slice"
    print("# QBN accuracy: the seven arms of section 7.3")
    print(f"# {N_IMAGES} seeded validation images; KL is against the same model at fp32")
    print(f"# arm 6 re-estimates on {CALIB_IMAGES} unlabelled images from {source}")
    print("# success criteria, fixed in advance, repeated with verdicts at the end:")
    for line in qbn_criteria([]):
        print(f"#   {line}")
    print()

    for name in args.models:
        torch.manual_seed(0)
        cells = _cells(args.arms, args.passes)
        ref = timm.create_model(name, pretrained=True).eval()
        sites = conv_bn_sites(ref)
        fractions = case_fractions(sites)
        spread = ", ".join(f"{letter} {pct:.1f}%" for letter, pct in fractions.items())
        print(f"# {name} input-moment cases over {len(sites)} convolutions: {spread}. "
              "Case D is pooling and squeeze-excitation gates, left uncorrected by "
              "construction, and section 8.7 asks for that number next to the accuracy.")
        print("| model | tier | arm | data-free | top-1 | KL vs fp32 | = arm 2 |")
        print("|-------|------|-----|-----------|------:|-----------:|---------|")

        x = y = ref_logp = x_cal = transform = None
        quantized_logits: dict[int, torch.Tensor] = {}
        for passes, arm in cells:
            if (name, passes, arm) in done:
                _print_row(done[(name, passes, arm)])
                continue
            if ref_logp is None:
                transform = timm.data.create_transform(
                    **timm.data.resolve_model_data_config(ref), is_training=False)
                x, y = load_val(args.imagenet_val_dir, N_IMAGES, transform)
                ref_logp = torch.log_softmax(logits_of(ref.to(DEV), x), dim=-1)
                _release(ref)
            if arm == ARM_REESTIMATED and x_cal is None:
                x_cal, _ = load_val(args.calib_dir or args.imagenet_val_dir,
                                    CALIB_IMAGES, transform,
                                    skip=0 if args.calib_dir else HELD_OUT_SKIP)

            model = timm.create_model(name, pretrained=True).eval()
            arm_model(model, arm, passes, fold=fold_batchnorm,
                      reestimate=lambda m: reestimate_batchnorm(m, x_cal))
            logits = logits_of(model.to(DEV), x)
            _release(model)
            if arm == ARM_QUANTIZED:
                quantized_logits[passes] = logits
            reference = quantized_logits.get(passes)
            result = ArmResult(
                model=name, passes=passes, arm=arm,
                top1=PERCENT * (logits.argmax(1) == y).float().mean().item(),
                kl=F.kl_div(torch.log_softmax(logits, dim=-1), ref_logp,
                            log_target=True, reduction="batchmean").item(),
                identical_to_quantized=(None if reference is None or arm == ARM_QUANTIZED
                                        else bool(torch.equal(logits, reference))))
            results.append(result)
            _print_row(result)
            _append(args.out, result)
        print()

    print("# Success criteria of sections 7.3 and 7.4, judged:")
    for line in qbn_criteria(results):
        print(f"#   {line}")


def _parse_args() -> argparse.Namespace:
    """Parse and validate the selection flags, which are what makes a run resumable."""
    ap = argparse.ArgumentParser(
        description="Seven-arm QBN accuracy on ImageNet-1K (design document 7.3).")
    ap.add_argument("imagenet_val_dir", help="directory of validation parquet shards")
    ap.add_argument("--arms", default=",".join(str(a) for a in ARM_NUMBERS),
                    help="comma-separated arm numbers to run, 1 to 7")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma-separated timm model names to run")
    ap.add_argument("--passes", default=",".join(str(p) for p in PASSES),
                    help="comma-separated residual pass counts P")
    ap.add_argument("--calib-dir", default=None,
                    help="training-split shards for arm 6; without it arm 6 uses a "
                         "held-out slice of the validation shards, which is a ceiling "
                         "rather than a method number")
    ap.add_argument("--out", default=None,
                    help="JSONL of finished cells, appended to; cells already in it "
                         "are reprinted rather than recomputed, so a killed run resumes")
    args = ap.parse_args()
    args.arms = _selected(args.arms, ARM_NUMBERS, int, "arm")
    args.models = _selected(args.models, MODELS, str, "model")
    args.passes = _selected(args.passes, PASSES, int, "P")
    return args


def _selected(text: str, allowed: tuple, cast, label: str) -> tuple:
    """Return the parsed selection, in the canonical order of ``allowed``."""
    chosen = {cast(part) for part in text.split(",") if part}
    unknown = chosen - set(allowed)
    if not chosen or unknown:
        raise ValueError(f"{label} selection {text!r} must be a non-empty subset of "
                         f"{allowed}, got {sorted(unknown)} outside it")
    return tuple(item for item in allowed if item in chosen)


def _cells(arms: tuple, passes: tuple) -> list[tuple[int, int]]:
    """Return the ``(passes, arm)`` cells to run, fp32 first and then by ``P``."""
    cells = [(FP32_PASSES, ARM_FP32)] if ARM_FP32 in arms else []
    return cells + [(p, arm) for p in passes for arm in arms if arm != ARM_FP32]


def _print_row(result: ArmResult) -> None:
    """Print one finished cell as a row of the markdown table."""
    tier = "fp32" if result.passes == FP32_PASSES else f"P={result.passes}"
    same = {None: "", True: "yes", False: "no"}[result.identical_to_quantized]
    print(f"| {result.model} | {tier} | {result.arm} {ARM_NAMES[result.arm]} "
          f"| {ARM_DATA_FREE[result.arm]} | {result.top1:>5.2f}% | {result.kl:>10.4f} "
          f"| {same} |", flush=True)


def _load_done(path: str | None) -> dict[tuple[str, int, int], ArmResult]:
    """Read the cells a previous run finished, keyed by ``(model, passes, arm)``."""
    if not path or not os.path.exists(path):
        return {}
    done = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                result = ArmResult(**json.loads(line))
                done[(result.model, result.passes, result.arm)] = result
    return done


def _append(path: str | None, result: ArmResult) -> None:
    """Append one finished cell, immediately, so a kill loses at most one cell."""
    if not path:
        return
    with open(path, "a") as handle:
        handle.write(json.dumps(dataclasses.asdict(result)) + "\n")


def _release(model: torch.nn.Module) -> None:
    """Give the accelerator its memory back between cells."""
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
