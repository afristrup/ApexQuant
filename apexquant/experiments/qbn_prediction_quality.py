"""QBN test 7.2: does the analytic correction predict the measured shift?

The cheap decisive test of ``docs/apexquant-batchnorm.md``, run before any accuracy
number because it tests the *model* rather than the outcome. It pushes unlabelled
images through the fp32 network and through the quantized network, collects the
per-channel mean and variance of every BatchNorm's input (which is the layer output
``Y_c`` before normalization, and is what ``running_mean`` and ``running_var`` are
estimates of), and compares those measurements against what section 4.5 predicts
with no data at all.

The predictions are stated in the document in advance and are the point of the
experiment, so they are printed next to the measurements and each one gets a
verdict:

* mean: ``r > 0.95`` and relative error under 5%. ``r < 0.8`` refutes the
  mechanism, whatever accuracy does afterwards.
* variance: ``r`` between 0.3 and 0.5, relative error near 0.9. Above 0.8 would be
  a pleasant surprise and would mean the dropped cross term of section 8.4 is
  smaller than expected.
* anchor: median ``|v_stored - v_fp32_measured| / v`` below 1%. Above that, the
  variance correction is inside the noise of the quantity it corrects and section
  8.5's stated response is to delete the variance term from the paper.

**Two metrics, and the one the thresholds belong to.** The document writes the
comparison as "predicted ``mu + d_mean`` against measured ``E[Y'_c]``". Taken on
the levels that comparison is uninformative: ``mu`` dominates ``d_mean`` by about
20x, so the level correlation is ~1.000 for any anchor at all and could never fall
below the 0.8 that is supposed to refute the mechanism. What the correction
actually predicts is the *shift*, so the verdicts are read off the shift columns,
``d_mean`` against ``E[Y'_c] - E[Y_c]`` and ``d_var`` against
``Var[Y'_c] - Var[Y_c]``, both measured. The level columns are printed too, so
nothing the document asks for is missing. A relative error of 1.0 on a shift column
means the prediction is worth exactly as much as predicting no shift at all, which
is the reading section 8.4 uses when it calls a relative error of 1.7 to 4.1 "worse
than predicting zero".

The measured shift is taken against the *measured* fp32 moments rather than against
the stored buffers, so that the EMA anchor error does not leak into it. That anchor
error is reported separately, which is exactly what section 7.2 asks for.

Network (b) is the plain quantized network with its stale statistics, arm 2 of
section 7.3, not the QBN-corrected one: the correction is local by section 4.6, so
what it claims to predict is the shift the quantized weights produce, and measuring
on a network whose downstream BatchNorms have already been edited would fold QBN's
own effect into the thing QBN is being scored against.

**The image window is disjoint from the evaluation window.** The 2026-09-12
correction in ``.scratch/apexquant-iclr2027/issues/28-batchnorm-corrector.md``
withdrew a whole set of conclusions that came from calibrating on the first rows of
the evaluation tensor: the validation shards are class ordered, so that slice was
both transductive and a roughly 200-class sample, and it flipped the sign of a
result. ``load_val``'s ``skip`` slices the same per-shard permutation past the
evaluation draw's own take, so the two windows are disjoint by construction.

    uv run python experiments/qbn_prediction_quality.py /path/to/ImageNet1K-val

"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch
from torch import nn

# Both this directory and the library, derived from __file__ rather than from the
# working directory: the pod runs these scripts by path, from wherever it happens
# to be, and `pip install -e .` is not what puts the package there.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

from apexquant.bn.apply_qbn import apply_qbn  # noqa: E402
from apexquant.bn.case_letters import CORRECTABLE_CASES  # noqa: E402
from apexquant.bn.qbn_modes import MODE_MEAN_VAR  # noqa: E402
from apexquant.bn.quantize_with_residuals import (  # noqa: E402
    quantize_with_residuals)
from progressive_vision_imagenet import DEV, N_IMAGES, load_val  # noqa: E402

MODELS = ("resnet50", "efficientnet_b0")
PASSES = (2,)
N_MEASURE = 2048          # section 7.2's image budget
BATCH = 32

# `load_val`'s `skip` is a per-shard offset into the per-shard permutation, so a
# window starting past the evaluation draw's own per-shard take is disjoint from
# it. The evaluation draw is N_IMAGES spread over the 14 shards the ImageNet-1K
# validation parquet set ships in, which is what the sibling scripts assume; if
# the set is ever resharded this offset has to move with it.
VAL_SHARDS = 14
MEASURE_SKIP = -(-N_IMAGES // VAL_SHARDS)

# BatchNorm stores the *unbiased* minibatch variance in `running_var`, so the
# measurement it is scored against uses the same denominator.
UNBIASED_DOF = 1
VAR_EPS = 1e-12           # only ever reached by a dead channel with v == 0
MIN_PEARSON_N = 2

# Section 7.2's predictions, stated in advance.
MEAN_R_FLOOR = 0.95
MEAN_R_REFUTES = 0.80
MEAN_REL_MAX = 0.05
VAR_R_LOW = 0.30
VAR_R_HIGH = 0.50
VAR_R_SURPRISE = 0.80
VAR_REL_EXPECTED = 0.90
# "near 0.9" is not given a tolerance in the document; this is the band the
# verdict column calls "near", and it is printed so the reader can re-judge it.
VAR_REL_NEAR = 0.20
ANCHOR_VAR_MAX = 0.01


@torch.no_grad()
def measure_pre_bn_moments(model: nn.Module, x: torch.Tensor, device: str,
                           batch: int = BATCH) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Collect each BatchNorm's per-channel input mean and variance over ``x``.

    A forward *pre* hook sees the tensor the BatchNorm is about to normalize,
    which is the convolution output ``Y`` of section 3.1 and the quantity the
    stored statistics estimate. Reductions run per batch in float32 and
    accumulate in float64, because the second moment of a few hundred million
    activations is where a float32 accumulator quietly loses the digits this test
    is trying to resolve.

    Args:
        model: The network to measure, in eval mode. It is moved to ``device``.
        x: Images, ``(N, 3, H, W)``, on any device.
        device: Where to run the forward passes.
        batch: Images per forward pass.

    Returns:
        Dict from BatchNorm module name to ``(mean, var)``, both float64 on the
        CPU, shape ``(num_features,)``.

    Raises:
        ValueError: A BatchNorm saw too few samples for an unbiased variance,
            which means ``x`` was empty or the module was never reached.
    """
    sums: dict[str, torch.Tensor] = {}
    squares: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def make_hook(name: str, channels: int):
        sums[name] = torch.zeros(channels, dtype=torch.float64)
        squares[name] = torch.zeros(channels, dtype=torch.float64)
        counts[name] = 0

        def hook(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            y = args[0].detach().float()
            dims = [0] + list(range(2, y.ndim))
            sums[name] += y.sum(dim=dims).double().cpu()
            squares[name] += y.square().sum(dim=dims).double().cpu()
            counts[name] += y.numel() // y.shape[1]

        return hook

    model = model.to(device).eval()
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            handles.append(
                module.register_forward_pre_hook(make_hook(name, module.num_features)))
    for start in range(0, x.shape[0], batch):
        model(x[start:start + batch].to(device))
    for handle in handles:
        handle.remove()

    moments = {}
    for name, total in sums.items():
        n = counts[name]
        if n <= UNBIASED_DOF:
            raise ValueError(f"{name} saw {n} samples, too few for a variance")
        mean = total / n
        moments[name] = (mean, (squares[name] - n * mean * mean) / (n - UNBIASED_DOF))
    return moments


def pearson_r(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the Pearson correlation of two per-channel vectors.

    Args:
        a: Predicted values, 1-D.
        b: Measured values, same shape.

    Returns:
        The correlation, or NaN when either vector is constant or too short for
        one to be defined.

    Raises:
        ValueError: The shapes disagree.
    """
    if a.shape != b.shape:
        raise ValueError(f"shapes disagree: {tuple(a.shape)} against {tuple(b.shape)}")
    if a.numel() < MIN_PEARSON_N:
        return float("nan")
    da, db = a.double() - a.double().mean(), b.double() - b.double().mean()
    denominator = da.norm() * db.norm()
    if denominator == 0:
        return float("nan")
    return float(torch.dot(da, db) / denominator)


def relative_error(predicted: torch.Tensor, measured: torch.Tensor) -> float:
    """Return ``||predicted - measured|| / ||measured||`` over the channels.

    Normalized this way, 1.0 is the score of predicting zero, so a relative error
    above 1 means the prediction is worth less than no prediction at all. That is
    the reading section 8.4 uses.

    Args:
        predicted: The analytic prediction, 1-D.
        measured: What the forward passes measured, same shape.

    Returns:
        The relative error, or NaN when the measured vector is identically zero.

    Raises:
        ValueError: The shapes disagree.
    """
    if predicted.shape != measured.shape:
        raise ValueError(f"shapes disagree: {tuple(predicted.shape)} against "
                         f"{tuple(measured.shape)}")
    scale = measured.double().norm()
    if scale == 0:
        return float("nan")
    return float((predicted.double() - measured.double()).norm() / scale)


def site_vectors(correction, stored: tuple[torch.Tensor, torch.Tensor],
                 fp32: tuple[torch.Tensor, torch.Tensor],
                 quant: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Assemble one layer's predicted, measured and anchor vectors.

    Args:
        correction: The :class:`QbnCorrection` for this layer. Its *raw* ``d_mean``
            and ``d_var`` are used, which is what section 7.2 writes; the guard
            counts are reported separately so a reader can see whether the guarded
            buffers differ at all.
        stored: ``(running_mean, running_var)`` as the checkpoint shipped them,
            read before :func:`apply_qbn` edited anything.
        fp32: Measured ``(E[Y], Var[Y])`` of the fp32 network.
        quant: Measured ``(E[Y'], Var[Y'])`` of the quantized network.

    Returns:
        Dict of float64 per-channel vectors: the levels, the shifts, and the two
        EMA anchor errors.
    """
    mu, v = stored
    mean_fp32, var_fp32 = fp32
    mean_quant, var_quant = quant
    d_mean = correction.d_mean.detach().double().cpu()
    d_var = correction.d_var.detach().double().cpu()
    scale = v.clamp_min(VAR_EPS)
    return {
        "mean_pred": mu + d_mean,
        "mean_meas": mean_quant,
        "mean_shift_pred": d_mean,
        "mean_shift_meas": mean_quant - mean_fp32,
        "var_pred": v + d_var,
        "var_meas": var_quant,
        "var_shift_pred": d_var,
        "var_shift_meas": var_quant - var_fp32,
        "anchor_mean": (mu - mean_fp32).abs() / scale.sqrt(),
        "anchor_var": (v - var_fp32).abs() / scale,
    }


def qbn_prediction_rows(model: nn.Module, x: torch.Tensor, passes: int,
                        device: str = DEV, batch: int = BATCH):
    """Measure both networks and pair every corrected layer with its prediction.

    The order is the forced one of section 4.2, with the measurement wedged into
    it: measure fp32, quantize, measure quantized, and only then apply QBN. QBN
    runs last because it edits BatchNorm buffers, and measuring after those edits
    would score the correction against a network the correction had already
    changed.

    Args:
        model: The fp32 model, in eval mode. It is moved to ``device`` and back
            but its weights and buffers are left alone; the quantized network is
            a deep copy.
        x: Images, ``(N, 3, H, W)``.
        passes: Residual passes ``P``.
        device: Where to run the forward passes.
        batch: Images per forward pass.

    Returns:
        ``(rows, reports)``. Each row is a dict with ``conv``, ``bn``, ``case``
        and ``vectors``, one per corrected layer whose BatchNorm was measured;
        ``reports`` is the full :func:`apply_qbn` result, skipped layers included.
    """
    target = copy.deepcopy(model).eval()
    fp32 = measure_pre_bn_moments(model, x, device, batch)
    model.to("cpu")
    stored = {name: (module.running_mean.detach().double().cpu(),
                     module.running_var.detach().double().cpu())
              for name, module in target.named_modules()
              if isinstance(module, nn.modules.batchnorm._BatchNorm)
              and module.running_var is not None}

    residuals = quantize_with_residuals(target, passes)
    quant = measure_pre_bn_moments(target, x, device, batch)
    reports = apply_qbn(target, residuals, MODE_MEAN_VAR)

    rows = []
    for report in reports.values():
        if not report.corrected or report.bn not in stored:
            continue
        rows.append({
            "conv": report.conv,
            "bn": report.bn,
            "case": report.case,
            "vectors": site_vectors(report.correction, stored[report.bn],
                                    fp32[report.bn], quant[report.bn]),
        })
    return rows, reports


def pool(rows, case: str | None = None) -> dict[str, torch.Tensor]:
    """Concatenate the per-channel vectors of every row, or of one case.

    Args:
        rows: Rows from :func:`qbn_prediction_rows`.
        case: Input-moment case letter to keep, or None for all of them.

    Returns:
        The same keys as one row's vectors, concatenated over channels. Empty
        when no row matches.
    """
    chosen = [row for row in rows if case is None or row["case"] == case]
    if not chosen:
        return {}
    return {key: torch.cat([row["vectors"][key] for row in chosen])
            for key in chosen[0]["vectors"]}


def statistics(vectors: dict[str, torch.Tensor]) -> dict[str, float]:
    """Return test 7.2's numbers for one pool of channels.

    Args:
        vectors: A row's vectors, or a :func:`pool` of several rows'.

    Returns:
        Correlations and relative errors on the shifts (what the correction
        predicts) and on the levels (what the document writes), the two median
        anchor errors, and the channel count.
    """
    return {
        "channels": float(vectors["mean_pred"].numel()),
        "mean_r": pearson_r(vectors["mean_shift_pred"], vectors["mean_shift_meas"]),
        "mean_rel": relative_error(vectors["mean_shift_pred"], vectors["mean_shift_meas"]),
        "var_r": pearson_r(vectors["var_shift_pred"], vectors["var_shift_meas"]),
        "var_rel": relative_error(vectors["var_shift_pred"], vectors["var_shift_meas"]),
        "mean_r_level": pearson_r(vectors["mean_pred"], vectors["mean_meas"]),
        "mean_rel_level": relative_error(vectors["mean_pred"], vectors["mean_meas"]),
        "var_r_level": pearson_r(vectors["var_pred"], vectors["var_meas"]),
        "var_rel_level": relative_error(vectors["var_pred"], vectors["var_meas"]),
        "anchor_mean": float(vectors["anchor_mean"].median()),
        "anchor_var": float(vectors["anchor_var"].median()),
    }


def verdicts(stat: dict[str, float]) -> list[tuple[str, str, str, str]]:
    """Score each advance prediction of section 7.2 against one pool.

    Args:
        stat: The output of :func:`statistics` for the pooled channels.

    Returns:
        Rows of ``(quantity, prediction, measurement, verdict)``, ready to print.
    """
    mean_r, var_r = stat["mean_r"], stat["var_r"]
    mean_rel, var_rel, anchor_var = stat["mean_rel"], stat["var_rel"], stat["anchor_var"]
    if mean_r < MEAN_R_REFUTES:
        mean_r_verdict = "MECHANISM REFUTED (section 7.4)"
    elif mean_r > MEAN_R_FLOOR:
        mean_r_verdict = "held"
    else:
        mean_r_verdict = "below prediction, not refuting"
    if var_r > VAR_R_SURPRISE:
        var_r_verdict = "SURPRISE: cross term smaller than expected"
    elif VAR_R_LOW <= var_r <= VAR_R_HIGH:
        var_r_verdict = "held"
    else:
        var_r_verdict = "outside the predicted band"
    return [
        ("mean shift r", f"> {MEAN_R_FLOOR:.2f}, and < {MEAN_R_REFUTES:.2f} refutes",
         f"{mean_r:.3f}", mean_r_verdict),
        ("mean shift relative error", f"< {MEAN_REL_MAX:.2f}",
         f"{mean_rel:.3f}", "held" if mean_rel < MEAN_REL_MAX else "above prediction"),
        ("variance shift r", f"in [{VAR_R_LOW:.2f}, {VAR_R_HIGH:.2f}]",
         f"{var_r:.3f}", var_r_verdict),
        ("variance shift relative error",
         f"near {VAR_REL_EXPECTED:.2f} (+/- {VAR_REL_NEAR:.2f})", f"{var_rel:.3f}",
         "held" if abs(var_rel - VAR_REL_EXPECTED) <= VAR_REL_NEAR else "off prediction"),
        # The bars are escaped because this row is printed inside a table.
        (r"anchor \|v_stored - v_fp32\| / v, median", f"< {ANCHOR_VAR_MAX:.3f}",
         f"{anchor_var:.4f}",
         "held" if anchor_var < ANCHOR_VAR_MAX
         else "DELETE THE VARIANCE TERM (section 8.5)"),
    ]


def report(name: str, passes: int, rows, reports) -> None:
    """Print one model-and-tier block: coverage, per layer, pooled, verdicts.

    Args:
        name: Model name.
        passes: Residual passes ``P``.
        rows: Rows from :func:`qbn_prediction_rows`.
        reports: The full :func:`apply_qbn` result, for the coverage lines.
    """
    corrected = [r for r in reports.values() if r.corrected]
    skipped: dict[str, int] = {}
    for entry in reports.values():
        if not entry.corrected:
            skipped[entry.skipped] = skipped.get(entry.skipped, 0) + 1
    mean_hits = sum(r.correction.mean_shift_hits for r in corrected)
    floor_hits = sum(r.correction.var_floor_hits for r in corrected)

    print(f"\n## {name}, P={passes}\n")
    print(f"# {len(corrected)} of {len(reports)} convolutions corrected"
          + (f"; left alone: {skipped}" if skipped else ""))
    print(f"# guards: {mean_hits} channels hit the mean-shift clamp, "
          f"{floor_hits} hit the variance floor (section 4.7 expects none at P=2)")
    if not rows:
        print("# nothing to score: no corrected layer's BatchNorm was measured")
        return

    print("\n| layer | case | channels | mean r | mean rel | var r | var rel "
          "| anchor mu | anchor v |")
    print("|-------|------|---------:|-------:|---------:|------:|--------:"
          "|----------:|---------:|")
    for row in rows:
        s = statistics(row["vectors"])
        print(f"| {row['conv']} | {row['case']} | {int(s['channels'])} "
              f"| {s['mean_r']:>6.3f} | {s['mean_rel']:>8.3f} | {s['var_r']:>6.3f} "
              f"| {s['var_rel']:>7.3f} | {s['anchor_mean']:>9.4f} "
              f"| {s['anchor_var']:>8.4f} |", flush=True)

    print("\n| pool | channels | mean r | mean rel | var r | var rel | mean r (level) "
          "| mean rel (level) | var r (level) | var rel (level) | anchor mu | anchor v |")
    print("|------|---------:|-------:|---------:|------:|--------:|---------------:"
          "|-----------------:|--------------:|----------------:|----------:|---------:|")
    for label, case in [("all", None)] + [(f"case {c}", c) for c in CORRECTABLE_CASES]:
        vectors = pool(rows, case)
        if not vectors:
            continue
        s = statistics(vectors)
        print(f"| {label} | {int(s['channels'])} | {s['mean_r']:>6.3f} "
              f"| {s['mean_rel']:>8.3f} | {s['var_r']:>6.3f} | {s['var_rel']:>7.3f} "
              f"| {s['mean_r_level']:>14.4f} | {s['mean_rel_level']:>16.4f} "
              f"| {s['var_r_level']:>13.4f} | {s['var_rel_level']:>15.4f} "
              f"| {s['anchor_mean']:>9.4f} | {s['anchor_var']:>8.4f} |")

    print("\n| quantity | predicted in advance | measured (pooled) | verdict |")
    print("|----------|----------------------|------------------:|---------|")
    for quantity, prediction, measured, verdict in verdicts(statistics(pool(rows))):
        print(f"| {quantity} | {prediction} | {measured} | {verdict} |")


def main() -> None:
    """Run test 7.2 over the models and tiers named on the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("imagenet_val_dir")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--passes", nargs="+", type=int, default=list(PASSES))
    ap.add_argument("--images", type=int, default=N_MEASURE)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--skip", type=int, default=MEASURE_SKIP,
                    help="per-shard offset into load_val's permutation; the "
                         "default starts past the evaluation window")
    ap.add_argument("--device", default=DEV)
    args = ap.parse_args()
    import timm

    print("# QBN test 7.2: per-channel prediction quality and EMA anchor error")
    print(f"# {args.images} unlabelled images, held out at skip={args.skip}, "
          f"disjoint from the {N_IMAGES}-image evaluation window")
    print("# r and relative error are on the SHIFT (what the correction predicts);")
    print("# the level columns are the same comparison on mu + d_mean itself.")
    print("# A relative error of 1.0 is the score of predicting no shift at all.")

    for name in args.models:
        torch.manual_seed(0)
        model = timm.create_model(name, pretrained=True).eval()
        transform = timm.data.create_transform(
            **timm.data.resolve_model_data_config(model), is_training=False)
        x, _ = load_val(args.imagenet_val_dir, args.images, transform, skip=args.skip)
        for passes in args.passes:
            rows, reports = qbn_prediction_rows(model, x, passes, args.device,
                                                args.batch)
            report(name, passes, rows, reports)


if __name__ == "__main__":
    main()
