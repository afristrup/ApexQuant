"""Tests for experiments/qbn_prediction_quality.py, QBN design test 7.2.

The design document warns twice that published versions of these formulas carry
typos and that Monte Carlo is what caught them, so the two formulas this script
scores predictions against -- section 4.5's mean line and variance line -- are
checked here against measured forward passes rather than against themselves.

The measurement is made exact rather than merely statistical. A random input
would leave the isotropy corollary ``E[dw^T C dw] = ||dw||^2 tr(C)/d`` true only
in expectation, with a sampling error larger than the effect at any batch size
this test can afford, so the input is *whitened*: its empirical covariance over
the receptive field is exactly ``tau I``. Section 4.5's variance line is then an
identity, not an approximation, and it can be asserted to five digits. What the
whitening does not excuse is the dropped cross term; it makes it exactly zero,
which is the regime the formula claims to be exact in.

Everything here runs on the CPU. ``gemlite`` is stubbed out before the library is
imported because its import queries the CUDA device and would take a context on a
box whose GPU belongs to another job.
"""

from __future__ import annotations

import io
import math
import os
import sys

import pytest
import torch
from torch import nn

sys.modules.setdefault("gemlite", None)          # keep the import chain off the GPU

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from apexquant.bn.qbn_correction import qbn_correction  # noqa: E402
from progressive_vision_imagenet import N_IMAGES, load_val  # noqa: E402
from qbn_prediction_quality import (MEAN_R_FLOOR, MEAN_R_REFUTES,  # noqa: E402
                                    MEASURE_SKIP, UNBIASED_DOF, VAL_SHARDS,
                                    measure_pre_bn_moments, pearson_r, pool,
                                    qbn_prediction_rows, relative_error,
                                    site_vectors, statistics, verdicts)

CHANNELS = 32
KERNEL = 3
OUT_CHANNELS = 24
SAMPLES = 4096
GROUP = 32
TAU = 0.75
INPUT_MEAN = 0.4
PERTURBATION = 0.3
REDUCTION_RTOL = 1e-5


class ConvBn(nn.Module):
    """A convolution and the BatchNorm that reads it, and nothing else."""

    def __init__(self, kernel: int = KERNEL) -> None:
        super().__init__()
        self.conv = nn.Conv2d(CHANNELS, OUT_CHANNELS, kernel, bias=True)
        self.bn = nn.BatchNorm2d(OUT_CHANNELS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


def whitened_input(samples: int = SAMPLES, seed: int = 0) -> torch.Tensor:
    """Return ``(samples, CHANNELS, KERNEL, KERNEL)`` with an exact covariance.

    The flattened receptive field has empirical mean ``INPUT_MEAN`` in every
    coordinate and empirical covariance exactly ``TAU * I`` under the same
    unbiased denominator the measurement uses, so every expectation section 4.5
    takes is realized exactly by these samples.
    """
    torch.manual_seed(seed)
    d = CHANNELS * KERNEL * KERNEL
    raw = torch.randn(samples, d, dtype=torch.float64)
    raw = raw - raw.mean(dim=0, keepdim=True)
    covariance = raw.T @ raw / (samples - UNBIASED_DOF)
    root = torch.linalg.cholesky(covariance)
    white = torch.linalg.solve_triangular(root, raw.T, upper=False).T
    white = white * math.sqrt(TAU) + INPUT_MEAN
    return white.reshape(samples, CHANNELS, KERNEL, KERNEL).float()


def measured(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the BatchNorm input's measured per-channel mean and variance."""
    return measure_pre_bn_moments(model, x, "cpu", batch=512)["bn"]


def test_measure_pre_bn_moments_matches_a_direct_reduction():
    torch.manual_seed(1)
    model = ConvBn().eval()
    x = whitened_input(samples=512)
    mean, var = measured(model, x)
    with torch.no_grad():
        y = model.conv(x).double().reshape(x.shape[0], OUT_CHANNELS)
    # The hook reduces each batch in float32 and accumulates in float64, so it
    # agrees with a float64 reduction of the whole tensor to about 1e-7 relative,
    # not to the last bit.
    assert torch.allclose(mean, y.mean(dim=0), rtol=REDUCTION_RTOL, atol=1e-9)
    assert torch.allclose(var, y.var(dim=0, unbiased=True), rtol=REDUCTION_RTOL)


def test_measure_pre_bn_moments_uses_the_denominator_batchnorm_stores():
    torch.manual_seed(2)
    model = ConvBn().eval()
    x = whitened_input(samples=512)
    _, var = measured(model, x)
    with torch.no_grad():
        y = model.conv(x).double().reshape(x.shape[0], OUT_CHANNELS)
    biased = y.var(dim=0, unbiased=False)
    # Not the biased one: BatchNorm's running_var is the unbiased estimate, and at
    # 512 samples the two differ by more than this assertion's tolerance.
    assert not torch.allclose(var, biased, rtol=1e-4)
    assert torch.allclose(var, biased * 512 / (512 - UNBIASED_DOF), rtol=REDUCTION_RTOL)


def test_the_whitened_input_realizes_its_moments_exactly():
    x = whitened_input()
    flat = x.double().reshape(SAMPLES, -1)
    covariance = (flat - flat.mean(0)).T @ (flat - flat.mean(0)) / (SAMPLES - UNBIASED_DOF)
    assert torch.allclose(flat.mean(0), torch.full((flat.shape[1],), INPUT_MEAN,
                                                   dtype=torch.float64), atol=1e-5)
    assert torch.allclose(covariance, TAU * torch.eye(flat.shape[1], dtype=torch.float64),
                          atol=1e-5)


def _correction_against_measurement(seed: int = 3):
    """Quantize nothing, perturb the weight by hand, and measure both networks."""
    torch.manual_seed(seed)
    model = ConvBn().eval()
    x = whitened_input()
    mean_fp32, var_fp32 = measured(model, x)

    weight = model.conv.weight.detach().clone()
    dw = torch.randn_like(weight)
    dw *= PERTURBATION * weight.norm() / dw.norm()
    with torch.no_grad():
        model.conv.weight.copy_(weight + dw)
    mean_quant, var_quant = measured(model, x)

    e = torch.full((CHANNELS,), INPUT_MEAN)
    s = torch.full((CHANNELS,), TAU + INPUT_MEAN ** 2)
    correction = qbn_correction(w=weight.reshape(OUT_CHANNELS, -1),
                                dw=dw.reshape(OUT_CHANNELS, -1), e=e, s=s,
                                mu=mean_fp32.float(), v=var_fp32.float(),
                                group_size=GROUP)
    return correction, (mean_fp32, var_fp32), (mean_quant, var_quant)


def test_the_mean_line_matches_the_measured_shift():
    correction, fp32, quant = _correction_against_measurement()
    predicted = correction.d_mean.double()
    assert torch.allclose(predicted, quant[0] - fp32[0], rtol=2e-4, atol=1e-6)


def test_the_variance_line_matches_the_measured_shift():
    correction, fp32, quant = _correction_against_measurement()
    predicted = correction.d_var.double()
    shift = quant[1] - fp32[1]
    # The correction is a real fraction of the variance here, not a rounding of
    # it, so agreeing to this tolerance is a statement about the formula.
    assert (shift.abs() / fp32[1]).median() > 0.01
    assert torch.allclose(predicted, shift, rtol=5e-3, atol=1e-6)


def test_the_variance_line_needs_the_centred_second_moment():
    # Section 4.5: tau is tr(Cov x)/d, not tr(E[x x^T])/d. Passing the raw second
    # moment where the variance belongs inflates tau by INPUT_MEAN^2, which is
    # the error that moved ResNet-50's median tau from 0.610 to 0.955.
    correction, fp32, quant = _correction_against_measurement()
    torch.manual_seed(3)
    model = ConvBn().eval()
    weight = model.conv.weight.detach().clone()
    dw = torch.randn_like(weight)
    dw *= PERTURBATION * weight.norm() / dw.norm()
    raw = torch.full((CHANNELS,), TAU + INPUT_MEAN ** 2)
    wrong = qbn_correction(w=weight.reshape(OUT_CHANNELS, -1),
                           dw=dw.reshape(OUT_CHANNELS, -1),
                           e=torch.zeros(CHANNELS), s=raw,
                           mu=fp32[0].float(), v=fp32[1].float(), group_size=GROUP)
    shift = quant[1] - fp32[1]
    assert not torch.allclose(wrong.d_var.double(), shift, rtol=5e-3, atol=1e-6)
    assert torch.allclose(correction.d_var.double(), shift, rtol=5e-3, atol=1e-6)


def test_pearson_r_matches_a_hand_computed_correlation():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert pearson_r(a, 2.0 * a + 1.0) == pytest.approx(1.0, abs=1e-12)
    assert pearson_r(a, -a) == pytest.approx(-1.0, abs=1e-12)
    b = torch.tensor([2.0, 1.0, 4.0, 3.0])
    expected = float(torch.corrcoef(torch.stack([a, b]).double())[0, 1])
    assert pearson_r(a, b) == pytest.approx(expected, rel=1e-9)


def test_pearson_r_is_nan_on_a_degenerate_vector():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert math.isnan(pearson_r(a, torch.ones(3)))
    assert math.isnan(pearson_r(torch.ones(1), torch.ones(1)))
    with pytest.raises(ValueError):
        pearson_r(a, torch.ones(4))


def test_relative_error_of_predicting_zero_is_one():
    measured_shift = torch.tensor([0.5, -1.5, 2.0])
    assert relative_error(torch.zeros(3), measured_shift) == pytest.approx(1.0)
    assert relative_error(measured_shift, measured_shift) == pytest.approx(0.0)
    assert math.isnan(relative_error(measured_shift, torch.zeros(3)))
    with pytest.raises(ValueError):
        relative_error(measured_shift, torch.zeros(4))


class _Correction:
    """The two fields :func:`site_vectors` reads off a QbnCorrection."""

    def __init__(self, d_mean: torch.Tensor, d_var: torch.Tensor) -> None:
        self.d_mean, self.d_var = d_mean, d_var


def test_site_vectors_pairs_the_raw_correction_with_the_stored_anchor():
    mu = torch.tensor([1.0, -2.0], dtype=torch.float64)
    v = torch.tensor([4.0, 9.0], dtype=torch.float64)
    fp32 = (torch.tensor([1.5, -2.0], dtype=torch.float64),
            torch.tensor([4.4, 9.0], dtype=torch.float64))
    quant = (torch.tensor([1.8, -1.7], dtype=torch.float64),
             torch.tensor([4.1, 8.6], dtype=torch.float64))
    correction = _Correction(torch.tensor([0.25, 0.30]), torch.tensor([-0.35, -0.40]))
    vectors = site_vectors(correction, (mu, v), fp32, quant)

    assert torch.allclose(vectors["mean_pred"], mu + correction.d_mean.double())
    assert torch.allclose(vectors["mean_shift_meas"], quant[0] - fp32[0])
    assert torch.allclose(vectors["var_pred"], v + correction.d_var.double())
    assert torch.allclose(vectors["var_shift_meas"], quant[1] - fp32[1])
    # The anchor is the stored buffer against the measured fp32 network, in units
    # of sqrt(v) and v, which is what section 7.2 asks for.
    assert torch.allclose(vectors["anchor_mean"], (mu - fp32[0]).abs() / v.sqrt())
    assert torch.allclose(vectors["anchor_var"], (v - fp32[1]).abs() / v)


def _stats(mean_r: float, var_r: float, mean_rel: float, var_rel: float,
           anchor_var: float) -> dict[str, float]:
    return {"mean_r": mean_r, "var_r": var_r, "mean_rel": mean_rel,
            "var_rel": var_rel, "anchor_var": anchor_var}


def test_verdicts_score_the_documents_thresholds():
    rows = verdicts(_stats(0.97, 0.40, 0.03, 0.90, 0.004))
    assert [row[3] for row in rows] == ["held"] * len(rows)

    refuted = verdicts(_stats(MEAN_R_REFUTES - 0.01, 0.40, 0.03, 0.90, 0.004))
    assert "REFUTED" in refuted[0][3]

    soft = verdicts(_stats((MEAN_R_REFUTES + MEAN_R_FLOOR) / 2, 0.40, 0.03, 0.90, 0.004))
    assert "REFUTED" not in soft[0][3] and soft[0][3] != "held"

    surprise = verdicts(_stats(0.97, 0.90, 0.03, 0.90, 0.004))
    assert "SURPRISE" in surprise[2][3]

    bad_anchor = verdicts(_stats(0.97, 0.40, 0.03, 0.90, 0.02))
    assert "DELETE THE VARIANCE TERM" in bad_anchor[4][3]


def _write_shards(root: str, shards: int, rows: int) -> None:
    """Write ``shards`` parquet files whose images encode their own row index."""
    from PIL import Image
    import pyarrow as pa
    import pyarrow.parquet as pq

    index = 0
    for shard in range(shards):
        images, labels = [], []
        for _ in range(rows):
            buffer = io.BytesIO()
            colour = (index // 65536, (index // 256) % 256, index % 256)
            Image.new("RGB", (1, 1), colour).save(buffer, format="PNG")
            images.append({"bytes": buffer.getvalue(), "path": f"{index}.png"})
            labels.append(index % 10)
            index += 1
        pq.write_table(pa.table({"image": images, "label": labels}),
                       os.path.join(root, f"shard-{shard:02d}.parquet"))


def _drawn_indices(root: str, limit: int, skip: int) -> set[int]:
    """Decode which rows ``load_val`` drew, from the colours it decoded."""
    def transform(image):
        return torch.tensor(image.getpixel((0, 0)), dtype=torch.float64)

    x, _ = load_val(root, limit, transform, skip=skip)
    return {int(row[0] * 65536 + row[1] * 256 + row[2]) for row in x}


def test_the_measurement_window_is_disjoint_from_the_evaluation_window(tmp_path):
    # This is the check the 2026-09-12 correction in issue 28 was written to
    # force: the previous version of this measurement calibrated on the first
    # rows of the evaluation tensor, which was transductive and, because the
    # shards are class ordered, a roughly 200-class sample. It flipped a
    # conclusion. The two windows must not share a single image.
    root = str(tmp_path)
    _write_shards(root, shards=VAL_SHARDS, rows=40)
    limit = 140
    evaluation = _drawn_indices(root, limit, skip=0)
    measurement = _drawn_indices(root, limit, skip=-(-limit // VAL_SHARDS))
    assert len(evaluation) == limit and len(measurement) == limit
    assert evaluation.isdisjoint(measurement)


def test_measure_skip_clears_the_evaluation_draws_per_shard_take():
    # load_val takes ceil(limit / shards) rows per shard, so any skip at least
    # that large starts past the evaluation window in every shard.
    assert MEASURE_SKIP >= -(-N_IMAGES // VAL_SHARDS)


def test_qbn_prediction_rows_scores_every_corrected_layer_and_spares_the_model():
    timm = pytest.importorskip("timm")
    torch.manual_seed(0)
    model = timm.create_model("resnet18", pretrained=False).eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.normal_(0.0, 0.5)
            module.running_var.uniform_(0.5, 2.0)
    before = {name: parameter.detach().clone()
              for name, parameter in model.state_dict().items()}

    x = torch.randn(8, 3, 64, 64)
    rows, reports = qbn_prediction_rows(model, x, passes=1, device="cpu", batch=4)

    after = model.state_dict()
    for name, value in before.items():
        assert torch.equal(value, after[name]), f"{name} was modified"

    corrected = {report.conv for report in reports.values() if report.corrected}
    assert corrected and {row["conv"] for row in rows} == corrected
    assert all(report.correction is None for report in reports.values()
               if not report.corrected)
    for row in rows:
        channels = model.get_submodule(row["conv"]).out_channels
        for vector in row["vectors"].values():
            assert vector.shape == (channels,)
            assert torch.isfinite(vector).all()

    pooled = statistics(pool(rows))
    assert pooled["channels"] == sum(len(row["vectors"]["mean_pred"]) for row in rows)
    assert set(pool(rows, "A")) == set(pool(rows))
