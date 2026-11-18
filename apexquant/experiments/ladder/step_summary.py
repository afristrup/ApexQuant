"""Collect finished rungs into one CSV per model, ready for both rate-quality plots."""

from __future__ import annotations

import csv
import glob
import json
import os
from statistics import mean

from .settings import TASKS

DATA = {"apex": "none", "apex_l0": "none", "apex_sinkhorn": "none", "mixed": "none", "residual_rtn": "none", "hqq": "none", "sinq": "none",
        "dash": "none", "gptq": "calibrated", "awq": "calibrated", "liftquant": "calibrated"}


def step_summary(work: str) -> None:
    """Write ``summary/e1_rows.csv``; the ``data`` column splits the two plots."""
    rows = []
    for path in sorted(glob.glob(os.path.join(work, "results", "*.json"))):
        r = json.load(open(path))
        row = {k: r[k] for k in ("model", "arm", "level", "group", "bits_per_weight",
                                 "bits_per_quantized_weight",
                                 "encode_seconds", "encode_peak_gpu_bytes_above_model")}
        row.update({"data": DATA[r["arm"]],
                    **{k: r["kl"][k] for k in ("kl_mean", "kl_stderr", "kl_median", "kl_p99",
                                               "kl_p999", "same_top_p", "delta_p_rms")}, "accuracy_mean": mean(r["accuracy"][t] for t in TASKS)})
        row.update({f"acc_{t}": r["accuracy"][t] for t in TASKS})
        rows.append(row)
    if not rows:
        raise RuntimeError("no finished rungs under results/")
    out = os.path.join(work, "summary")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "e1_rows.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["model"], r["data"], r["bits_per_weight"])))
