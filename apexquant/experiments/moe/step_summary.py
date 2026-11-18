"""Collect finished E6 rows into one CSV."""

from __future__ import annotations

import csv
import glob
import json
import os
from statistics import mean

from .settings import TASKS


def step_summary(work: str) -> None:
    """Write ``summary/e6_rows.csv`` with KL, accuracy, envelope and controller share."""
    rows = []
    for path in sorted(glob.glob(os.path.join(work, "results", "*.json"))):
        r = json.load(open(path))
        row = {k: r[k] for k in ("model", "policy", "average_bits", "encode_seconds")}
        row.update({"kl_mean": r["kl"]["kl_mean"], "kl_p99": r["kl"]["kl_p99"],
                    "accuracy_mean": mean(r["accuracy"][t] for t in TASKS),
                    "controller_share": r["controller_ms"] / r["forward_ms"], **r["envelope"]})
        row.update({f"acc_{t}": r["accuracy"][t] for t in TASKS})
        rows.append(row)
    if not rows:
        raise RuntimeError("no finished rows under results/")
    out = os.path.join(work, "summary")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "e6_rows.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["model"], r["average_bits"], r["policy"])))
