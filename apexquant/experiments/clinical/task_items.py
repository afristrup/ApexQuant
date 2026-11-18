"""Per-item correctness that the harness logged for one task."""

from __future__ import annotations

import glob
import json
import os


def task_items(out_dir: str, task: str) -> dict[int, float]:
    """``doc_id -> acc`` from the newest ``samples_{task}_*.jsonl`` under ``out_dir``.

    A rerun of the same harness call writes a second, timestamped file next to the
    first; the newest one is the run whose results the summary also reads.
    """
    files = glob.glob(os.path.join(out_dir, "**", f"samples_{task}_*.jsonl"), recursive=True)
    if not files:
        raise RuntimeError(f"{out_dir}: no sample file for {task}")
    newest = max(files, key=os.path.getmtime)
    return {row["doc_id"]: float(row["acc"]) for row in map(json.loads, open(newest))}
