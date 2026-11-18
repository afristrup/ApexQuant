"""Collect every finished arm into the rows of the medical and shift tables."""

from __future__ import annotations

import csv
import glob
import json
import os
from statistics import mean, pstdev

from .settings import CLINICAL_TASKS, REFERENCES, RESULTS
from .significance_gate import significance_gate
from .spearman import spearman


def step_summarize(work: str) -> None:
    """Write ``summary/e3_rows.csv`` and ``summary/e3.json``; missing references fail fast."""
    refs = {v: json.load(open(os.path.join(work, REFERENCES, v, "summary.json")))
            for v in ("base", "ft", "controlled", "constrained")}
    clinical = {v: mean(r["accuracy"][t] for t in CLINICAL_TASKS) for v, r in refs.items()}
    gates = {v: significance_gate(os.path.join(work, REFERENCES, v, "lm_eval"),
                                  os.path.join(work, REFERENCES, "base", "lm_eval"),
                                  CLINICAL_TASKS) for v in ("ft", "constrained")}
    rows = []
    for path in sorted(glob.glob(os.path.join(work, RESULTS, "*.json"))):
        r = json.load(open(path))
        v, kl = r["variant"], r["kl"]
        a_q = mean(r["accuracy"][t] for t in CLINICAL_TASKS)
        gain = clinical[v] - clinical["base"]
        layers = [m for m in r["layers"].values() if "reversion" in m]
        row = {"variant": v, "arm": r["arm"], "level": r["level"], "calib": r["calib"],
               "seed": r["seed"], "clinical_acc": a_q,
               "retention": (a_q - clinical["base"]) / gain if v in gates and gates[v]["passes"] else None,
               "undo_index": kl["pubmed"]["kl_mean"] / refs[v]["kl_to_base"]["pubmed"]["kl_mean"],
               "reversion": mean(m["reversion"] for m in layers),
               "nonvacuous_share": mean(m["bound_nonvacuous"] for m in layers),
               "kl_generic": kl["wikitext"]["kl_mean"]}
        for name in ("pubmed", "code", "danish"):
            row[f"{name}_over_generic"] = kl[name]["kl_mean"] / kl["wikitext"]["kl_mean"]
        if len(layers) >= 3:
            row["spearman_kappa_reversion"] = spearman([m["kappa_delta"] for m in layers],
                                                       [m["reversion"] for m in layers])
            row["spearman_channel_reversion"] = spearman(
                [m["channel_weighted_shift"] for m in layers], [m["reversion"] for m in layers])
        if r["shrink"] is not None:
            row["haar_reversion"] = mean(m["haar_reversion"] for m in layers)
        rows.append(row)
    if not rows:
        raise RuntimeError("no finished arms under results/")
    spread = {}
    for row in rows:
        if row["calib"].startswith("c4_"):
            spread.setdefault((row["variant"], row["arm"], row["level"]), []).append(row["pubmed_over_generic"])
    out = os.path.join(work, "summary")
    os.makedirs(out, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with open(os.path.join(out, "e3_rows.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json.dump({"clinical_fp16": clinical, "gates": gates,
               "draw_sd_pubmed_over_generic": {"/".join(map(str, k)): pstdev(v)
                                               for k, v in spread.items() if len(v) > 1},
               "references": refs}, open(os.path.join(out, "e3.json"), "w"), indent=1)
