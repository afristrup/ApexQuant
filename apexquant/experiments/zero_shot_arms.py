"""Zero-shot task accuracy for the arms of tab:llm-full.

KL against the unquantized model is a model-preservation metric, and a referee is
entitled to ask whether the ordering it induces survives on a task. This runs the
same arms, through the same code, on ARC-Easy, WinoGrande and HellaSwag.

The arms are imported from ``iclr_llm_table`` rather than reimplemented. That is
the whole point of the file: a second implementation of "our 4-bit arm" would make
any disagreement with the KL table unattributable. The task machinery is imported
from ``zero_shot_eval`` for the same reason.

Scoring is length-normalised log-likelihood over the answer choices, the
lm-eval-harness convention, so the numbers are comparable with published ones even
though the harness is local.

    HF_HOME=/mnt/bulk/hf uv run python experiments/zero_shot_arms.py --out results.jsonl
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import pathlib
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iclr_llm_table import (  # noqa: E402
    DEV,
    GROUP,
    HQQ_GROUP,
    MODELS,
    SINQ_GROUP,
    TRELLIS_BITS,
    apply_hqq,
    apply_progressive,
    apply_sinq,
    apply_vector_stage,
    collect_awq_scales,
    target_linears,
)
from zero_shot_eval import accuracy, load_tasks  # noqa: E402

from apexquant.vq.trellis_codec import TrellisCodec  # noqa: E402

# (label, arm, bits). The 4-bit class reaches its rate as two 2-bit passes for a
# vector stage, which is the only form that keeps a valid 2-bit prefix, so the
# label and the bit-width deliberately disagree there.
CELLS = [
    ("fp16", None, 0),
    ("trellis 4-bit class", "apex-tcq-P2", 2),
    ("trellis 2-bit class", "apex-tcq", 2),
    ("scalar 4-bit class", "apex", 4),
    ("HQQ g=64, 4-bit", "hqq", 4),
    ("HQQ g=64, 2-bit", "hqq", 2),
    ("A-SINQ, 4-bit", "asinq", 4),
]


def build_arm(model, arm: str, bits: int, tok, caches: dict) -> None:
    """Apply one arm in place, using the table harness's own code paths."""
    if arm.startswith("apex-tcq") and bits not in caches["tcq"]:
        caches["tcq"][bits] = TrellisCodec(d=GROUP, bits=bits,
                                           trellis_bits=TRELLIS_BITS, device=DEV)
    if arm == "asinq" and bits not in caches["awq"]:
        caches["awq"][bits] = collect_awq_scales(model, tok, bits)
    passes = 2 if arm.endswith("-P2") else 1
    for name, mod in target_linears(model):
        if arm == "apex":
            apply_progressive(mod, bits, passes=1)
        elif arm.startswith("apex-tcq"):
            apply_vector_stage(mod, bits, passes, caches["tcq"][bits])
        elif arm == "hqq":
            apply_hqq(mod, bits, group=HQQ_GROUP)
        elif arm == "asinq":
            apply_sinq(mod, bits, caches["awq"][bits].get(name), group=SINQ_GROUP)
        else:
            raise ValueError(f"unknown arm {arm}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0,
                    help="subsample each task; 0 runs them in full")
    ap.add_argument("--out", default="zero_shot_arms.jsonl",
                    help="JSONL of finished cells, appended to and resumed from")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            d = json.loads(line)
            done.add((d["model"], d["label"]))

    tasks = load_tasks(args.limit)
    names = ", ".join(f"{k} ({len(v)})" for k, v in tasks.items())
    print(f"# zero-shot accuracy, length-normalised log-likelihood over choices")
    print(f"# tasks: {names}")
    if args.limit:
        print(f"# SMOKE RUN: subsampled to {args.limit} per task, directional only")
    print(flush=True)

    for mname in args.models.split(","):
        repo = MODELS[mname]
        tok = AutoTokenizer.from_pretrained(repo)
        caches = {"tcq": {}, "awq": {}}
        pristine = None
        for label, arm, bits in CELLS:
            if (mname, label) in done:
                continue
            model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.float16).to(DEV).eval()
            if arm is not None:
                if pristine is None:
                    pristine = {n: m.weight.data.clone().cpu()
                                for n, m in target_linears(model)}
                build_arm(model, arm, bits, tok, caches)
            row = {"model": mname, "label": label, "arm": arm, "bits": bits}
            for task, examples in tasks.items():
                row[task] = accuracy(model, tok, examples)
                print(f"  {mname:16s} {label:22s} {task:12s} "
                      f"{100 * row[task]:6.2f}", flush=True)
            row["mean"] = sum(row[t] for t in tasks) / len(tasks)
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
