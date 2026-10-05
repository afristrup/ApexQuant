"""Zero-shot task accuracy: does near-lossless perplexity survive reasoning?

Low-bit methods can hold perplexity while degrading on downstream reasoning, so
"near-lossless" needs task anchors. This evaluates ARC-Easy, WinoGrande and
HellaSwag with length-normalised log-likelihood scoring over answer choices --
the lm-eval-harness convention, implemented directly so the pinned torch/CUDA
environment stays untouched. (PIQA's Hub loader is a legacy script unsupported
by datasets 5.x; WinoGrande substitutes, scored by the partial-evaluation
convention: logprob of the post-blank continuation given each option.)

Configs: FP16, progressive 2-bit x2 (4.06 effective bits) and x3 (6.09), on
Phi-1.5 and Mistral-7B. HellaSwag is subsampled to 2,000 validation examples
(binomial 95% interval ~ +-2pp); ARC-Easy (2,376) and WinoGrande (1,267) run
in full.

    HF_HOME=... uv run python experiments/zero_shot_eval.py
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerun_apexquant_llm import DEV  # noqa: E402

from apexquant import (  # noqa: E402
    ApexQuantConfig,
    dequantize_weight,
    turboquantize_weight,
)

MODELS = {
    "Phi-1.5": "microsoft/phi-1_5",
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
}
HELLASWAG_N = 2000


def load_tasks(limit: int = 0):
    """limit > 0 subsamples each task (seeded) for a fast smoke run: the
    binomial 95% interval at n=300 is ~ +-5.7pp, directional only."""
    tasks = {}
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    tasks["ARC-Easy"] = [
        (f"Question: {ex['question']}\nAnswer:",
         [" " + c for c in ex["choices"]["text"]],
         ex["choices"]["label"].index(ex["answerKey"]))
        for ex in arc if ex["answerKey"] in ex["choices"]["label"]]
    wino = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    tasks["WinoGrande"] = []
    for ex in wino:
        pre, _, post = ex["sentence"].partition("_")
        # Partial evaluation: same continuation (post) scored under each option
        # substituted into the context; gold answer is "1" or "2".
        tasks["WinoGrande"].append(
            ((pre + ex["option1"], pre + ex["option2"]), [post, post],
             int(ex["answer"]) - 1))
    hs = load_dataset("Rowan/hellaswag", split="validation")
    hs = hs.shuffle(seed=0).select(range(HELLASWAG_N))
    tasks["HellaSwag"] = [
        (ex["ctx"], [" " + e for e in ex["endings"]], int(ex["label"]))
        for ex in hs]
    if limit:
        rng = random.Random(0)
        for name, ex in tasks.items():
            rng.shuffle(ex)
            tasks[name] = ex[:limit]
    return tasks


@torch.no_grad()
def choice_logprob(model, tok, context: str, choice: str) -> float:
    """Length-normalised logprob of ``choice`` given ``context``."""
    ctx = tok(context, return_tensors="pt").input_ids
    full = tok(context + choice, return_tensors="pt").input_ids
    n_choice = full.shape[1] - ctx.shape[1]
    if n_choice <= 0:
        return -1e9
    logits = model(full.to(DEV)).logits[0]
    lp = torch.log_softmax(logits[:-1].float(), dim=-1)
    tgt = full[0, 1:].to(DEV)
    tail = lp[-n_choice:].gather(1, tgt[-n_choice:, None]).sum()
    return float(tail) / n_choice


@torch.no_grad()
def accuracy(model, tok, examples) -> float:
    correct = 0
    for ctx, choices, gold in examples:
        # ctx is one string shared by all choices, or one context per choice
        # (WinoGrande partial evaluation).
        ctxs = ctx if isinstance(ctx, tuple) else (ctx,) * len(choices)
        scores = [choice_logprob(model, tok, c0, c) for c0, c in zip(ctxs, choices)]
        correct += int(max(range(len(scores)), key=scores.__getitem__) == gold)
    return correct / len(examples)


def quantise_(model, passes: int) -> None:
    cfg = ApexQuantConfig(bits=2, passes=passes, codebook="apex", group_size=512)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "layers." in name and mod.weight.shape[1] >= 512:
            qw = turboquantize_weight(mod.weight.data.to(torch.float32), cfg)
            mod.weight.data = dequantize_weight(qw).to(mod.weight.dtype)
            del qw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma list of keys from MODELS")
    ap.add_argument("--limit", type=int, default=0,
                    help="subsample each task to N examples (0 = full)")
    args = ap.parse_args()

    tasks = load_tasks(args.limit)
    print("# Zero-shot accuracy: FP16 vs progressive 2-bit x2 / x3\n")
    print("# length-normalised log-likelihood over choices (lm-eval convention).",
          flush=True)
    if args.limit:
        print(f"# SMOKE RUN: tasks subsampled to {args.limit} examples "
              "(binomial 95% ~ +-6pp at n=300); directional only.\n", flush=True)
    else:
        print(flush=True)
    for mname in args.models.split(","):
        repo = MODELS[mname]
        tok = AutoTokenizer.from_pretrained(repo)
        print(f"## {mname}\n")
        print("| config | eff. bits | " + " | ".join(tasks) + " |")
        print("|---|---:|" + "---:|" * len(tasks))
        for label, passes, eb in [("FP16", 0, "16.00"),
                                  ("2-bit x2", 2, "4.06"),
                                  ("2-bit x3", 3, "6.09")]:
            model = AutoModelForCausalLM.from_pretrained(
                repo, dtype=torch.float16).to(DEV).eval()
            if passes:
                quantise_(model, passes)
            accs = [accuracy(model, tok, ex) for ex in tasks.values()]
            print(f"| {label} | {eb} | "
                  + " | ".join(f"{a*100:.1f}%" for a in accs) + " |", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
