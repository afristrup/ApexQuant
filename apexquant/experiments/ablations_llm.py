"""End-to-end validation of the two decisive ablations on real Phi-1.5 perplexity.

The reconstruction ablations (ablations.py) predict two things end-to-end:
  A1  fresh rotation per pass beats a fixed rotation (re-isotropization), and
  A2  the density-matched (apex) codebook beats the uniform grid at every pass.
Here we confirm both on WikiText-2 perplexity of the real Phi-1.5, applying the
same unified quantizer to each decoder Linear weight. One knob changes per column.

    HF_HOME=... uv run python experiments/ablations_llm.py
"""

from __future__ import annotations

import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablations import reconstruct  # noqa: E402
from rerun_apexquant_llm import DEV, perplexity  # noqa: E402

REPO = "microsoft/phi-1_5"


def apply_(model, **kw):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "layers." in name and mod.weight.shape[1] >= 512:
            w = mod.weight.data
            mod.weight.data = reconstruct(w.to(DEV, torch.float32), **kw).to(w.dtype)


def main():
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    tok = AutoTokenizer.from_pretrained(REPO)
    enc = tok(text, return_tensors="pt").input_ids
    base = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float16).to(DEV).eval()
    ppl_fp = perplexity(base, (enc,))
    sd = {k: v.detach().clone() for k, v in base.state_dict().items()}

    def run(**kw):
        base.load_state_dict(sd)
        apply_(base, **kw)
        p = perplexity(base, (enc,))
        return f"{p:.3f}" if p < 1000 else f"{p:.3e}"

    common = dict(bits=2, gs=512, rotation="hadamard")
    print(f"# End-to-end ablations on real Phi-1.5 (WikiText-2 perplexity, FP16={ppl_fp:.3f})\n")

    print("## A1  Fresh vs fixed rotation per pass (apex codebook)\n")
    print("| passes | fresh | fixed |")
    print("|-------:|-------|-------|")
    for P in (1, 2, 3):
        fresh = run(**common, codebook="apex", passes=P, fresh=True)
        fixed = run(**common, codebook="apex", passes=P, fresh=False)
        print(f"| {P} | {fresh} | {fixed} |")

    print("\n## A2  Density-matched (apex) vs uniform codebook (fresh rotation)\n")
    print("| passes | apex | uniform |")
    print("|-------:|------|---------|")
    for P in (1, 2, 3):
        apex = run(**common, codebook="apex", passes=P, fresh=True)
        unif = run(**common, codebook="uniform", passes=P, fresh=True)
        print(f"| {P} | {apex} | {unif} |")


if __name__ == "__main__":
    main()
