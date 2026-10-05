"""Our pipeline vs vanilla ApexQuant on the real paper LLMs (WikiText-2 perplexity).

rerun_apexquant_llm.py reproduced ApexQuant's single-pass table. Here we put OUR
pipeline -- the library's density-matched codebook with progressive residual passes
(turboquantize_weight + dequantize_weight, grouped rotation, group_size=512) --
head-to-head against vanilla single-pass ApexQuant on the same real TinyLlama-1.1B
and Phi-1.5, at a 2-bit codebook. Each extra residual pass adds ~2 effective bits
but reuses the SAME codebook (the re-isotropization theorem), so progressive 2-bit
should recover quality that single-pass ApexQuant leaves on the table.

Reference row: single-pass ApexQuant at 4 bits, to compare our 2-bit x2 (both ~4
effective bits) at matched storage.

    HF_HOME=... uv run python experiments/benchmark_ours_vs_apexquant_llm.py
"""

from __future__ import annotations

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset  # noqa: E402
from rerun_apexquant_llm import DEV, MODELS, perplexity  # noqa: E402

from apexquant import (  # noqa: E402
    ApexQuantConfig,
    dequantize_weight,
    turboquantize_weight,
)

GS = 512


def apply_ours_(model, bits: int, passes: int) -> None:
    """Replace every decoder Linear weight with our pipeline's reconstruction."""
    cfg = ApexQuantConfig(bits=bits, passes=passes, codebook="apex", group_size=GS)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "layers." in name and mod.weight.shape[1] >= 512:
            w = mod.weight.data
            qw = turboquantize_weight(w.to(DEV, torch.float32), cfg)
            mod.weight.data = dequantize_weight(qw).to(w.dtype)


def eff_bits(bits: int, passes: int) -> float:
    return (bits + 16.0 / GS) * passes


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="",
                    help="comma list of model keys; default all")
    args = ap.parse_args()
    models = ({k: MODELS[k] for k in args.models.split(",")} if args.models
              else MODELS)
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    # (label, bits, passes)
    configs = [
        ("ApexQuant 2-bit (single pass)", 2, 1),
        ("Ours: apex 2-bit x2", 2, 2),
        ("Ours: apex 2-bit x3", 2, 3),
        ("ApexQuant 4-bit (ref, matched to x2)", 4, 1),
        # The matched-budget baseline at our HEADLINE operating point. Our own
        # design chart predicts a single wide pass wins here; print it anyway.
        ("ApexQuant 6-bit (ref, matched to x3)", 6, 1),
    ]
    print("# Our progressive pipeline vs vanilla ApexQuant -- real LLMs, WikiText-2\n")
    print(f"# density-matched codebook, group_size={GS}, decoder Linear weights.\n")
    for mname, repo in models.items():
        tok = AutoTokenizer.from_pretrained(repo)
        enc = tok(text, return_tensors="pt").input_ids
        base = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16).to(DEV).eval()
        ppl_fp = perplexity(base, (enc,))
        sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
        print(f"## {mname}\n")
        print("| method                                | eff. bits | perplexity |")
        print("|---------------------------------------|----------:|-----------:|")
        print(f"| FP16                                  |     16.00 | {ppl_fp:>10.3f} |")
        for label, bits, passes in configs:
            base.load_state_dict(sd)
            apply_ours_(base, bits, passes)
            p = perplexity(base, (enc,))
            pstr = f"{p:.3f}" if p < 1000 else f"{p:.3e}"
            print(f"| {label:<37} | {eff_bits(bits, passes):>9.2f} | {pstr:>10} |")
        base.load_state_dict(sd)
        del base, sd
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
