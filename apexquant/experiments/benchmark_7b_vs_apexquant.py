"""7B-scale benchmark: progressive 2-bit vs ApexQuant 2-bit AND 4-bit.

Mistral-7B-v0.1 and Qwen2.5-7B are ApexQuant's strongest illustration (its
Table 2), and their fan-ins (4096-18944) sit squarely in the regime where the
fused 2-bit GEMV also wins latency. This runs the paper's Table-3 comparison at
7B scale: WikiText-2 raw test perplexity (seq 2048) for

    FP16                      (reference)
    ApexQuant 2-bit x1        (single pass -- their method at 2 bits)
    ours: apex 2-bit x2/x3    (progressive residual passes, same codebook)
    ApexQuant 4-bit x1        (their method at 4 bits; matched budget for x2)

All quantised rows use the library pipeline (density-matched codebook, grouped
rotation gs=512) so passes is the only variable. A 7B model in fp16 is ~14 GB;
we reload from disk per config instead of cloning the state dict, and quantise
layer by layer in place on the GPU.

    HF_HOME=... uv run python experiments/benchmark_7b_vs_apexquant.py
"""

from __future__ import annotations

import gc
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerun_apexquant_llm import DEV, perplexity  # noqa: E402

from apexquant import (  # noqa: E402
    ApexQuantConfig,
    dequantize_weight,
    turboquantize_weight,
)

GS = 512
MODELS = {
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
}
# (label, bits, passes); passes=0 means FP16 reference.
CONFIGS = [
    ("FP16", 16, 0),
    ("ApexQuant 2-bit (single pass)", 2, 1),
    ("Ours: apex 2-bit x2", 2, 2),
    ("Ours: apex 2-bit x3", 2, 3),
    ("ApexQuant 4-bit (single pass)", 4, 1),
]


def quantize_model_(model, bits: int, passes: int) -> int:
    cfg = ApexQuantConfig(bits=bits, passes=passes, codebook="apex", group_size=GS)
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "layers." in name and mod.weight.shape[1] >= 512:
            w = mod.weight.data
            qw = turboquantize_weight(w.to(torch.float32), cfg)
            mod.weight.data = dequantize_weight(qw).to(w.dtype)
            del qw
            n += 1
    return n


def main():
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    print("# 7B benchmark: progressive 2-bit vs ApexQuant 2-bit / 4-bit "
          "(WikiText-2 ppl, seq 2048)\n")
    print(f"# density-matched codebook, gs={GS}; decoder Linear weights; "
          "model reloaded from disk per config.\n")
    for mname, repo in MODELS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        enc = tok(text, return_tensors="pt").input_ids
        print(f"## {mname}  ({enc.numel()} test tokens)\n")
        print("| method | eff. bits | perplexity |")
        print("|--------|----------:|-----------:|")
        for label, bits, passes in CONFIGS:
            model = AutoModelForCausalLM.from_pretrained(
                repo, dtype=torch.float16).to(DEV).eval()
            if passes:
                nq = quantize_model_(model, bits, passes)
            eb = 16.0 if passes == 0 else (bits + 16.0 / GS) * passes
            p = perplexity(model, (enc,))
            pstr = f"{p:.3f}" if p < 1000 else f"{p:.3e}"
            print(f"| {label:<29} | {eb:>8.2f} | {pstr:>10} |", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
