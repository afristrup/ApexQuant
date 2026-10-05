"""The matched-budget single-pass baseline at our HEADLINE operating point.

Table `tab:llm` compares the three-pass 6.09-bit configuration against a 4.03-bit
single pass and 4.13-bit GPTQ. Neither is matched to 6.09 bits, so the flagship
row has no like-for-like single-pass baseline -- and our own design chart
predicts we lose there. That absence looks like concealment; this script removes
it.

The library's `ApexQuantConfig` restricts `bits` to {2,4,8} because `pack_bits`
packs indices into bytes. That is a storage constraint, not a codebook one:
`apex_beta_levels` fits a Lloyd-Max codebook for any level count. Since this
measurement is quality-only (reconstruct, then run the dense forward), we
replicate `turboquantize_weight`'s arithmetic here -- identical rotation,
normaliser and level assignment -- without packing, so 3-, 5- and 6-bit
single-pass points become measurable. Nothing in the library is modified, so the
published numbers are unaffected; the b in {2,4,8} rows below are a cross-check
against the library path and must agree.

    HF_HUB_OFFLINE=1 uv run python experiments/matched_budget_baseline.py --models Phi-1.5
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerun_apexquant_llm import DEV, MODELS, perplexity  # noqa: E402

from apexquant.codebook.apex_beta_levels import apex_beta_levels  # noqa: E402
from apexquant.quantize.quantize_to_indices import quantize_to_indices  # noqa: E402
from apexquant.rotation.inverse_random_rotation import (  # noqa: E402
    inverse_random_rotation,
)
from apexquant.rotation.random_rotation import random_rotation  # noqa: E402

GS = 512
SEED = 42

# The 7B repos are registered in the benchmark scripts rather than in
# rerun_apexquant_llm.MODELS, which several scripts iterate wholesale to
# reproduce published tables; extend the mapping here instead of editing that
# dict, so nothing else changes behaviour.
MODELS = {
    **MODELS,
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
}


def reconstruct(weight: torch.Tensor, bits: int, passes: int,
                group_size: int = GS, seed: int = SEED) -> torch.Tensor:
    """Progressive apex quantisation -> reconstruction, unpacked.

    Mirrors `turboquantize_weight` + `dequantize_weight` exactly: fresh rotation
    per pass, per-group L2 normalisation onto the sphere, nearest apex level,
    reconstruct and peel.
    """
    out_f, in_f = weight.shape
    gs = min(group_size, 1 << (in_f.bit_length() - 1)) if in_f < group_size else group_size
    levels = apex_beta_levels(bits, gs, device=weight.device)
    pad = (-in_f) % gs
    w = F.pad(weight, (0, pad)).to(torch.float32)
    ngroups = w.shape[1] // gs
    w = w.view(out_f, ngroups, gs)

    residual = w
    acc = torch.zeros_like(w)
    for p in range(passes):
        s = seed + p
        rotated = random_rotation(residual, s)
        scale = rotated.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        idx = quantize_to_indices(rotated / scale, levels).long()
        recon_rot = scale * levels[idx]
        back = inverse_random_rotation(recon_rot, s)
        acc = acc + back
        residual = residual - back
    dense = acc.reshape(out_f, ngroups * gs)
    return dense[:, :in_f].contiguous()


def quantise_(model, bits: int, passes: int) -> None:
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            if mod.in_features < 32:
                continue
            w = mod.weight.data
            mod.weight.data = reconstruct(w.to(torch.float32), bits, passes).to(w.dtype)


def eff_bits(bits: int, passes: int) -> float:
    return (bits + 16.0 / GS) * passes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Phi-1.5")
    ap.add_argument("--configs", default="2x1,2x2,2x3,4x1,6x1,8x1,3x2",
                    help="comma list of BITSxPASSES")
    args = ap.parse_args()

    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])

    configs = []
    for tok_ in args.configs.split(","):
        b, p = tok_.lower().split("x")
        configs.append((int(b), int(p)))
    configs.sort(key=lambda bp: eff_bits(*bp))

    print("# Matched-budget baselines: single wide pass vs progressive passes\n")
    print(f"# apex codebook, group_size={GS}, decoder Linear weights, WikiText-2.\n")
    for mname in args.models.split(","):
        repo = MODELS[mname]
        tok = AutoTokenizer.from_pretrained(repo)
        enc = tok(text, return_tensors="pt").input_ids
        base = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16).to(DEV).eval()
        ppl_fp = perplexity(base, (enc,))
        # Keep the pristine copy on the host: a 7B fp16 model plus a GPU-side
        # clone is ~27 GiB and does not fit in 24 GiB. Exact copy either way,
        # so the 1.1B/1.3B numbers are unaffected.
        sd = {k: v.detach().to("cpu", copy=True) for k, v in base.state_dict().items()}
        print(f"## {mname}\n")
        print("| config | eff. bits | perplexity |")
        print("|---|---:|---:|")
        print(f"| FP16 | 16.00 | {ppl_fp:.3f} |")
        for b, p in configs:
            base.load_state_dict(sd)
            quantise_(base, b, p)
            val = perplexity(base, (enc,))
            label = (f"progressive {b}-bit x{p}" if p > 1
                     else f"single-pass {b}-bit")
            pstr = f"{val:.3f}" if val < 1000 else f"{val:.3e}"
            print(f"| {label} | {eff_bits(b, p):.2f} | {pstr} |", flush=True)
        del base, sd
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
