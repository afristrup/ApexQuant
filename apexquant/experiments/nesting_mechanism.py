"""Two ways to make a quantised model elastic, both data-free, matched on everything else.

Our nesting comes from residual passes: pass p is a fresh rotation of what passes
< p left behind, so the first P' passes ARE the P'-pass quantisation and every
prefix is itself an optimally-quantised model (Corollary 1).

The other mechanism in the literature is MSB truncation of one wide code: store a
B-bit index and read back its top k bits. Any-Precision LLM stores the code
bit-plane-wise so a k-bit read costs k planes of bandwidth, and MatQuant slices
an integer datatype the same way.

CONTROLLING THE COMPARISON. A naive version of this experiment quantises the MSB
arm on the raw weights and finds it catastrophic -- but that measures the absence
of incoherence processing, not the nesting mechanism (the paper's own Table 2 has
unrotated RTN at 3.5e5 on Phi-1.5 at two bits). So both arms here get the
*identical* front end: the same per-group random rotation, the same L2
normalisation onto the sphere, and the same density-matched Lloyd-Max codebook
family. They differ in one thing only -- how the lower tiers are produced.

  residual passes (ours):  P independent passes of a 2-bit apex codebook; tier P'
                           is the first P' passes.
  MSB truncation:          ONE pass with a B-bit apex codebook; tier k merges
                           each group of 2^(B-k) adjacent codebook cells and
                           reconstructs at the merged cell's centroid, which is
                           exactly "index by the top k bits of the code".

Both are data-free and both are genuinely nested. Neither is trained.

READ THE RESULT CORRECTLY. MatQuant and Any-Precision both train (10-100M C4
tokens; a calibrated seed model) and they train precisely to repair what an
untrained sliced code loses. This comparison therefore does NOT say those methods
are bad -- it isolates what their training has to buy, against a nesting that
needs no training to get optimal tiers.

    HF_HUB_OFFLINE=1 uv run python experiments/nesting_mechanism.py --models Phi-1.5
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
from matched_budget_baseline import reconstruct as residual_reconstruct  # noqa: E402
from rerun_apexquant_llm import DEV, MODELS, perplexity  # noqa: E402

from apexquant.codebook.apex_beta_levels import apex_beta_levels  # noqa: E402
from apexquant.quantize.quantize_to_indices import quantize_to_indices  # noqa: E402
from apexquant.rotation.inverse_random_rotation import (  # noqa: E402
    inverse_random_rotation,
)
from apexquant.rotation.random_rotation import random_rotation  # noqa: E402

GS = 512
SEED = 42


def merged_levels(parent_bits: int, read_bits: int, gs: int, device) -> torch.Tensor:
    """The k-bit codebook a top-k-bits read induces on a B-bit apex codebook.

    Truncating the index to its top `read_bits` bits merges each run of
    2^(B-k) adjacent cells; the MSE-optimal reconstruction for a merged cell is
    the mean of the levels it contains, so the induced k-bit codebook is the
    block-mean of the parent's levels. Returns a table indexed by the PARENT
    code, so the caller just gathers.
    """
    lv = apex_beta_levels(parent_bits, gs, device=device)      # (2^B,) ascending
    run = 1 << (parent_bits - read_bits)
    merged = lv.view(-1, run).mean(dim=1)                      # (2^k,)
    return merged.repeat_interleave(run)                       # (2^B,)


def msb_reconstruct(weight: torch.Tensor, parent_bits: int, read_bits: int,
                    group_size: int = GS, seed: int = SEED) -> torch.Tensor:
    """One wide apex pass, read back at `read_bits` by truncating the index."""
    out_f, in_f = weight.shape
    gs = group_size if in_f >= group_size else 1 << (in_f.bit_length() - 1)
    parent = apex_beta_levels(parent_bits, gs, device=weight.device)
    table = merged_levels(parent_bits, read_bits, gs, weight.device)

    pad = (-in_f) % gs
    w = F.pad(weight, (0, pad)).to(torch.float32).view(out_f, -1, gs)
    rotated = random_rotation(w, seed)
    scale = rotated.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
    idx = quantize_to_indices(rotated / scale, parent).long()
    recon_rot = scale * table[idx]
    dense = inverse_random_rotation(recon_rot, seed).reshape(out_f, -1)
    return dense[:, :in_f].contiguous()


def _apply(model, fn) -> None:
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            if mod.in_features < 32:
                continue
            w = mod.weight.data
            mod.weight.data = fn(w.to(torch.float32)).to(w.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Phi-1.5")
    ap.add_argument("--parent-bits", type=int, default=6)
    args = ap.parse_args()
    B = args.parent_bits

    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])

    print("# Two nesting mechanisms, matched front end, both data-free\n")
    print(f"# Both arms: same per-group rotation, same L2 normalisation, same apex")
    print(f"# Lloyd-Max codebook family, group_size={GS}. Parent for the MSB arm is")
    print(f"# {B}-bit; tier k reads its top k bits. WikiText-2, decoder Linears.\n")

    for mname in args.models.split(","):
        repo = MODELS[mname]
        tok = AutoTokenizer.from_pretrained(repo)
        enc = tok(text, return_tensors="pt").input_ids
        base = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16).to(DEV).eval()
        ppl_fp = perplexity(base, (enc,))
        sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
        print(f"## {mname}  (FP16 {ppl_fp:.3f})\n")
        print(f"| tier | eff. bits | MSB read of {B}-bit parent | residual passes (ours) |")
        print("|---|---:|---:|---:|")

        f = lambda v: f"{v:.3f}" if v < 1000 else f"{v:.3e}"  # noqa: E731
        for read_bits, passes in [(2, 1), (4, 2), (6, 3)]:
            base.load_state_dict(sd)
            _apply(base, lambda w: msb_reconstruct(w, B, read_bits))
            p_msb = perplexity(base, (enc,))
            base.load_state_dict(sd)
            _apply(base, lambda w: residual_reconstruct(w, 2, passes))
            p_res = perplexity(base, (enc,))
            print(f"| top-{read_bits} bits / {passes} pass(es) | "
                  f"{read_bits + 16.0/GS:.2f} | {f(p_msb)} | {f(p_res)} |", flush=True)

        del base, sd
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
