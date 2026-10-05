"""Progressive refinement with a VECTOR stage quantiser, end to end on a real model.

This closes the largest concession in the paper. `tab:stages` shows that swapping
the scalar Lloyd-Max stage for an E8 lattice or a trellis code cuts the per-pass
distortion (rho^2 = 0.1171 -> 0.0906 -> 0.0702) and that the geometric law holds
for all three, but it measures that on uniform directions only, and the paper
concedes: "we have not run an end-to-end model with a lattice or trellis stage".
This script runs one.

WHY THIS IS STILL DATA-FREE. This is the point of the exercise. Every method we
borrow from splits, in YAQA's own words, into "a quantizer Q that defines a set
of representable points" and "a rounding algorithm that assigns theta* -> C".
The rounding half (LDLQ, BlockLDLQ, YAQA's Hessian sketch) needs a calibration
corpus; the quantizer half does not. E8 is a *fixed* lattice -- QuIP# gives a
counterexample to fitting one -- and QTIP's trellis attaches a *computed* hash to
each state, so there is no codebook to fit either. We take the quantizer column
and decline the rounding column, and the zero-data constraint survives intact:
no calibration corpus, no Hessian, no gradients, at any point below.

What changes versus the published pipeline is exactly one line of the recursion:

    scalar   u -> nearest apex Lloyd-Max level, coordinate-wise   (published)
    e8       u -> nearest point of a finite E8 lattice, 8 dims at a time
    tcq      u -> Viterbi over a bitshift trellis with computed codes

Everything else -- fresh independent rotation per pass, per-group L2
normalisation, reconstruct-and-peel -- is identical, which is what Proposition 2
being quantiser-agnostic buys us (machine-checked as
`integral_norm_resid_sq_vectorStage`).

    uv run python experiments/progressive_vector_stage_llm.py --models Phi-1.5
    uv run python experiments/progressive_vector_stage_llm.py --models Phi-1.5 --stages scalar,e8
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matched_budget_baseline import MODELS  # noqa: E402
from rerun_apexquant_llm import DEV, perplexity  # noqa: E402
from vector_stage_rho import e8_codebook, make_tcq_stage, sample_sphere  # noqa: E402

from apexquant.codebook.apex_beta_levels import apex_beta_levels  # noqa: E402
from apexquant.quantize.quantize_to_indices import quantize_to_indices  # noqa: E402
from apexquant.rotation.inverse_random_rotation import (  # noqa: E402
    inverse_random_rotation,
)
from apexquant.rotation.random_rotation import random_rotation  # noqa: E402

GS = 512
SEED = 42


# ---------------------------------------------------------------- stages
def make_scalar_stage(bits: int, gs: int):
    """The published stage: nearest apex Lloyd-Max level, coordinate-wise."""
    levels = apex_beta_levels(bits, gs, device=DEV)

    def Q(u: torch.Tensor) -> torch.Tensor:
        return levels[quantize_to_indices(u, levels).long()]

    Q.name = f"scalar-b{bits}"
    return Q


def make_e8_stage_fast(bits: int, d: int, chunk_blocks: int = 8192,
                       n_probe: int = 20000, seed: int = 0):
    """Finite E8 lattice at `bits` per coordinate, one globally tuned scale.

    Same construction and same golden-section scale search as
    `vector_stage_rho.make_e8_stage`; only the chunk size differs, because that
    routine's 256-block chunks make a whole-model sweep Python-bound. The scale
    is fitted to the KNOWN spherical marginal via probe samples from the sphere,
    not to any weights, so this stays data-free.
    """
    n_cw = 1 << (8 * bits)
    cb = e8_codebook(n_cw)                                     # (n_cw, 8)
    cb_sq = cb.pow(2).sum(1)

    def _q(blocks: torch.Tensor, scale: float) -> torch.Tensor:
        out = torch.empty_like(blocks)
        for i in range(0, blocks.shape[0], chunk_blocks):
            b = blocks[i : i + chunk_blocks] / scale
            idx = (cb_sq.unsqueeze(0) - 2.0 * (b @ cb.T)).argmin(dim=1)
            out[i : i + chunk_blocks] = cb[idx] * scale
        return out

    g = torch.Generator(device=DEV).manual_seed(seed)
    probe = sample_sphere(n_probe // (d // 8) + 1, d, g).reshape(-1, 8)
    lo, hi = 1e-3 / math.sqrt(d), 4.0 / math.sqrt(d)
    for _ in range(28):
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        d1 = (probe - _q(probe, m1)).pow(2).sum().item()
        d2 = (probe - _q(probe, m2)).pow(2).sum().item()
        hi, lo = (m2, lo) if d1 < d2 else (hi, m1)
    scale = 0.5 * (lo + hi)

    def Q(u: torch.Tensor) -> torch.Tensor:
        return _q(u.reshape(-1, 8), scale).reshape(u.shape)

    Q.name = f"e8-b{bits}"
    Q.scale = scale
    return Q


def stage_rho_sq(Q, d: int, n: int = 4096, seed: int = 1) -> float:
    """rho^2 = E||u - Q(u)||^2 on uniform directions: the tab:stages quantity.

    Reported alongside every end-to-end run so the perplexity can be read
    against the distortion the theory predicts from.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    u = sample_sphere(n, d, g)
    return (u - Q(u)).pow(2).sum(1).mean().item()


# ------------------------------------------------------- the recursion
def reconstruct_stage(weight: torch.Tensor, Q, passes: int,
                      group_size: int = GS, seed: int = SEED) -> torch.Tensor:
    """Progressive quantisation with an arbitrary stage quantiser `Q`.

    Line-for-line the published `matched_budget_baseline.reconstruct`, except
    that the stage is `Q` instead of a coordinate-wise apex level assignment.
    Fresh independent rotation per pass, per-group L2 normalisation onto the
    sphere, reconstruct and peel.
    """
    out_f, in_f = weight.shape
    gs = min(group_size, 1 << (in_f.bit_length() - 1)) if in_f < group_size else group_size
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
        u = (rotated / scale).reshape(-1, gs)
        recon_rot = scale * Q(u).reshape(out_f, ngroups, gs)
        back = inverse_random_rotation(recon_rot, s)
        acc = acc + back
        residual = residual - back
    dense = acc.reshape(out_f, ngroups * gs)
    return dense[:, :in_f].contiguous()


def quantise_(model, Q, passes: int) -> None:
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            if mod.in_features < 32:
                continue
            with torch.no_grad():
                w = mod.weight.data
                mod.weight.data = reconstruct_stage(
                    w.to(DEV), Q, passes).to(w.dtype)


def eff_bits(bits: int, passes: int, gs: int = GS) -> float:
    return passes * (bits + 16.0 / gs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Phi-1.5")
    ap.add_argument("--stages", default="scalar,e8")
    ap.add_argument("--passes", default="1,2,3")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--trellis-bits", type=int, default=12)
    ap.add_argument("--checkpoint", default=None,
                    help="JSON file of completed rows. Written after every row "
                         "and read on startup, so a run killed partway resumes "
                         "instead of repeating hours of Viterbi. Rows are keyed "
                         "by (model, stage name, passes), and the stage name "
                         "carries its own configuration (e.g. tcq-L12-b2), so a "
                         "run with different settings will not reuse them.")
    args = ap.parse_args()

    ckpt: dict[str, float] = {}
    if args.checkpoint and os.path.exists(args.checkpoint):
        with open(args.checkpoint) as fh:
            ckpt = json.load(fh)
        print(f"# resuming from {args.checkpoint}: {len(ckpt)} rows already done\n")

    def remember(key: str, value: float) -> None:
        if not args.checkpoint:
            return
        ckpt[key] = value
        tmp = args.checkpoint + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(ckpt, fh, indent=1)
        os.replace(tmp, args.checkpoint)   # atomic: never a half-written file

    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    passes = [int(p) for p in args.passes.split(",")]

    print("# Progressive refinement with a vector stage quantiser, end to end\n")
    print(f"# b={args.bits}/coordinate, group_size={GS}, WikiText-2, decoder "
          "Linear weights. Every stage below is DATA-FREE: fixed lattice or "
          "computed code, one scale tuned to the known spherical marginal.\n")

    stages = {}
    for sname in args.stages.split(","):
        if sname == "scalar":
            Q = make_scalar_stage(args.bits, GS)
        elif sname == "e8":
            Q = make_e8_stage_fast(args.bits, GS)
        elif sname == "e8p":
            # QuIP#'s padded construction: same 2^16 codewords, 1 KiB table,
            # and an exact structured search (verified in e8p_stage.verify).
            from e8p_stage import E8PStage
            Q = E8PStage(GS)
        elif sname == "tcq":
            Q = make_tcq_stage(args.bits, GS, trellis_bits=args.trellis_bits)
        else:
            raise SystemExit(f"unknown stage {sname}")
        r2 = stage_rho_sq(Q, GS)
        stages[sname] = (Q, r2)
        print(f"# stage {Q.name:16s} rho^2 = {r2:.4f}  rho = {math.sqrt(r2):.4f}",
              flush=True)
    print()

    for mname in args.models.split(","):
        repo = MODELS[mname]
        tok = AutoTokenizer.from_pretrained(repo)
        enc = tok(text, return_tensors="pt").input_ids
        base = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.float16).to(DEV).eval()
        fp_key = f"{mname}|fp16"
        if fp_key in ckpt:
            ppl_fp = ckpt[fp_key]
        else:
            ppl_fp = perplexity(base, (enc,))
            remember(fp_key, ppl_fp)
        sd = {k: v.detach().to("cpu", copy=True)
              for k, v in base.state_dict().items()}
        print(f"## {mname}\n")
        print("| stage | rho^2 | passes | eff. bits | predicted err(P) | perplexity |")
        print("|---|---:|---:|---:|---:|---:|")
        print(f"| FP16 | - | - | 16.00 | - | {ppl_fp:.3f} |")
        for sname, (Q, r2) in stages.items():
            for P in passes:
                key = f"{mname}|{Q.name}|{P}"
                if key in ckpt:
                    val = ckpt[key]
                else:
                    base.load_state_dict(sd)
                    quantise_(base, Q, P)
                    val = perplexity(base, (enc,))
                    remember(key, val)
                pstr = f"{val:.3f}" if val < 1000 else f"{val:.3e}"
                print(f"| {sname} | {r2:.4f} | {P} | {eff_bits(args.bits, P):.2f} "
                      f"| {math.sqrt(r2) ** P:.4f} | {pstr} |", flush=True)
        del base, sd
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
