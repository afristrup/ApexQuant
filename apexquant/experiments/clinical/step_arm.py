"""One quantized arm of one variant: per-layer statistics, token KL and harness accuracy."""

from __future__ import annotations

import gc
import json
import os
import shutil

import torch
from transformers import AutoTokenizer

from apexquant.vq.trellis_codec import TrellisCodec

from .checkpoint_dir import checkpoint_dir
from .input_key import input_key
from .layer_metrics import layer_metrics
from .load_causal_lm import load_causal_lm
from .quantize_apex import quantize_apex
from .quantize_apex_sinkhorn import quantize_apex_sinkhorn
from .quantize_dash import quantize_dash
from .quantize_hqq import quantize_hqq
from .quantize_sinq import quantize_sinq
from .read_tensor import read_tensor
from .restore_architecture import restore_architecture
from .run_liftquant import run_liftquant
from .run_llmcompressor import run_llmcompressor
from .run_lm_eval import run_lm_eval
from .settings import (CHAT_TEMPLATE, CLINICAL_TASKS, GENERIC_TASKS, GROUP, KL_SETS, LLMC_GROUP,
                       LM_EVAL, LQ_NET, RESULTS, STAGE_BITS, TRELLIS_BITS)
from .stage_beta import stage_beta
from .target_linears import target_linears
from .token_kl import token_kl

GROUPS = {"hqq": 64, "sinq": 64, "gptq": LLMC_GROUP, "awq": LLMC_GROUP}


def step_arm(work: str, variant: str, arm: str, level, calib: str, seed: int,
             keep: bool = False) -> None:
    """Write ``results/{tag}.json``; ``level`` is passes for apex, the lifting ratio for
    LiftQuant, and bits otherwise."""
    if (calib == "none") != (arm in ("apex", "apex_sinkhorn", "hqq", "sinq", "dash")):
        raise ValueError(f"{arm} with calibration {calib!r}")
    tag = f"{variant}_{arm}_{level}_{calib}_s{seed}"
    dest = os.path.join(work, RESULTS, f"{tag}.json")
    if os.path.exists(dest):
        print(f"{tag}: done already", flush=True)
        return
    path, base_path = checkpoint_dir(variant, work), checkpoint_dir("base", work)
    reference, candidate = load_causal_lm(path), load_causal_lm(path)
    shrink, stage = None, None
    if arm == "apex":
        stage = TrellisCodec(GROUP, bits=STAGE_BITS, trellis_bits=TRELLIS_BITS, device="cuda")
        shrink = stage_beta(stage, GROUP, 4096, "cuda") ** level
        weights = quantize_apex(candidate, [stage] * level, seed)
    elif arm == "apex_sinkhorn":
        stage = TrellisCodec(GROUP, bits=STAGE_BITS, trellis_bits=TRELLIS_BITS, device="cuda")
        weights = quantize_apex_sinkhorn(candidate, [stage] * level, seed)
    elif arm == "hqq":
        weights = quantize_hqq(candidate, level, GROUPS[arm])
    elif arm == "sinq":
        weights = quantize_sinq(candidate, level, GROUPS[arm])
    elif arm == "dash":
        weights = quantize_dash(candidate, level, seed)
    elif arm == "liftquant":
        weights = run_liftquant(path, LQ_NET, level, os.path.join(work, "windows", f"calib_{calib}.pt"),
                                os.path.join(work, "staging", tag))
    elif arm in ("gptq", "awq"):
        weights = run_llmcompressor(path, arm, level, GROUPS[arm],
                                    os.path.join(work, "windows", f"calib_{calib}.pt"),
                                    os.path.join(work, "staging", tag))
    else:
        raise ValueError(f"unknown arm {arm!r}")
    params = dict(candidate.named_parameters())
    missing = sorted(set(weights) - set(params))
    if missing:
        raise KeyError(f"quantizer returned unknown parameters {missing[:3]}")
    for name, w in weights.items():
        params[name].data.copy_(w)

    moments = os.path.join(work, "moments", variant)
    layers = {}
    for name in target_linears(candidate):
        key = input_key(name)
        if not os.path.exists(os.path.join(moments, "generic", f"{key}.pt")):
            continue
        layers[name] = layer_metrics(
            read_tensor(base_path, f"{name}.weight"), read_tensor(path, f"{name}.weight"),
            params[f"{name}.weight"].data.cpu(),
            torch.load(os.path.join(moments, "generic", f"{key}.pt")),
            torch.load(os.path.join(moments, "pubmed_eval", f"{key}.pt")), shrink)
    kl = {name: token_kl(reference, candidate,
                         torch.load(os.path.join(work, "windows", f"eval_{name}.pt")))
          for name in KL_SETS}
    del reference

    staging = os.path.join(work, "staging", tag)
    candidate.save_pretrained(staging, safe_serialization=True)
    AutoTokenizer.from_pretrained(path).save_pretrained(staging)
    restore_architecture(path, staging)
    del candidate, params, weights, stage
    gc.collect()
    torch.cuda.empty_cache()
    tasks = CLINICAL_TASKS + (GENERIC_TASKS if variant == "ft" else ())
    try:
        accuracy = run_lm_eval(staging, tasks, os.path.join(work, LM_EVAL, tag),
                               chat=CHAT_TEMPLATE)
    finally:
        if not keep:
            shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump({"variant": variant, "arm": arm, "level": level, "calib": calib, "seed": seed,
               "shrink": shrink, "kl": kl, "accuracy": accuracy, "layers": layers},
              open(dest, "w"), indent=1)
