"""One rung of the E1 ladder: encode, WikiText-2 KL, harness accuracy, storage."""

from __future__ import annotations

import gc
import json
import os
import shutil
import time

import torch
from transformers import AutoTokenizer

from apexquant.vq.trellis_codec import TrellisCodec
from clinical.load_causal_lm import load_causal_lm
from clinical.quantize_apex import quantize_apex
from clinical.quantize_apex_l0 import quantize_apex_l0
from clinical.quantize_apex_sinkhorn import quantize_apex_sinkhorn
from clinical.quantize_dash import quantize_dash
from clinical.quantize_hqq import quantize_hqq
from clinical.quantize_sinq import quantize_sinq
from clinical.restore_architecture import restore_architecture
from clinical.run_liftquant import run_liftquant
from clinical.run_llmcompressor import run_llmcompressor
from clinical.run_lm_eval import run_lm_eval
from clinical.token_kl import token_kl

from .model_bits import model_bits
from .model_dir import model_dir
from .quantize_residual_rtn import quantize_residual_rtn
from .scalar_stage import scalar_stage
from .settings import GROUP, LQ_NETS, TASKS, TRELLIS_BITS


def step_rung(work: str, model: str, arm: str, level, group: int, keep: bool = False) -> None:
    """Write ``results/{tag}.json``; finished rungs are skipped."""
    tag = f"{model}_{arm}_{level}_g{group}"
    dest = os.path.join(work, "results", f"{tag}.json")
    if os.path.exists(dest):
        print(f"{tag}: done already", flush=True)
        return
    path = model_dir(model)
    staging = os.path.join(work, "staging", tag)
    windows = os.path.join(work, "windows", model)
    calib = os.path.join(windows, "calib_c4.pt")
    external = arm in ("gptq", "awq", "liftquant")
    weights, start = None, time.perf_counter()
    if arm in ("gptq", "awq"):
        weights = run_llmcompressor(path, arm, level, group, calib, staging)
    elif arm == "liftquant":
        weights = run_liftquant(path, LQ_NETS[model], level,
                                os.path.join(windows, "calib_c4_lq.pt"), staging)
    calibrated_s = time.perf_counter() - start
    candidate = load_causal_lm(path)
    devices = range(torch.cuda.device_count())
    resident = sum(torch.cuda.memory_allocated(d) for d in devices)
    for d in devices:
        torch.cuda.reset_peak_memory_stats(d)
    start = time.perf_counter()
    trellis = lambda: TrellisCodec(GROUP, bits=2, trellis_bits=TRELLIS_BITS, device="cuda")
    if arm == "apex":
        weights = quantize_apex(candidate, [trellis()] * level, seed=0)
    elif arm == "apex_l0":
        weights = quantize_apex_l0(candidate, [trellis()] * level, seed=0)
    elif arm == "apex_sinkhorn":
        weights = quantize_apex_sinkhorn(candidate, [trellis()] * level, seed=0)
    elif arm == "mixed":
        weights = quantize_apex(candidate, [scalar_stage(4, GROUP, "cuda")] + [trellis()] * (level - 1),
                                seed=0)
    elif arm == "residual_rtn":
        weights = quantize_residual_rtn(candidate, level, group)
    elif arm == "dash":
        weights = quantize_dash(candidate, level, seed=0)
    elif arm == "hqq":
        weights = quantize_hqq(candidate, level, group)
    elif arm == "sinq":
        weights = quantize_sinq(candidate, level, group)
    elif not external:
        raise ValueError(f"unknown arm {arm!r}")
    torch.cuda.synchronize()
    encode_s = calibrated_s + time.perf_counter() - start
    peak = (None if external else
            sum(torch.cuda.max_memory_allocated(d) for d in devices) - resident)

    params = dict(candidate.named_parameters())
    missing = sorted(set(weights) - set(params))
    if missing:
        raise KeyError(f"quantizer returned unknown parameters {missing[:3]}")
    for name, w in weights.items():
        params[name].data.copy_(w)
    del weights, params
    bits = model_bits(candidate, arm, level, group)
    per_pass = 2 + 16 / GROUP
    expected = (level * per_pass if arm == "apex"
                else per_pass + 2 + (level - 1) * per_pass if arm == "mixed" else None)
    if expected is not None and abs(bits["quantized"] - expected) > 1e-9:
        raise RuntimeError(f"{tag}: {bits['quantized']} bits per quantized weight, expected {expected}")
    reference = load_causal_lm(path)
    kl = token_kl(reference, candidate, torch.load(os.path.join(windows, "wikitext.pt")))
    del reference

    candidate.save_pretrained(staging, safe_serialization=True)
    AutoTokenizer.from_pretrained(path).save_pretrained(staging)
    restore_architecture(path, staging)
    del candidate
    gc.collect()
    torch.cuda.empty_cache()
    # A 70B staging copy is 141 GB; a failed harness run must not leave it behind.
    try:
        accuracy = run_lm_eval(staging, TASKS, os.path.join(work, "lm_eval", tag),
                               tensor_parallel=len(devices))
    finally:
        if not keep:
            shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump({"model": model, "arm": arm, "level": level, "group": group,
               "bits_per_weight": bits["whole_model"],
               "bits_per_quantized_weight": bits["quantized"], "encode_seconds": encode_s,
               "encode_peak_gpu_bytes_above_model": peak, "kl": kl, "accuracy": accuracy},
              open(dest, "w"), indent=1)
