"""Harness accuracy through vLLM, in a child process so its GPU memory is freed."""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

from .free_fraction import free_fraction
from .settings import EVAL_LIMIT, GPU_UTIL


def run_lm_eval(model_dir: str, tasks: tuple[str, ...], out_dir: str,
                tensor_parallel: int = 1, chat: bool = False) -> dict[str, float]:
    """Accuracy per task, with per-item samples logged under ``out_dir`` for bootstraps.

    ``chat`` renders every prompt through the model's chat template. With
    ``APEX_GPU_SHARE=1`` two tasks share the GPU, so vLLM takes 42% of it, and a
    run that fails (vLLM's start-up memory check trips when the neighbour frees
    memory mid-profile) is retried up to three times.
    ``$APEX_ATTENTION_BACKEND`` overrides vLLM's attention backend; an image without
    the CUDA toolkit, so FlashInfer's JIT-built Blackwell kernels are replaced
    there by Triton's.
    """
    config = json.load(open(os.path.join(model_dir, "config.json")))
    context = min(4096, config.get("max_position_embeddings", 4096))
    util = 0.42 if os.environ.get("APEX_GPU_SHARE") == "1" else GPU_UTIL
    # The caller can still hold memory after freeing its models; vLLM refuses to
    # start when its target exceeds what is free, so ask for no more than that.
    util = round(min(util, free_fraction() - 0.03), 3)
    args = ",".join([f"pretrained={model_dir}", "dtype=bfloat16", f"max_model_len={context}",
                     f"gpu_memory_utilization={util}", "seed=0",
                     f"tensor_parallel_size={tensor_parallel}"])
    if os.environ.get("APEX_ATTENTION_BACKEND"):
        args += f",attention_backend={os.environ['APEX_ATTENTION_BACKEND']}"
    cmd = [sys.executable, "-m", "lm_eval", "--model", "vllm", "--model_args", args,
           "--tasks", ",".join(tasks), "--batch_size", "auto", "--seed", "0",
           "--output_path", out_dir, "--log_samples",
           "--include_path", os.path.join(os.path.dirname(__file__), "tasks")]
    if EVAL_LIMIT:
        cmd += ["--limit", str(EVAL_LIMIT)]
    if chat:
        cmd += ["--apply_chat_template"]
    # FlashInfer's tensor-parallel all-reduce JIT-compiles with nvcc, which an
    # image without the CUDA toolkit lacks; vLLM's own all-reduce needs no compiler.
    env = dict(os.environ, VLLM_ALLREDUCE_USE_FLASHINFER="0")
    for attempt in range(4):
        if subprocess.run(cmd, env=env).returncode == 0:
            break
        if attempt == 3:
            raise RuntimeError(f"lm_eval failed four times on {model_dir}")
        time.sleep(60 * (attempt + 1))
    files = sorted(glob.glob(os.path.join(out_dir, "**", "results_*.json"), recursive=True),
                   key=os.path.getmtime)
    if not files:
        raise RuntimeError(f"lm_eval wrote no results under {out_dir}")
    results = json.load(open(files[-1]))["results"]
    return {task: results[task].get("acc,none", results[task].get("exact_match,strict-match"))
            for task in tasks}

