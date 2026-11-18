"""Memory and throughput of one arm, with the packed weights the arm would ship."""

from __future__ import annotations

import itertools
import json
import os
import time

from .settings import (ARMS, BATCHES, CONTEXT, GPU_UTIL, MODEL, NEW_TOKENS, REPEATS,
                       REVISION, SINKHORN_ITERS)


def measure_serving(name: str, out: str) -> None:
    """Weight bytes, KV capacity, load time, and prefill and decode tokens per second.

    Prompts are ``CONTEXT`` random token ids, so prefix caching (off anyway)
    and content cannot matter. Each batch size is timed ``REPEATS`` times after
    one warm-up, once with a single new token (prefill) and once with
    ``NEW_TOKENS`` (prefill plus decode); the fastest run of each is kept.
    """
    stage, passes, kv = ARMS[name]
    if os.environ.get("APEXQUANT_DENSE_RECON") == "1":
        raise RuntimeError("serving numbers need the packed weights; unset APEXQUANT_DENSE_RECON")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if stage is not None:
        os.environ.update(APEXQUANT_STAGE=stage, APEXQUANT_PASSES=str(passes))
    if "sinkhorn" in name:
        os.environ["APEXQUANT_SINKHORN_ITERS"] = str(SINKHORN_ITERS)
    import torch
    from vllm import LLM, SamplingParams

    start = time.perf_counter()
    llm = LLM(model=MODEL, revision=REVISION, dtype="float16",
              max_model_len=CONTEXT + NEW_TOKENS, gpu_memory_utilization=GPU_UTIL,
              kv_cache_dtype=kv, quantization="apexquant" if stage else None,
              enable_prefix_caching=False, max_num_seqs=max(BATCHES))
    load_s = time.perf_counter() - start
    weight_bytes = llm.apply_model(lambda m: sum(
        t.numel() * t.element_size() for t in itertools.chain(m.parameters(), m.buffers())))[0]
    cache = llm.llm_engine.vllm_config.cache_config
    generator = torch.Generator().manual_seed(0)
    rows = []
    for batch in BATCHES:
        prompts = [{"prompt_token_ids": torch.randint(1000, 30000, (CONTEXT,),
                                                      generator=generator).tolist()}
                   for _ in range(batch)]

        def fastest(tokens: int) -> float:
            params = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
            llm.generate(prompts, params, use_tqdm=False)
            times = []
            for _ in range(REPEATS):
                torch.cuda.synchronize()
                t = time.perf_counter()
                llm.generate(prompts, params, use_tqdm=False)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            return min(times)

        prefill, full = fastest(1), fastest(NEW_TOKENS)
        rows.append({"batch": batch, "prefill_s": prefill, "total_s": full,
                     "prefill_tok_s": batch * CONTEXT / prefill,
                     "decode_tok_s": batch * (NEW_TOKENS - 1) / (full - prefill),
                     "decode_ms_per_step": 1e3 * (full - prefill) / (NEW_TOKENS - 1)})
        print(name, rows[-1], flush=True)
    result = {"arm": name, "model": MODEL, "revision": REVISION, "stage": stage,
              "passes": passes, "kv_cache_dtype": kv, "gpu": torch.cuda.get_device_name(),
              "load_s": load_s, "weight_bytes": weight_bytes,
              "weight_gib": weight_bytes / 2 ** 30,
              "kv_cache_tokens": cache.num_gpu_blocks * cache.block_size,
              "gpu_memory_utilization": GPU_UTIL,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
              "context": CONTEXT, "new_tokens": NEW_TOKENS, "throughput": rows}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(result, open(out, "w"), indent=1)
