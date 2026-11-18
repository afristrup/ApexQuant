"""Measure vLLM decode throughput over context lengths."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", nargs="+", type=int,
                    default=[512, 2048, 4096, 8192, 16384])
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    args = ap.parse_args()

    import apexquant.vllm as apex_vllm
    apex_vllm.register()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        dtype="float16",
        max_model_len=max(args.contexts) + args.new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization="apexquant",
        kv_cache_dtype=os.environ.get("APEXQUANT_BENCH_KV_DTYPE", "apexquant_e8p2"),
        enforce_eager=os.environ.get("APEXQUANT_BENCH_ENFORCE_EAGER", "1") == "1",
        compilation_config=({"compile_sizes": [1], "cudagraph_capture_sizes": [1], "max_cudagraph_capture_size": 1} if os.environ.get("APEXQUANT_BENCH_SINGLE_GRAPH") == "1" else None),
        trust_remote_code=True,
    )
    rows = []
    for context in args.contexts:
        seed = tokenizer.encode(
            "Benchmark the following context. ", add_special_tokens=False
        )
        ids = (seed * ((context + len(seed) - 1) // len(seed)))[:context]
        prompt = tokenizer.decode(ids, skip_special_tokens=False)
        params = SamplingParams(temperature=0.0, max_tokens=args.new_tokens,
                                ignore_eos=True)
        for _ in range(2):
            llm.generate([prompt], params, use_tqdm=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        generated = sum(len(o.outputs[0].token_ids) for o in outputs)
        rows.append({
            "context_tokens": context,
            "generated_tokens": generated,
            "elapsed_seconds": elapsed,
            "decode_tokens_per_second": generated / elapsed,
        })
        print(json.dumps(rows[-1]), flush=True)
    result = {
        "model": args.model,
        "weights": "Trellis P=2, L=14 runtime vLLM path",
        "kv": "E8P P=2 custom vLLM backend",
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
