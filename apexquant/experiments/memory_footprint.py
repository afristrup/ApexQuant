"""Execution-time weight-memory footprint: fused 2-bit vs fp16 vs ApexQuant.

The distinction that matters is STORED vs EXECUTED bytes. ApexQuant's inference
path (its Algorithm 2) reconstructs the floating-point weight, so its 4-bit
model is small on disk but occupies fp16 bytes in accelerator memory at
inference. The fused kernels execute directly from the packed payload, so
stored == executed.

Per model we count decoder/transformer Linear weights with fan-in >= 512 (the
quantised set, matching every benchmark in the paper) and everything else
(embeddings, lm_head, norms, small layers) as kept dense in fp16. LLM shapes
come from meta-device instantiation (no weights loaded); Clay's numbers are the
measured FusionReport bytes from the real checkpoint fusion (7.76x on the
transformer linears at 2-bit x2).

    HF_HOME=... uv run python experiments/memory_footprint.py
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

GS, BITS = 512, 2
MODELS = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "Phi-1.5": "microsoft/phi-1_5",
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
}


def quantised_bytes(params: int, passes: int) -> int:
    """Packed indices + fp16-equivalent scale bytes for the quantised layers."""
    return params * BITS * passes // 8 + params // GS * 2 * passes


def split_params(repo: str) -> tuple[int, int]:
    """(quantised linear params, kept-dense params) from a meta-device model."""
    cfg = AutoConfig.from_pretrained(repo)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    q = d = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "layers." in name and mod.weight.shape[1] >= 512:
            q += mod.weight.numel()
    total = sum(p.numel() for p in model.parameters())
    d = total - q
    return q, d


def gb(x: int) -> float:
    return x / 1e9


def main():
    print("# Execution-time weight memory (GB): fused progressive 2-bit vs fp16\n")
    print("# ApexQuant executes at fp16 (Algorithm 2 reconstructs); fused executes packed.")
    print(f"# Quantised set: decoder linears fan-in >= {GS}; rest dense fp16. gs={GS}.\n")
    print("| model | fp16 exec | ours x1 | ours x2 | ours x3 | x2 reduction |")
    print("|-------|----------:|--------:|--------:|--------:|-------------:|")
    for mname, repo in MODELS.items():
        q, d = split_params(repo)
        fp16 = (q + d) * 2
        row = [quantised_bytes(q, p) + d * 2 for p in (1, 2, 3)]
        print(f"| {mname} | {gb(fp16):>8.2f} | {gb(row[0]):>6.2f} | {gb(row[1]):>6.2f} "
              f"| {gb(row[2]):>6.2f} | {fp16/row[1]:>10.2f}x |", flush=True)

    # Clay v1.5: measured FusionReport numbers from the real checkpoint
    # (experiments/calibration_frontier_clay.py): transformer linears 302.0M
    # params, fp32 checkpoint 1208 MB -> x1 77.9 MB / x2 155.7 MB / x3 233.6 MB
    # stored+executed (compression 15.51x / 7.76x / 5.17x vs fp32); the
    # hypernetwork patch embedding (30.7M params) stays dense.
    q_clay, d_clay = 302_000_000, 30_700_000
    fp32 = (q_clay + d_clay) * 4
    print("\n# Clay v1.5 (vs its fp32 checkpoint; transformer linears measured "
          "via FusionReport):\n")
    print("| model | fp32 exec | ours x1 | ours x2 | ours x3 | x2 reduction |")
    print("|-------|----------:|--------:|--------:|--------:|-------------:|")
    row = [q_clay * 4 / r + d_clay * 4 for r in (15.51, 7.76, 5.17)]
    print(f"| Clay v1.5 | {gb(fp32):>8.2f} | {gb(row[0]):>6.2f} | {gb(row[1]):>6.2f} "
          f"| {gb(row[2]):>6.2f} | {fp32/row[1]:>10.2f}x |")


if __name__ == "__main__":
    main()
