"""``python -m clinical.llmc_quantize``: GPTQ or AWQ in the llm-compressor environment.

llm-compressor and vLLM pin different compressed-tensors releases, so this runs
under its own interpreter and hands the baked weights back as one safetensors file.
"""

from __future__ import annotations

import argparse

import torch
from safetensors.torch import save_file

from .quantize_llmcompressor import quantize_llmcompressor


def main() -> None:
    """Quantize ``--path`` on ``--windows`` and write every parameter to ``--out``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--method", choices=("gptq", "awq"), required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    weights = quantize_llmcompressor(args.path, args.method, args.bits, args.group,
                                     torch.load(args.windows))
    save_file({k: v.contiguous().clone() for k, v in weights.items()}, args.out)


if __name__ == "__main__":
    main()
