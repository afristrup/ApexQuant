"""Run the GPTQ/AWQ arm under the llm-compressor interpreter and read its weights back."""

from __future__ import annotations

import os
import subprocess

import torch
from safetensors.torch import load_file


def run_llmcompressor(path: str, method: str, bits: int, group: int, windows: str,
                      scratch: str) -> dict[str, torch.Tensor]:
    """Baked parameters from ``clinical.llmc_quantize``, run with ``$LLMC_PYTHON``.

    The child sees only this package on its path, so nothing from the vLLM
    environment leaks into the llm-compressor one.
    """
    python = os.environ.get("LLMC_PYTHON")
    if not python:
        raise RuntimeError("set LLMC_PYTHON to the llm-compressor environment's python")
    out = os.path.join(scratch, f"{method}_{bits}.safetensors")
    os.makedirs(scratch, exist_ok=True)
    subprocess.run([python, "-m", "clinical.llmc_quantize", "--path", path, "--method", method,
                    "--bits", str(bits), "--group", str(group), "--windows", windows,
                    "--out", out], check=True,
                   env=dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(__file__))))
    weights = load_file(out)
    os.remove(out)
    return weights
