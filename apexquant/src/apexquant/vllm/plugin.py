"""Small public entry points for loading ApexQuant checkpoints with vLLM."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ApexVLLMManifest:
    """The checkpoint contract shared by the vLLM and custom-model paths."""

    stage: str = "scalar"
    bits: int = 2
    group_size: int = 512
    passes: int = 3
    seed: int = 42
    moe: bool = False
    kv: bool = False

    @classmethod
    def from_path(cls, model: str | Path) -> "ApexVLLMManifest":
        path = Path(model) / "apexquant_config.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing ApexQuant manifest: {path}")
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("apexquant_config.json must contain an object")
        manifest = cls(
            stage=str(data.get("stage", "scalar")),
            bits=int(data.get("bits", 2)),
            group_size=int(data.get("group_size", 512)),
            passes=int(data.get("passes", 3)),
            seed=int(data.get("seed", 42)),
            moe=bool(data.get("moe", False)),
            kv=bool(data.get("kv", False)),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.stage not in ("scalar", "e8", "trellis"):
            raise ValueError("stage must be scalar, e8, or trellis")
        if self.bits not in (2, 4, 8):
            raise ValueError("bits must be 2, 4, or 8")
        if self.group_size <= 0 or self.passes <= 0:
            raise ValueError("group_size and passes must be positive")

    def native_vllm_supported(self) -> bool:
        """Return whether vLLM can consume this manifest without model code."""
        return self.stage == "scalar" and not self.moe and not self.kv


def single_request_compilation_config() -> dict[str, Any]:
    """Return a CUDA-graph config specialized for one live request."""
    return {
        "compile_sizes": [1],
        "cudagraph_capture_sizes": [1],
        "max_cudagraph_capture_size": 1,
    }


def load(model: str | Path, **kwargs: Any):
    """Load a manifest-backed ApexQuant checkpoint through ``vllm.LLM``.

    Scalar linear artifacts work with every vLLM-supported HF architecture.
    E8, trellis, MoE, and trellis KV artifacts require the explicit adapters
    exported from this package because their tensor contracts are model-layer
    specific.
    """
    manifest = ApexVLLMManifest.from_path(model)
    if not manifest.native_vllm_supported():
        raise ValueError(
            "this manifest needs a custom model adapter; use make_moe or "
            "make_kv_cache for e8, trellis, MoE, or KV artifacts"
        )
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("install apexquant[vllm] before calling apexquant.vllm.load") from exc
    kwargs.setdefault("trust_remote_code", True)
    kwargs.setdefault("quantization", "apexquant")
    if os.environ.get("APEXQUANT_SINGLE_REQUEST") == "1":
        kwargs.setdefault("compilation_config", single_request_compilation_config())
    return LLM(model=str(model), **kwargs)
