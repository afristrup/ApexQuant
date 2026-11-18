"""Llama-3.1-70B-Instruct served through vLLM: every stage at one to three passes.

``SERVING_SMOKE=1`` swaps in TinyLlama-1.1B-Chat so the whole pipeline can be
checked in minutes; ``SERVING_MODEL=8b`` runs the same arms on Llama-3.1-8B-Instruct.
"""

from __future__ import annotations

import os

MODEL = "meta-llama/Llama-3.1-70B-Instruct"
REVISION = "1605565b47bb9346c5515c34102e054115b4f98b"
if os.environ.get("SERVING_MODEL") == "8b":
    MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
if os.environ.get("SERVING_SMOKE") == "1":
    MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"

# name: (weight stage or None for FP16, passes, vLLM kv_cache_dtype)
ARMS = {"fp16_fp16": (None, 0, "auto")}
ARMS.update({f"{stage}_p{p}_fp16": (stage, p, "auto")
             for stage in ("scalar", "e8p", "trellis") for p in (1, 2, 3)})
ARMS["trellis_p2_e8p2"] = ("trellis", 2, "apexquant_e8p2")
# Trellis on SINQ's data-free Sinkhorn diagonals; arms named *sinkhorn* run with
# APEXQUANT_SINKHORN_ITERS=SINKHORN_ITERS, folded into the scales and the rotation.
ARMS.update({f"trellis_sinkhorn_p{p}_fp16": ("trellis", p, "auto") for p in (1, 2, 3)})
ARMS["trellis_sinkhorn_p2_e8p2"] = ("trellis", 2, "apexquant_e8p2")
# Localization ablations at trellis P=2: each arm keeps the layers matching its
# regex dense (APEXQUANT_KEEP_DENSE) and quantizes the rest.
LAYER = r"model\.layers\."
KEEP_DENSE = {
    "trellis_keep1_p2_fp16": LAYER + r"0\.mlp\.down_proj",
    "trellis_keep4_p2_fp16": LAYER + r"[0-3]\.mlp\.down_proj",
    "trellis_attnonly_p2_fp16": LAYER + r"\d+\.mlp\..*",
    "trellis_mlponly_p2_fp16": LAYER + r"\d+\.self_attn\..*",
    "trellis_firsthalf_p2_fp16": LAYER + r"[4-7]\d\..*",
    "trellis_secondhalf_p2_fp16": LAYER + r"([0-9]|[1-3]\d)\..*",
    "trellis_keepL0_p2_fp16": LAYER + r"0\..*",
    "trellis_keepL01_p2_fp16": LAYER + r"[01]\..*",
}
KEEP_DENSE.update({
    "trellis_keepL0attn_p2_fp16": LAYER + r"0\.self_attn\..*",
    "trellis_keepL0qkv_p2_fp16": LAYER + r"0\.self_attn\.qkv_proj",
    "trellis_keepL0o_p2_fp16": LAYER + r"0\.self_attn\.o_proj",
})
ARMS.update({name: ("trellis", 2, "auto") for name in KEEP_DENSE})
# The fix: quantizing layer 0 collapses 70B at P<=2, so *_L0 arms keep the first
# decoder layer in BF16 (one of 80 layers, about 0.15 bits per weight).
for base in ("trellis_p1_fp16", "trellis_p2_fp16", "trellis_p3_fp16", "e8p_p1_fp16",
             "e8p_p2_fp16", "e8p_p3_fp16", "scalar_p2_fp16", "trellis_p2_e8p2"):
    ARMS[f"{base}_L0"] = ARMS[base]
    KEEP_DENSE[f"{base}_L0"] = LAYER + r"0\..*"
# Both fixes at once: layer 0 BF16 and weights beyond 4 sd kept exact.
for z, p in ((4, 1), (4, 2), (3, 1), (2, 1)):
    ARMS[f"trellis_out{z}_p{p}_fp16_L0"] = ("trellis", p, "auto")
    KEEP_DENSE[f"trellis_out{z}_p{p}_fp16_L0"] = LAYER + r"0\..*"
# Outlier ablations: arms named *out<Z>* keep weights beyond Z sd exact (sparse FP16).
ARMS.update({f"trellis_out{z}_p2_fp16": ("trellis", 2, "auto") for z in (4, 6, 10, 20)})
SINKHORN_ITERS = 10

CONTEXT = 2048
NEW_TOKENS = 256
MMLU_MAX_LEN = 4096
# Quantization noise pushes Llama-3.1-70B's largest activations past the FP16
# range (every quantized arm scored 0.2295, one letter always), so accuracy runs
# in the checkpoint's own bfloat16.
MMLU_DTYPE = "bfloat16"
if os.environ.get("SERVING_SMOKE") == "1":
    CONTEXT, NEW_TOKENS, MMLU_MAX_LEN = 1024, 128, 2048
BATCHES = (1, 16)
REPEATS = 3
GPU_UTIL = 0.92
