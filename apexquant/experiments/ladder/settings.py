"""Fixed protocol constants for E1, the rate ladder at scale.

``E1_SMOKE=1`` swaps in TinyLlama-1.1B and tiny sizes for a one-GPU check; its
layer widths, like Llama-3.1's, divide by 128 as auto-round's W2A16 kernels need.
"""

import os

SMOKE = os.environ.get("E1_SMOKE") == "1"

MODELS = {
    "8b": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
    "70b": ("meta-llama/Llama-3.1-70B-Instruct", "1605565b47bb9346c5515c34102e054115b4f98b"),
}
MOE = {
    "phimoe": ("microsoft/Phi-tiny-MoE-instruct", "2fe50e88d0e2a5a132563815686ea0dcc8e252b5"),
    "qwen35moe": ("Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754"),
}
MODELS.update(MOE)
MOE_GROUP = 64

SEQ_LEN = 512 if SMOKE else 2048
KL_WINDOWS = 4 if SMOKE else 128
CALIB_WINDOWS = 8 if SMOKE else 128

GROUP = 512
TRELLIS_BITS = 14
ABLATION_GROUP = 128
LQ_NETS = {"8b": "Llama-3.1-8B-Instruct", "70b": "Llama-3.1-70B-Instruct"}
LQ_RATIOS = ("16to8", "20to8", "24to8")
LLMC_GROUP = 128
WEIGHT_ONLY_GROUPS = (64, 512)

TASKS = ("mmlu", "hellaswag", "arc_challenge", "winogrande", "gsm8k")

if SMOKE:
    MOE = {"phimoe": MOE["phimoe"]}
    MODELS = {"8b": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "fe8a4ea1ffedaf415f4da2f062534de366a451e6"),
              **MOE}
    TASKS = ("arc_challenge", "winogrande")
    LQ_NETS = {"8b": "Llama-2-smoke"}
