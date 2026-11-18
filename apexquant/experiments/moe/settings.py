"""Fixed protocol constants for E6, expert allocation.

``E6_SMOKE=1`` keeps Phi-tiny-MoE only, with tiny window counts; ``E6_SMOKE_MODEL``
points it at a local checkpoint of the same architecture instead.
"""

import os

SMOKE = os.environ.get("E6_SMOKE") == "1"

MODELS = {
    "phimoe": ("microsoft/Phi-tiny-MoE-instruct", "2fe50e88d0e2a5a132563815686ea0dcc8e252b5"),
    "qwen35moe": ("Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754"),
}

SEQ_LEN = 512 if SMOKE else 2048
KL_WINDOWS = 4 if SMOKE else 64
HELDOUT_WINDOWS = 4 if SMOKE else 64
USAGE_WINDOWS = 4 if SMOKE else 128
ENVELOPE_WINDOWS = 2 if SMOKE else 4
ENVELOPE_EVERY = 4

GROUP = 512
TRELLIS_BITS = 14
PMAX = 3
DENSE_PASSES = 2
AVERAGE_BITS = (3, 4, 5)
POLICIES = ("uniform", "usage", "oracle", "gate")

TASKS = ("mmlu", "arc_challenge", "hellaswag", "winogrande")
EVAL_LIMIT = 20 if SMOKE else None

if SMOKE:
    MODELS = {"phimoe": (os.environ.get("E6_SMOKE_MODEL", MODELS["phimoe"][0]),
                         None if "E6_SMOKE_MODEL" in os.environ else MODELS["phimoe"][1])}
    TASKS = ("arc_challenge",)
