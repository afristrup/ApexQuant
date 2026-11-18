"""Fixed protocol constants for E3, the clinical study.

``E3_SMOKE=1`` swaps in TinyLlama (base and chat) and tiny sizes, to exercise every
step on one workstation GPU; nothing it produces is a result. Its widths divide by
128, as auto-round's W2A16 kernels need.
"""

import os

SMOKE = os.environ.get("E3_SMOKE") == "1"

BASE = "meta-llama/Llama-3.1-8B-Instruct"
BASE_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
FINETUNE = "TsinghuaC3I/Llama-3.1-8B-UltraMedical"
FINETUNE_REVISION = "c982b9e34f7cf396f41627e7d851629f9e4e5792"

PUBMED = "MedRAG/pubmed"
PUBMED_REVISION = "33da3593d5756bc04c8909f170003c0b14197957"
PUBMED_SHARDS = [f"chunk/pubmed23n{i:04d}.jsonl" for i in range(900, 904)]
PUBMED_SPLITS = {"oracle": range(0, 6), "dev": range(6, 8), "eval": range(8, 10)}

SEQ_LEN = 512 if SMOKE else 2048
EVAL_WINDOWS = 4 if SMOKE else 128
CALIB_WINDOWS = 8 if SMOKE else 128
CALIB_DRAWS = 2 if SMOKE else 5
SAMPLE_TOKENS = 256 if SMOKE else 4096
EVAL_LIMIT = 20 if SMOKE else None
GPU_UTIL = 0.2 if SMOKE else 0.85

GROUP = 512
LLMC_GROUP = 128
LQ_SAMPLES = 8 if SMOKE else 4096
LQ_EPOCHS = 1 if SMOKE else 2
LQ_WBITS = 2
# LiftQuant's exhaustive 2^24-codeword search takes about a day per 8B arm at 24to8,
# so its arms are left out of the grids until a faster search is settled on.
RUN_LIFTQUANT = False
# Plain prompts, the harness default. A chat-template run did not rescue the medical
# fine-tune (still below the base model) and cut the base model's MedQA from 0.62 to
# 0.34, so it is kept only as an option; its outputs live in separate *_chat folders.
CHAT_TEMPLATE = False
REFERENCES = "references_chat" if CHAT_TEMPLATE else "references"
RESULTS = "results_chat" if CHAT_TEMPLATE else "results"
LM_EVAL = "lm_eval_chat" if CHAT_TEMPLATE else "lm_eval"
# The fine-tune variant's arms wait until its reference beats the base model.
ARM_VARIANTS = ("controlled", "constrained")
LQ_NET = "Llama-2-smoke" if SMOKE else "Llama-3.1-8B-Instruct"
TRELLIS_BITS = 14
STAGE_BITS = 2
APEX_SEEDS = 1 if SMOKE else 5

CONTROLLED_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
CONTROLLED_MODULE = "mlp.down_proj"
CONTROLLED_RANK = 8
CONTROLLED_KL = 0.05
LORA_SPAN = 32

CLINICAL_TASKS = ("medqa_4options", "pubmedqa_labeled", "medmcqa")
GENERIC_TASKS = ("mmlu", "arc_challenge", "hellaswag", "winogrande")
KL_SETS = ("pubmed", "wikitext", "code", "danish")

if SMOKE:
    BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    BASE_REVISION = "59f6f375b26bde864a6ca194a9a3044570490064"
    FINETUNE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    FINETUNE_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    CONTROLLED_LAYERS = (3, 7, 11, 15, 19)
    PUBMED_SHARDS = PUBMED_SHARDS[:1]
    GENERIC_TASKS = ("arc_challenge",)
