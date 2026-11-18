"""Harness accuracy of an in-memory model, through the Hugging Face backend."""

from __future__ import annotations

import json
import os

from .settings import EVAL_LIMIT, TASKS


def harness_accuracy(model, tokenizer_path: str, out_dir: str) -> dict[str, float]:
    """Accuracy per task; the allocated experts are custom modules, so vLLM cannot serve them."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    lm = HFLM(pretrained=model, tokenizer=AutoTokenizer.from_pretrained(tokenizer_path), batch_size=8)
    result = simple_evaluate(model=lm, tasks=list(TASKS), limit=EVAL_LIMIT, random_seed=0,
                             numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(result["results"], open(os.path.join(out_dir, "results.json"), "w"), indent=1, default=str)
    return {t: result["results"][t]["acc,none"] for t in TASKS}
