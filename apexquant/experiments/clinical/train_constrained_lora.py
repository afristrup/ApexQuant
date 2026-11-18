"""A rank-r LoRA on clinical QA whose input projection lives in a fixed span."""

from __future__ import annotations

import random

import torch
from datasets import load_dataset

PROMPT = "Question: {q}\nChoices:\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"


def train_constrained_lora(model, tokenizer, spans: dict[str, torch.Tensor], rank: int,
                           steps: int, batch: int, lr: float, seed: int) -> dict[str, torch.Tensor]:
    """Train ``B R V`` per layer on MedMCQA train, ``V`` fixed, and return the merged updates.

    ``spans[name]`` is ``V``, a ``(k, d_in)`` matrix whose rows span the allowed
    input directions; ``R`` is ``(rank, k)`` and ``B`` is ``(d_out, rank)``, zero at
    start. The loss is the answer letter only. MedMCQA validation, which the
    harness scores, is never read.
    """
    torch.manual_seed(seed)
    modules = dict(model.named_modules())
    adapters, handles = {}, []
    for name, v in spans.items():
        mod = modules[name]
        v = v.to(mod.weight.device, torch.float32)
        r = torch.nn.Parameter(torch.randn(rank, v.shape[0], device=v.device) / v.shape[0] ** 0.5)
        b = torch.nn.Parameter(torch.zeros(mod.out_features, rank, device=v.device))
        adapters[name] = (v, r, b)
        handles.append(mod.register_forward_hook(
            lambda m, i, o, v=v, r=r, b=b: o + ((i[0].float() @ v.T) @ r.T @ b.T).to(o.dtype)))
    rows = load_dataset("openlifescienceai/medmcqa", split="train")
    rows = rows.filter(lambda r: r["cop"] in (0, 1, 2, 3))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    letters = [tokenizer(f" {c}", add_special_tokens=False)["input_ids"][-1] for c in "ABCD"]
    optim = torch.optim.AdamW([p for _, r, b in adapters.values() for p in (r, b)], lr=lr)
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        for step in range(steps):
            picked = [rows[i] for i in order[step * batch:(step + 1) * batch]]
            prompts = [PROMPT.format(q=r["question"].strip(), a=r["opa"], b=r["opb"],
                                     c=r["opc"], d=r["opd"]) for r in picked]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=512).to(device)
            logits = model(**enc).logits[:, -1, letters].float()
            target = torch.tensor([r["cop"] for r in picked], device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.4f}", flush=True)
    finally:
        for h in handles:
            h.remove()
    return {f"{n}.weight": (b @ r @ v).detach().cpu() for n, (v, r, b) in adapters.items()}
