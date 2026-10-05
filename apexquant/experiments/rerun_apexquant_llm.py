"""Rerun ApexQuant's Table 2: WikiText-2 perplexity of quantized LLMs.

Faithful rerun of the paper's headline language-model experiment on the two models
small enough to run here in full: TinyLlama-1.1B and Phi-1.5 (both fan-in d >= 2048,
squarely in ApexQuant's favourable regime). Each decoder Linear weight is quantized
then reconstructed (ApexQuant's inference path) with the three training-free methods:

  * RTN-absmax : per-row absmax, uniform grid, NO rotation.
  * QuaRot-RTN : random orthogonal rotation over the fan-in, absmax, uniform grid.
  * ApexQuant  : same rotation, per-row L2 norm, density-matched Lloyd-Max codebook.

We report WikiText-2 raw test perplexity at sequence length 2048 (the paper's
protocol), at bit budgets {2, 4, 8}. (Our packer supports 2/4/8, not the paper's 6.)

    HF_HOME=... uv run python experiments/rerun_apexquant_llm.py
"""

from __future__ import annotations

import argparse
import functools

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from apexquant.codebook.apex_beta_levels import apex_beta_levels
from apexquant.codebook.uniform_levels import uniform_levels
from apexquant.quantize.quantize_to_indices import quantize_to_indices

DEV = torch.device("cuda")
MODELS = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "Phi-1.5": "microsoft/phi-1_5",
}


@functools.lru_cache(maxsize=None)
def rand_orthogonal(d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g))
    q *= torch.sign(torch.diagonal(r))
    return q.to(DEV, torch.float32)


@functools.lru_cache(maxsize=None)
def apex_levels(bits: int, d: int) -> torch.Tensor:
    return apex_beta_levels(bits, d, device=DEV)


def quantize_weight(w: torch.Tensor, method: str, bits: int) -> torch.Tensor:
    """Return the reconstructed weight (ApexQuant's inference path)."""
    orig_dtype = w.dtype
    r = w.to(DEV, torch.float32)
    d = r.shape[1]
    if method == "rtn":
        scale = r.abs().amax(1, keepdim=True).clamp_min(1e-12)
        idx = quantize_to_indices(r / scale, uniform_levels(bits, device=DEV))
        recon = scale * uniform_levels(bits, device=DEV)[idx.long()]
        return recon.to(orig_dtype)
    M = rand_orthogonal(d)
    rot = r @ M.T
    if method == "quarot":
        scale = rot.abs().amax(1, keepdim=True).clamp_min(1e-12)
        levels = uniform_levels(bits, device=DEV)
    else:
        scale = rot.pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-12)
        levels = apex_levels(bits, d)
    idx = quantize_to_indices(rot / scale, levels)
    recon = (scale * levels[idx.long()]) @ M
    return recon.to(orig_dtype)


def quantize_model_(model, method: str, bits: int) -> None:
    """In-place: replace every decoder Linear weight with its reconstruction."""
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "layers." in name and mod.weight.shape[1] >= 512:
            mod.weight.data = quantize_weight(mod.weight.data, method, bits)


def bos_token_id(model) -> int | None:
    """The model's beginning-of-sequence id, or None if it declares none.

    Checked in three places because a multimodal wrapper keeps the text
    generation settings a level down: Gemma-3 arrives as
    ``Gemma3ForConditionalGeneration`` whose language settings live under
    ``config.text_config``.
    """
    for cfg in (model.config, getattr(model.config, "text_config", None),
                getattr(model, "generation_config", None)):
        bos = getattr(cfg, "bos_token_id", None) if cfg is not None else None
        if isinstance(bos, int):
            return bos
    return None


@torch.no_grad()
def perplexity(model, enc, seq_len=2048) -> float:
    """WikiText-style perplexity over non-overlapping windows.

    Each window is given the model's BOS as its first token. Cutting a corpus
    into windows puts every window but the first at a position the model was
    never trained to see -- text starting cold, with no BOS -- and models differ
    enormously in how much they mind. Llama-family models mind a little. Gemma
    minds a great deal: measured on gemma-3-12b, the mean loss over the first
    eight tokens of a window is 11.29 without a BOS against 8.09 with one, and
    it is still elevated 128 tokens in, which inflated its WikiText-2
    perplexity from about 8 to about 80.

    That also makes the uncorrected number a function of the window length,
    since a short window pays the penalty more often per token -- which reads as
    a context-length effect and is not one.

    The BOS displaces rather than extends: the window keeps its length and its
    last token goes unscored, so the token count per window is ``seq_len - 1``
    exactly as before and the arithmetic below is unchanged. Models that declare
    no BOS are evaluated exactly as they were.

    Args:
        model: A causal LM already on ``DEV``.
        enc: ``(ids,)`` where ``ids`` is ``(1, total_tokens)``.
        seq_len: Window length.

    Returns:
        Perplexity over every full window.
    """
    ids = enc[0]
    bos = bos_token_id(model)
    n = ids.numel() // seq_len
    nll, ntok = 0.0, 0
    for i in range(n):
        chunk = ids[0, i * seq_len:(i + 1) * seq_len].unsqueeze(0).to(DEV)
        if bos is not None:
            chunk = torch.cat(
                [torch.full((1, 1), bos, dtype=chunk.dtype, device=chunk.device),
                 chunk[:, :-1]], dim=1)
        out = model(chunk, labels=chunk)
        nll += out.loss.float().item() * (seq_len - 1)
        ntok += seq_len - 1
    return float(torch.tensor(nll / ntok).exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()

    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])

    print("# Rerun of ApexQuant Table 2: WikiText-2 perplexity (seq len "
          f"{args.seq_len}, lower is better)\n")
    print("# Methods: RTN (no rotation), QuaRot (rotation+uniform), Apex "
          "(rotation+density-matched). Decoder Linear weights only.\n")

    for mname, repo in MODELS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        enc = tok(text, return_tensors="pt").input_ids
        base = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16).to(DEV).eval()
        ppl_fp = perplexity(base, (enc,), args.seq_len)
        sd = {k: v.detach().clone() for k, v in base.state_dict().items()}

        print(f"## {mname}  (fan-in d>=2048; {enc.numel()} test tokens)\n")
        print("| bits | RTN | QuaRot | Apex |")
        print("|-----:|----:|-------:|-----:|")
        print(f"| FP16 | {ppl_fp:.3f} | {ppl_fp:.3f} | {ppl_fp:.3f} |")
        for bits in (8, 4, 2):
            row = {}
            for method in ("rtn", "quarot", "apex"):
                base.load_state_dict(sd)
                quantize_model_(base, method, bits)
                p = perplexity(base, (enc,), args.seq_len)
                row[method] = p if p < 1e6 else float("inf")
            def fmt(v):
                return f"{v:.3f}" if v < 1000 else f"{v:.3e}"
            print(f"| {bits} | {fmt(row['rtn'])} | {fmt(row['quarot'])} | {fmt(row['apex'])} |")
        base.load_state_dict(sd)
        del base, sd
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
