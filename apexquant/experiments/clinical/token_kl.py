"""Token-level KL divergence of a candidate from a reference, llama.cpp's way."""

from __future__ import annotations

import torch

SLICE = 256


@torch.no_grad()
def token_kl(reference, candidate, windows: torch.Tensor, batch: int = 4) -> dict:
    """``KL(p_ref || p_cand)`` on the second half of every window, the first half as context.

    As ``llama-perplexity --kl-divergence`` does: with ``first = n/2`` only the
    predictions at positions ``first .. n - 2`` are scored, so every scored token
    has at least half a window of context. Reports the mean with its standard
    error, the median, 99th and 99.9th percentiles and maximum, the share of
    positions where both models put the same token on top, and the change in the
    probability of the actual next token (mean and RMS). ``batch`` windows go
    through each forward pass; the statistics do not depend on it.
    """
    kls, same, delta, per_window = [], [], [], []
    first = windows.shape[1] // 2
    for ids in windows.split(batch):
        ref_logits = reference(ids.to(reference.device)).logits[:, first:-1]
        cand_logits = candidate(ids.to(candidate.device)).logits[:, first:-1]
        for row in range(ids.shape[0]):
            lp = ref_logits[row].float().log_softmax(-1)
            lq = cand_logits[row].float().log_softmax(-1).to(lp.device)
            target = ids[row, first + 1:].to(lp.device)
            parts = [(a.exp() * (a - b)).sum(-1) for a, b in zip(lp.split(SLICE), lq.split(SLICE))]
            kl = torch.cat(parts).cpu()
            kls.append(kl)
            per_window.append(float(kl.mean()))
            same.append((lp.argmax(-1) == lq.argmax(-1)).cpu())
            pick = target[:, None]
            delta.append((lq.gather(1, pick).exp() - lp.gather(1, pick).exp()).squeeze(1).cpu())
            del lp, lq
        del ref_logits, cand_logits
    k, s, d = torch.cat(kls), torch.cat(same).float(), torch.cat(delta)
    q = torch.quantile(k[torch.randperm(len(k))[:2 ** 24]] if len(k) > 2 ** 24 else k,
                       torch.tensor([0.5, 0.99, 0.999]))
    return {"kl_mean": float(k.mean()), "kl_stderr": float(k.std() / len(k) ** 0.5),
            "kl_median": float(q[0]), "kl_p99": float(q[1]), "kl_p999": float(q[2]),
            "kl_max": float(k.max()), "same_top_p": float(s.mean()),
            "delta_p_mean": float(d.mean()), "delta_p_rms": float(d.square().mean().sqrt()),
            "scored_per_window": int(windows.shape[1] - 1 - windows.shape[1] // 2),
            "kl_windows": per_window}
