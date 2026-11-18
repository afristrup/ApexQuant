"""Per-layer retention, reversion and shift statistics for one quantized layer."""

from __future__ import annotations

import torch


@torch.no_grad()
def layer_metrics(w0: torch.Tensor, w: torch.Tensor, wq: torch.Tensor,
                  source: dict, target: dict, shrink: float | None) -> dict:
    """Statistics of ``E = W - W_hat`` against ``Delta = W - W_0``.

    ``source`` and ``target`` hold ``sigma`` and, for the target, a ``sample`` of
    inputs. ``shrink`` is ``beta^P`` for a data-free Haar-law arm, else ``None``.
    Ratios over ``Delta`` are ``None`` in layers the fine-tune leaves unchanged.
    ``kappa_diag`` is the largest per-channel ratio of target to source input
    energy, the coordinate-axis version of kappa that activation-scaled
    rounding such as AWQ's responds to; ``channel_weighted_shift`` averages that
    ratio over the channels the fine-tune moves.
    """
    dev = "cuda"
    w0, w, wq = (t.to(dev, torch.float32) for t in (w0, w, wq))
    s_s, s_t = source["sigma"].to(dev), target["sigma"].to(dev)
    x = target["sample"].to(dev, torch.float32)
    delta, err = w - w0, w - wq

    def form(a, s, b):
        return float(((a @ s) * b).sum())

    target_delta = form(delta, s_t, delta)
    channel = s_t.diagonal() / s_s.diagonal().clamp_min(1e-6 * s_s.diagonal().mean())
    out = {"frob_err": float(err.norm()), "frob_delta": float(delta.norm()),
           "trace_ratio": float(s_t.trace() / s_s.trace()),
           "kappa_diag": float(channel.max())}
    if target_delta <= 0:
        return out
    keep = (x @ (wq - w0).T).norm(dim=1).mean()
    moved = (x @ delta.T).norm(dim=1).mean()
    bound = (moved - out["frob_err"] * x.norm(dim=1).mean()) / moved
    out.update({
        "kappa_delta": target_delta / form(delta, s_s, delta),
        "retention": float(keep / moved),
        "retention_bound": float(bound),
        "bound_nonvacuous": bool(bound > 0),
        "reversion": form(err, s_t, delta) / target_delta,
        "channel_weighted_shift": float((delta.square().sum(0) * channel).sum()
                                        / delta.square().sum().clamp_min(1e-30)),
        "cosine_err_delta": float((err * delta).sum() / (err.norm() * delta.norm()).clamp_min(1e-30)),
    })
    if shrink is not None:
        out["haar_reversion"] = shrink * form(w, s_t, delta) / target_delta
    return out
