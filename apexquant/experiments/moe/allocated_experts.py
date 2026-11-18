"""An experts module that reads each expert at an allocated prefix depth."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AllocatedExperts(nn.Module):
    """Drop-in for a fused experts module, ``forward(hidden, top_k_index, top_k_weights)``.

    With ``static`` set, expert ``e`` is always read at ``static[e]`` passes. Without
    it, each token spends ``budget`` passes over its top-k experts, greedily where
    ``g^2 (U2_e(p) - U2_e(p + 1))`` is largest, which is optimal for a convex
    ``U2``. With ``reference`` set, per-token envelope statistics are accumulated.
    """

    def __init__(self, gate_up, down, u2, act_fn, budget=None, static=None, reference=None):
        super().__init__()
        self.gate_up, self.down, self.u2, self.act_fn = gate_up, down, u2, act_fn
        self.budget, self.static, self.reference = budget, static, reference
        self.controller_ms, self.envelope = 0.0, []

    def allocate(self, idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        if self.static is not None:
            passes = self.static.to(idx.device)[idx.clamp_max(len(self.static) - 1)]
        else:
            pmax = self.u2.shape[1]
            passes = torch.ones_like(idx)
            safe = idx.clamp_max(self.u2.shape[0] - 1)
            g2 = weights.float().square()
            for _ in range(self.budget - idx.shape[1]):
                now = self.u2[safe, (passes - 1).clamp_max(pmax - 1)]
                nxt = self.u2[safe, passes.clamp_max(pmax - 1)]
                gain = torch.where(passes < pmax, g2 * (now - nxt), torch.full_like(g2, -1.0))
                slot = gain.argmax(dim=1, keepdim=True)
                passes.scatter_add_(1, slot, (gain.gather(1, slot) > 0).long())
        end.record()
        torch.cuda.synchronize()
        self.controller_ms += start.elapsed_time(end)
        return passes

    def forward(self, hidden: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        passes = self.allocate(top_k_index, top_k_weights)
        out = torch.zeros_like(hidden)
        if self.reference is not None:
            delta = torch.zeros_like(hidden, dtype=torch.float32)
            tri = torch.zeros(hidden.shape[0], device=hidden.device)
            sq = torch.zeros(hidden.shape[0], device=hidden.device)
        experts = self.gate_up[0].shape[0]
        for e in top_k_index.unique().tolist():
            if e >= experts:
                continue
            for p in passes[top_k_index == e].unique().tolist():
                tok, slot = torch.where((top_k_index == e) & (passes == p))
                x = hidden[tok]
                gate, up = F.linear(x, self.gate_up[p - 1][e]).chunk(2, dim=-1)
                y = F.linear(self.act_fn(gate) * up, self.down[p - 1][e])
                g = top_k_weights[tok, slot, None]
                out.index_add_(0, tok, (y * g).to(out.dtype))
                if self.reference is not None:
                    rg, ru = F.linear(x, self.reference[0][e]).chunk(2, dim=-1)
                    yr = F.linear(self.act_fn(rg) * ru, self.reference[1][e])
                    d = ((yr - y).float() * g.float())
                    delta.index_add_(0, tok, d)
                    tri.index_add_(0, tok, d.norm(dim=-1))
                    sq.index_add_(0, tok, d.square().sum(-1))
        if self.reference is not None:
            self.envelope.append(torch.stack([delta.norm(dim=-1), tri, sq]).cpu())
        return out
