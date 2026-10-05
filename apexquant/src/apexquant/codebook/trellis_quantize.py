"""Trellis-coded scalar quantization (QTIP-style rate-distortion gain)."""

from __future__ import annotations

import torch

from .lloyd_max_levels import lloyd_max_levels

# 4-state trellis: for each state, (next_state, subset) for input bit 0 and 1.
# States {0,2} draw from the "even" cosets {D0,D2}, states {1,3} from {D1,D3};
# the path constrains which coset is available each step, so R bits buy access to
# a 2**(R+1)-level codebook (1 bit path + R-1 bits within a size-2**(R-1) subset).
_TRELLIS = {
    0: [(0, 0), (2, 2)],
    1: [(0, 2), (2, 0)],
    2: [(1, 1), (3, 3)],
    3: [(1, 3), (3, 1)],
}


def trellis_quantize(x: torch.Tensor, bits: int,
                     fine: torch.Tensor | None = None) -> torch.Tensor:
    """Quantize a 1-D unit-variance sequence with a 4-state trellis, via Viterbi.

    Trellis-coded quantization spends its rate on a *sequence* rather than each
    sample independently: a fine ``2**(bits+1)``-level codebook is partitioned into
    four cosets, and Viterbi finds the minimum-distortion path whose coset labels
    the trellis allows, at ``bits`` bits/sample. This is the scalar precursor of
    QTIP's trellis codes; it decouples codebook size from rate.

    Args:
        x: 1-D tensor, roughly unit variance (e.g. normalized rotated weights).
        bits: rate in bits/sample.
        fine: Optional ``2**(bits+1)``-level generator marginal. Defaults to the
            Gaussian Lloyd-Max codebook, which is what QTIP's 1MAD targets. Pass
            :func:`~apexquant.codebook.generalized_gaussian_levels` to run
            the heavy-tailed ``1MAD-p`` variant: the trellis, the cosets and the
            Viterbi search are unchanged, only the marginal moves.

    Returns:
        Reconstructed sequence, same shape as ``x``.
    """
    if fine is None:
        fine = lloyd_max_levels(bits + 1)                # 2**(bits+1) levels
    elif fine.numel() != 2 ** (bits + 1):
        raise ValueError(f"fine codebook must have {2 ** (bits + 1)} levels")
    fine = fine.to(dtype=x.dtype, device=x.device)
    subsets = [fine[j::4] for j in range(4)]             # 4 cosets
    # Per-sample, per-coset best distortion and the level that achieves it.
    cost_c, level_c = [], []
    for d in subsets:
        dist = (x[:, None] - d[None, :]).pow(2)          # (T, |coset|)
        c, idx = dist.min(dim=1)
        cost_c.append(c)
        level_c.append(d[idx])
    cost_c = torch.stack(cost_c, dim=1)                  # (T, 4)
    level_c = torch.stack(level_c, dim=1)                # (T, 4)

    t = x.shape[0]
    inf = torch.tensor(float("inf"))
    path_cost = torch.full((4,), float("inf"))
    path_cost[0] = 0.0
    back = torch.zeros((t, 4), dtype=torch.long)         # previous state
    chosen = torch.zeros((t, 4), dtype=torch.long)       # coset used to enter state

    for i in range(t):
        new_cost = torch.full((4,), float("inf"))
        new_back = torch.zeros(4, dtype=torch.long)
        new_chosen = torch.zeros(4, dtype=torch.long)
        for s in range(4):
            for nxt, sub in _TRELLIS[s]:
                cand = path_cost[s] + cost_c[i, sub]
                if cand < new_cost[nxt]:
                    new_cost[nxt] = cand
                    new_back[nxt] = s
                    new_chosen[nxt] = sub
        path_cost, back[i], chosen[i] = new_cost, new_back, new_chosen

    # Backtrack the best final state.
    state = int(torch.argmin(path_cost).item())
    recon = torch.empty_like(x)
    for i in range(t - 1, -1, -1):
        recon[i] = level_c[i, chosen[i, state]]
        state = int(back[i, state].item())
    return recon
