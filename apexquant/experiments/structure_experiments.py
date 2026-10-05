"""The four experiments of "What the rotation gives away" (HQQ / SINQ).

Run with a checkpoint to use real weights, or with no argument to use a synthetic
stand-in::

    uv run python experiments/structure_experiments.py [CKPT]

Each block prints the numbers that fill one ``\\needs{}`` flag in the paper. None
of them changes the recursion; every one is a stage-level or preprocessing-level
swap that the geometric decay law accommodates by construction.
"""

from __future__ import annotations

import sys

import torch

from apexquant import (
    ApexQuantConfig,
    dequantize_weight,
    error_tail_statistics,
    generalized_gaussian_levels,
    lp_levels,
    turboquantize_weight,
)
from apexquant.codebook import apex_beta_levels, trellis_quantize

BITS, GROUP = 2, 512


def load_weights(path: str | None) -> list[torch.Tensor]:
    """Return a list of 2-D weights: real ones if given a checkpoint, else synthetic."""
    if path is None:
        torch.manual_seed(0)
        # A column-imbalanced stand-in, which is the regime SINQ targets. Real
        # transformer weights have this structure; a plain iid draw does not, and
        # would make the Sinkhorn block look like a no-op for the wrong reason.
        return [torch.randn(1024, 2048) * torch.exp(torch.randn(2048) * 0.8)]
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    return [v.float() for v in state.values()
            if v.ndim == 2 and min(v.shape) >= GROUP][:8]


def sample_fd(fan_in: int, n: int, seed: int = 0) -> torch.Tensor:
    """Coordinates drawn from f_d, the post-rotation marginal."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, fan_in, generator=g)
    return (v / v.norm(dim=1, keepdim=True))[:, 0]


def experiment_1_error_tail(weights: list[torch.Tensor]) -> None:
    """Is our per-weight reconstruction error actually light-tailed?"""
    print("\n=== 1. Error tail in the original frame ===")
    print(f"{'stage':<22}{'P':>3}{'excess kurt':>14}{'top 0.1%':>11}{'gg shape':>11}")
    for label, codebook in (("apex (ours)", "apex"), ("absmax uniform", "uniform")):
        for passes in (1, 2, 3):
            stats = []
            for w in weights:
                cfg = ApexQuantConfig(bits=BITS, group_size=GROUP, passes=passes,
                                       codebook=codebook)
                err = w - dequantize_weight(turboquantize_weight(w, cfg))
                stats.append(error_tail_statistics(err))
            n = len(stats)
            print(f"{label:<22}{passes:>3}"
                  f"{sum(s['excess_kurtosis'] for s in stats) / n:>14.3f}"
                  f"{sum(s['top_share'] for s in stats) / n:>11.4f}"
                  f"{sum(s['gg_shape'] for s in stats) / n:>11.2f}")
    print("  Gaussian reference: excess kurtosis 0.00, top 0.1% ~0.0126, shape 2.00.")
    print("  HQQ's hyper-Laplacian model would sit at shape < 1.")


def experiment_2_lp_stage() -> None:
    """An l_p stage inside the l_2 recursion: rho must rise, l_p must fall."""
    print("\n=== 2. l_p stage codebook (HQQ objective, our recursion) ===")
    u = sample_fd(GROUP, 400_000)
    print(f"{'stage':<20}{'rho^2':>10}{'rho':>9}{'E|e|^0.7':>11}{'top 0.1%':>11}{'gg shape':>10}")
    for label, levels in (("apex (MSE)", apex_beta_levels(BITS, GROUP)),
                          ("lp p=1.0", lp_levels(BITS, GROUP, p=1.0)),
                          ("lp p=0.7", lp_levels(BITS, GROUP, p=0.7)),
                          ("lp p=0.5", lp_levels(BITS, GROUP, p=0.5))):
        err = u - levels[(u[:, None] - levels[None, :]).abs().argmin(1)]
        rho2 = (GROUP * err.pow(2).mean()).item()
        stats = error_tail_statistics(err)
        print(f"{label:<20}{rho2:>10.5f}{rho2 ** 0.5:>9.4f}"
              f"{err.abs().pow(0.7).mean():>11.5f}"
              f"{stats['top_share']:>11.4f}{stats['gg_shape']:>10.2f}")


def experiment_3_shaped_trellis() -> None:
    """1MAD-p: keep the trellis, reshape only the generator marginal."""
    print("\n=== 3. Heavy-tailed trellis generator (1MAD-p) ===")
    torch.manual_seed(0)
    x = torch.randn(16_384)
    print(f"{'generator':<20}{'rho^2':>10}{'rho':>9}{'max|e|':>10}{'top 0.1%':>11}")
    for shape in (2.0, 1.5, 1.0, 0.7):
        recon = trellis_quantize(x, BITS, fine=generalized_gaussian_levels(BITS + 1, shape))
        err = x - recon
        rho2 = (err.pow(2).mean() / x.pow(2).mean()).item()
        label = f"1MAD (shape {shape})" if shape == 2.0 else f"1MAD-p shape {shape}"
        print(f"{label:<20}{rho2:>10.5f}{rho2 ** 0.5:>9.4f}"
              f"{err.abs().max():>10.4f}{error_tail_statistics(err)['top_share']:>11.4f}")


def experiment_4_sinkhorn(weights: list[torch.Tensor]) -> None:
    """Sinkhorn before the rotation, on both the unweighted and weighted metric."""
    print("\n=== 4. Sinkhorn normalization before the rotation ===")
    print(f"{'':<14}{'P':>3}{'unweighted':>13}{'+sinkhorn':>12}"
          f"{'col-rel spread':>16}{'+sinkhorn':>12}")
    for passes in (1, 2, 3):
        rows = []
        for w in weights:
            base = dict(bits=BITS, group_size=GROUP, passes=passes, codebook="apex")
            out = []
            for kw in ({}, {"sinkhorn_iters": 10}):
                d = dequantize_weight(turboquantize_weight(w, ApexQuantConfig(**base, **kw))) - w
                rel = (d.norm() / w.norm()).item()
                col = d.norm(dim=0) / w.norm(dim=0).clamp_min(1e-12)
                out.append((rel, (col.std() / col.mean()).item()))
            rows.append(out)
        n = len(rows)
        agg = [[sum(r[i][j] for r in rows) / n for j in (0, 1)] for i in (0, 1)]
        print(f"{'apex':<14}{passes:>3}{agg[0][0]:>13.5f}{agg[1][0]:>12.5f}"
              f"{agg[0][1]:>16.4f}{agg[1][1]:>12.4f}")
    print("  Unweighted error is what our objective sees and it barely moves.")
    print("  The per-column relative-error spread is the mechanism: equalizing it")
    print("  is what an activation-weighted metric rewards, and it is why the sign")
    print("  of the effect depends on how activations correlate with the columns.")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    weights = load_weights(path)
    print(f"weights: {len(weights)} matrix/matrices, "
          f"{'synthetic' if path is None else path}, {BITS}-bit, group {GROUP}")
    experiment_1_error_tail(weights)
    experiment_2_lp_stage()
    experiment_3_shaped_trellis()
    experiment_4_sinkhorn(weights)


if __name__ == "__main__":
    main()
