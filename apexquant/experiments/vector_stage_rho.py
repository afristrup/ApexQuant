"""The geometric decay law is a property of the recursion, not of the codebook.

Proposition 2 is stated for a coordinate-wise scalar quantiser, but its proof
never uses the coordinate structure: it needs only that R_p is an isometry and
that u_p is independent of s_p with a pass-independent law. Both hold for *any*
deterministic q: R^d -> R^d. (Machine-checked: `integral_norm_resid_sq_vectorStage`
in thms/contributions.lean.)

This script measures that claim. For each stage quantiser Q it reports

    rho^2 = E_{u ~ Unif(S^{d-1})} ||u - Q(u)||^2                    (one pass)

and then runs the actual P-pass recursion with a fresh rotation per pass,
checking err(P) == rho^P.

Three stage quantisers, all DATA-FREE (no Hessian, no calibration corpus):

  scalar   the ApexQuant Lloyd-Max codebook for f_d, applied coordinate-wise.
           At b=2, d=512 this should reproduce rho = 0.3422, i.e. rho^2 = 0.1171
           -- which is also ApexQuant's tabulated one-stage distortion (0.117)
           and QTIP's Table-1 scalar Lloyd-Max row on an i.i.d. Gaussian (0.118).

  e8       a finite E8-lattice codebook at the same rate, 8 dimensions at a
           time. Fixed lattice, nothing fitted (cf. QuIP#'s E8P).

  tcq      trellis-coded quantisation with a bitshift trellis and computed
           codes, decoded by Viterbi. Data-free by construction: the codeword
           for a state is a hash of the state, so there is no codebook to fit.

The point of the comparison is NOT that we beat QTIP. QTIP's Table 1 reports
0.118 / 0.089 / 0.069 for the same three families on an i.i.d. Gaussian source,
standalone, and our 0.1171 / 0.0906 / 0.0702 reproduce it to 2% or better; the
point is that the geometric law holds for all three, so a better stage quantiser
buys a better envelope for free, and the greedy tax is a property of the stage,
not of refinement. What a better rho does NOT buy is a cheaper decode: see
experiments/lattice_decode_cost.py and experiments/generated_codebook_rho.py,
which bracket the decode cost of a lattice stage at 0.8-3.9x the scalar one and
show that the cheap (generated) end of that bracket costs distortion.

    uv run python experiments/vector_stage_rho.py
    uv run python experiments/vector_stage_rho.py --d 256 --trellis-bits 16
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from apexquant.codebook.apex_beta_levels import apex_beta_levels

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# sphere sampling and rotations
# --------------------------------------------------------------------------

def sample_sphere(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    """n points uniform on S^{d-1}."""
    x = torch.randn(n, d, generator=gen, device=DEV, dtype=torch.float32)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def hadamard(d: int) -> torch.Tensor:
    """Walsh-Hadamard matrix of order d (d a power of two), normalised."""
    assert d & (d - 1) == 0, "Hadamard needs a power of two"
    H = torch.ones(1, 1, device=DEV, dtype=torch.float32)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(d)


def randomised_hadamard(x: torch.Tensor, H: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Apply R = H diag(eps) with fresh iid signs -- the deployed rotation."""
    eps = torch.randint(0, 2, (x.shape[-1],), generator=gen, device=DEV,
                        dtype=torch.float32) * 2 - 1
    return (x * eps) @ H.T


# --------------------------------------------------------------------------
# stage quantisers:  Q : (n, d) unit vectors -> (n, d) reconstructions
# --------------------------------------------------------------------------

def make_scalar_stage(bits: int, d: int):
    """ApexQuant Lloyd-Max codebook for f_d, applied coordinate-wise."""
    levels = apex_beta_levels(bits, d).to(DEV, torch.float32)
    bnd = 0.5 * (levels[1:] + levels[:-1])

    def Q(u: torch.Tensor) -> torch.Tensor:
        return levels[torch.bucketize(u.contiguous(), bnd)]

    Q.rate = float(bits)
    Q.name = f"scalar-lloydmax-b{bits}"
    return Q


def e8_codebook(n_codewords: int) -> torch.Tensor:
    """The `n_codewords` E8 lattice points of smallest norm.

    E8 = D8 u (D8 + 1/2), D8 = {x in Z^8 : sum(x) even}. We enumerate integer
    points in a box, keep those in E8, and take the shortest `n_codewords`.
    """
    # All E8 points of norm^2 <= 12 have |coord| <= 3 in D8 and |coord| <= 3.5 in
    # the coset, and there are 117361 of them -- comfortably more than the 65536
    # we need at 2 bits/dim. Base box [-4,3] covers both after the +1/2 shift.
    rng = torch.arange(-4, 4, device=DEV, dtype=torch.int8)      # 8^8 = 16.8M
    grids = torch.meshgrid(*([rng] * 8), indexing="ij")
    pts = torch.stack([g.reshape(-1) for g in grids], dim=1).to(torch.float32)
    even = pts[(pts.sum(1).to(torch.int64) % 2) == 0]
    e8 = torch.cat([even, even + 0.5], dim=0)
    nrm = e8.pow(2).sum(1)
    keep = nrm <= 12.0 + 1e-6
    e8, nrm = e8[keep], nrm[keep]
    assert e8.shape[0] >= n_codewords, f"only {e8.shape[0]} E8 points enumerated"
    order = nrm.argsort()
    return e8[order[:n_codewords]].contiguous()


def make_e8_stage(bits_per_dim: int, d: int, n_probe: int = 20000,
                  gen: torch.Generator | None = None):
    """Finite E8-lattice codebook at `bits_per_dim`, scale tuned on the sphere.

    8 dims per block, so the codebook has 2^(8*bits_per_dim) entries.
    The single global scale is chosen to minimise distortion -- one scalar,
    fitted to the *known* spherical marginal, not to any data.
    """
    assert d % 8 == 0
    n_cw = 2 ** (8 * bits_per_dim)
    cb = e8_codebook(n_cw)                                    # (K, 8)
    cb_sq = cb.pow(2).sum(1)                                  # (K,)

    def _quantise(blocks: torch.Tensor, scale: float) -> torch.Tensor:
        """blocks: (M, 8) -> nearest scaled codeword."""
        out = torch.empty_like(blocks)
        step = max(1, (1 << 24) // max(1, n_cw))
        for i in range(0, blocks.shape[0], step):
            b = blocks[i : i + step] / scale
            # ||b - c||^2 = ||b||^2 - 2 b.c + ||c||^2 ; drop ||b||^2
            idx = (cb_sq.unsqueeze(0) - 2.0 * (b @ cb.T)).argmin(dim=1)
            out[i : i + step] = cb[idx] * scale
        return out

    # tune the scale by golden-section on a probe sample
    g = gen if gen is not None else torch.Generator(device=DEV).manual_seed(0)
    probe = sample_sphere(n_probe // (d // 8) + 1, d, g).reshape(-1, 8)
    lo, hi = 1e-3 / math.sqrt(d), 4.0 / math.sqrt(d)
    for _ in range(28):
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        d1 = (probe - _quantise(probe, m1)).pow(2).sum().item()
        d2 = (probe - _quantise(probe, m2)).pow(2).sum().item()
        if d1 < d2:
            hi = m2
        else:
            lo = m1
    scale = 0.5 * (lo + hi)

    def Q(u: torch.Tensor) -> torch.Tensor:
        n = u.shape[0]
        return _quantise(u.reshape(-1, 8), scale).reshape(n, -1)

    Q.rate = float(bits_per_dim)
    Q.name = f"e8-lattice-b{bits_per_dim}"
    Q.scale = scale
    return Q


def make_tcq_stage(bits_per_dim: int, d: int, trellis_bits: int = 12,
                   seed: int = 0):
    """Trellis-coded quantisation with a bitshift trellis and computed codes.

    State is a `trellis_bits`-bit register; each step shifts in `bits_per_dim`
    new bits, so state s has 2^bits_per_dim successors
    s' = ((s << k) | b) & (2^L - 1). The codeword attached to a state is a
    *computed* pseudo-Gaussian function of the state (a hash), so there is no
    codebook to store or fit -- this is what makes the scheme data-free.
    Decoding is exact Viterbi over the d-step trellis.
    """
    k = bits_per_dim
    L = trellis_bits
    S = 1 << L
    K = 1 << k

    # computed code: hash(state) -> uniform -> inverse normal CDF -> N(0,1)
    s_idx = torch.arange(S, device=DEV, dtype=torch.int64)
    h = (s_idx * 2654435761 + 0x9E3779B9 + seed * 1013904223) & 0xFFFFFFFF
    h = (h ^ (h >> 15)) * 2246822519 & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 3266489917 & 0xFFFFFFFF
    h = h ^ (h >> 16)
    unif = (h.to(torch.float64) + 0.5) / 2.0**32
    codeword = (torch.erfinv(2.0 * unif - 1.0) * math.sqrt(2.0)).to(torch.float32)
    codeword = codeword / codeword.std()                       # unit variance

    # successor table: succ[s, b] = ((s << k) | b) & (S-1)
    b_idx = torch.arange(K, device=DEV, dtype=torch.int64)
    succ = (((s_idx.unsqueeze(1) << k) | b_idx.unsqueeze(0)) & (S - 1))  # (S,K)
    # predecessors of state t: any s with succ[s,b] == t. Since the low k bits
    # of t are b and the shift drops the top k bits, the predecessors of t are
    # {(t >> k) | (j << (L-k)) : j in [0, 2^k)}.
    j_idx = torch.arange(K, device=DEV, dtype=torch.int64)
    pred = ((s_idx.unsqueeze(1) >> k) | (j_idx.unsqueeze(0) << (L - k)))  # (S,K)

    scale = 1.0 / math.sqrt(d)   # unit-norm vector -> coords ~ N(0, 1/d)

    pred_flat = pred.reshape(1, -1)                                # (1, S*K)

    def _viterbi(x: torch.Tensor) -> torch.Tensor:
        """x: (n, dd) ~ N(0,1). Exact Viterbi over the bitshift trellis."""
        n, dd = x.shape
        cost = torch.full((n, S), 1e30, device=DEV)
        cost[:, 0] = 0.0
        back = torch.empty((n, dd, S), dtype=torch.uint8, device=DEV)
        for t in range(dd):
            # cost of arriving at each state s (which emits codeword[s])
            prev = cost.gather(1, pred_flat.expand(n, -1)).reshape(n, S, K)
            best, arg = prev.min(dim=2)                                # (n, S)
            step = (x[:, t].unsqueeze(1) - codeword.unsqueeze(0)).pow(2)
            cost = best + step
            back[:, t] = arg.to(torch.uint8)
        state = cost.argmin(dim=1)                                     # (n,)
        rec = torch.empty_like(x)
        for t in range(dd - 1, -1, -1):
            rec[:, t] = codeword[state]
            j = back[:, t].gather(1, state.unsqueeze(1)).squeeze(1).to(torch.int64)
            state = (state >> k) | (j << (L - k))                       # pred[state, j]
        return rec

    def Q(u: torch.Tensor) -> torch.Tensor:
        n = u.shape[0]
        out = torch.empty_like(u)
        # back[] dominates memory: n * d * S bytes
        bs = max(1, min(n, (1 << 31) // max(1, u.shape[1] * S)))
        for i in range(0, n, bs):
            out[i : i + bs] = _viterbi(u[i : i + bs] / scale) * scale
        return out

    Q.rate = float(bits_per_dim)
    Q.name = f"tcq-L{L}-b{bits_per_dim}"
    return Q


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def measure_rho(Q, d: int, n: int, gen: torch.Generator) -> float:
    """rho^2 = E ||u - Q(u)||^2 over the sphere."""
    tot, cnt = 0.0, 0
    chunk = max(1, min(n, (1 << 22) // max(1, d)))
    while cnt < n:
        m = min(chunk, n - cnt)
        u = sample_sphere(m, d, gen)
        tot += (u - Q(u)).pow(2).sum().item()
        cnt += m
    return math.sqrt(tot / cnt)      # E||u||^2 = 1, so this is rho


def run_progressive(Q, d: int, n: int, passes: int, gen: torch.Generator):
    """The actual recursion with a fresh randomised-Hadamard rotation per pass.

    Returns err(P) = sqrt(E ||r_P||^2) / ||w|| for P = 0..passes.
    """
    H = hadamard(d)
    w = sample_sphere(n, d, gen)          # unit-norm rows, wlog
    r = w.clone()
    out = [1.0]
    for _ in range(passes):
        s = r.norm(dim=1, keepdim=True).clamp_min(1e-30)
        eps = torch.randint(0, 2, (d,), generator=gen, device=DEV,
                            dtype=torch.float32) * 2 - 1
        ru = ((r / s) * eps) @ H.T                     # u = R r / ||r||
        rec = Q(ru)
        r = r - s * (((rec @ H) * eps))                # r - s R^T q(u)
        out.append(math.sqrt(r.pow(2).sum().item() / n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--n-rho", type=int, default=200_000)
    ap.add_argument("--n-rho-tcq", type=int, default=20_000)
    ap.add_argument("--n-prog", type=int, default=4096)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--trellis-bits", type=int, default=12)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    d, b = args.d, args.bits
    print(f"device={DEV}  d={d}  bits/dim={b}\n")

    stages = []
    if "scalar" not in args.skip:
        stages.append(make_scalar_stage(b, d))
    if "e8" not in args.skip:
        t0 = time.time()
        stages.append(make_e8_stage(b, d))
        print(f"[e8] codebook built in {time.time()-t0:.1f}s  "
              f"scale={stages[-1].scale:.5f}")
    if "tcq" not in args.skip:
        stages.append(make_tcq_stage(b, d, trellis_bits=args.trellis_bits))

    results = {}
    for Q in stages:
        gen = torch.Generator(device=DEV).manual_seed(1234)
        t0 = time.time()
        n_rho = args.n_rho_tcq if Q.name.startswith("tcq") else args.n_rho
        rho = measure_rho(Q, d, n_rho, gen)
        gen = torch.Generator(device=DEV).manual_seed(99)
        errs = run_progressive(Q, d, args.n_prog, args.passes, gen)
        dt = time.time() - t0
        bits_gained = math.log2(1.0 / rho) if rho > 0 else float("inf")
        stored = b + 16.0 / d
        results[Q.name] = {
            "rho": rho, "rho_sq": rho * rho,
            "bits_gained_per_pass": bits_gained,
            "bits_stored_per_pass": stored,
            "greedy_tax": stored - bits_gained,
            "err_measured": errs,
            "err_predicted": [rho ** p for p in range(args.passes + 1)],
            "seconds": dt,
        }
        print(f"\n=== {Q.name} ===")
        print(f"  rho          = {rho:.5f}   (rho^2 = {rho*rho:.5f})")
        print(f"  bits/pass    = {bits_gained:.3f} gained vs {stored:.3f} stored"
              f"   -> greedy tax {stored - bits_gained:.3f}")
        print("  P :   measured   predicted   ratio")
        for p, (m, pr) in enumerate(zip(errs, results[Q.name]["err_predicted"])):
            r = m / pr if pr > 0 else float("nan")
            print(f"  {p} :   {m:.6f}   {pr:.6f}   {r:.4f}")
        print(f"  ({dt:.1f}s)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"d": d, "bits": b, "results": results}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
