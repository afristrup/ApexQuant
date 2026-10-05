"""What the trellis encoder costs, and what buys the state bits back.

The paper's trellis control runs $L$ from 10 to 16 and the arms cross over
between 12 and 14, so the interesting regime was exactly the one the encoder
could not reach: ACS work is $2^{L+k}$ per weight, $2^{14}$ at $L=12$ and
$2^{18}$ at $L=16$. This script measures the changes that move that number.

Two of the three have since landed in the codec, and the numbers below are with
them. The third has not, which is why the sweep still says something: $L=16$
exceeds the fused kernel's register budget and falls back to the torch encoder,
so it is the one row here still measured on the slow path.

The first is exact and is not a trade. Predecessors of state $s$ depend on $s$
only through $s \\gg k$, so the $2^k$ states sharing that value share one ACS
minimum; taking it once per class rather than once per state is a factor $2^k$
off both the comparisons and the survivor table, with the same output. The
second is also exact and is not about the algorithm at all: written in torch the
path metrics live in DRAM and cross it four times a step, so the encoder ran at
memory bandwidth doing sixteen thousand comparisons of arithmetic per weight.
Keeping them in registers and looping the $d$ steps inside one launch is another
factor 44 at $L=12$ and 68 at $L=14$, rising with $L$ until the registers run out.
The third is the M-algorithm, which *is* a trade, and the point of
measuring rather than asserting: in error correction dropping the true path is a
frame error, but here every surviving path is a valid encoding, so a pruning
error costs reconstruction and nothing else. `rho` is the readout, and the exact
encoder is the reference to read it against.

Uniform directions rather than real weights, because that is the measurement the
paper's Table of stage `rho` is made of, and a beam that is indistinguishable
there is indistinguishable in the only place the number is used.

WHAT IT FOUND, so nobody re-runs it hoping otherwise. The two exact changes are
worth their factors and the beam is not. `trellis_encoder_levers.txt` has the
sweep; the line to read is that beam `rho^2` at fixed B barely moves between
$L=12$ and $L=16$ (B=64: 0.0813 against 0.0806; B=256: 0.0764 against 0.0744). A
beam of width B caps the search at roughly B continuations no matter how many
states exist, so the pruned encoder cannot spend state bits it refuses to visit
-- which is the one thing it was wanted for. Concretely, B=256 at $L=16$ reaches
$rho^2=0.0744$ in 21 min/7B, and exact $L=12$ reaches 0.0752 in 0.2: the beam is
dominated by a factor of a hundred by just running the smaller trellis exactly,
and it is slower in wall clock than the encoder it was meant to accelerate.
`beam` stays in the codec as the instrument that measured this, and defaults off.

    uv run python experiments/trellis_encoder_levers.py
    uv run python experiments/trellis_encoder_levers.py --trellis-bits 16
"""

from __future__ import annotations

import argparse
import time

import torch

from apexquant.vq.trellis_codec import (
    HAVE_TRITON,
    _ACS_MAX_STATES as ACS_MAX,
    TrellisCodec,
)

HAVE_ACS = HAVE_TRITON and torch.cuda.is_available()

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sample_sphere(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    """Uniform on the unit sphere, the same construction as vector_stage_rho."""
    v = torch.randn(n, d, generator=gen, device=DEV)
    return v / v.norm(dim=1, keepdim=True)


def rho_sq(codec: TrellisCodec, u: torch.Tensor) -> float:
    """rho^2 = E ||u - Q(u)||^2 with ||u|| = 1, so this is the paper's number."""
    return (u - codec(u)).pow(2).sum().item() / u.shape[0]


def timed(fn, reps: int = 1):
    """Wall clock with the queue drained, since every call here is a launch.

    The discarded first call is not politeness about warm caches: the exact arm
    runs a Triton kernel and a cold call pays its compile, which at these sizes
    is most of the measurement -- 0.24s against 0.006s steady state, so a cold
    number understates the encoder by 40x. The beam arms are pure torch and move
    by a few percent, but they are warmed too so the columns compare.

    ``reps`` takes the best of several, which the kernel arm needs and the beam
    arm does not. Once a launch is a few milliseconds the clock ramp is most of
    the spread: the same L=12 point measured 23x and 42x in two runs of this
    script that differed only in what had run before them. The minimum is the
    right summary because the noise is one-sided -- contention and low clocks
    can only add time -- and it is what makes the column reproducible enough to
    justify a constant in the codec.
    """
    fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.time()
        out = fn()
        if DEV == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.time() - t0)
    return out, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512, help="group size, = trellis length")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--trellis-bits", type=int, default=12)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--beams", type=int, nargs="*",
                    default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--seed", type=int, default=0)
    # The path table sweeps L itself, so it is the same table whatever
    # --trellis-bits says and running it twice only produces two copies that
    # disagree: the second lands on a GPU already hot from the first, which at
    # these sizes moves the absolute throughputs by 3x.
    ap.add_argument("--skip-paths", action="store_true")
    args = ap.parse_args()

    L, k, d = args.trellis_bits, args.bits, args.d
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    u = sample_sphere(args.n, d, gen)
    print(f"device={DEV}  d={d}  bits={k}  L={L}  states={1 << L}  n={args.n}")
    print(f"exact ACS per weight = 2^{L + k} = {1 << (L + k):,}\n")

    exact = TrellisCodec(d=d, bits=k, trellis_bits=L, device=DEV)
    # Best-of-3 here for the same reason as the table below, and it matters most
    # on this line: it is the first GPU work the script does, so at reps=1 it is
    # measured at idle clocks and reads 258 M w/s against ~700 warm.
    (codes_ref, t_ref) = timed(lambda: exact.encode(u), reps=3)
    r_ref = rho_sq(exact, u)
    thr = args.n * d / t_ref
    print(f"exact       rho^2={r_ref:.6f}  {t_ref:6.2f}s  "
          f"{thr / 1e6:6.2f} M weights/s  "
          f"7B ETA {7e9 / thr / 60:8.1f} min")

    # The beam is a real search over a smaller frontier, so it can only lose;
    # the column that matters is how much, not whether.
    print(f"\n{'beam':>6} {'rho^2':>10} {'vs exact':>9} {'codes=':>7} "
          f"{'time':>8} {'M w/s':>8} {'7B ETA':>10}")
    for B in args.beams:
        if B >= (1 << L):
            continue
        codec = TrellisCodec(d=d, bits=k, trellis_bits=L, device=DEV, beam=B)
        (codes, t) = timed(lambda: codec.encode(u), reps=3)
        r = rho_sq(codec, u)
        agree = (codes == codes_ref).float().mean().item()
        thr = args.n * d / t
        print(f"{B:6d} {r:10.6f} {r / r_ref:8.4f}x {agree:6.1%} "
              f"{t:7.2f}s {thr / 1e6:7.2f} {7e9 / thr / 60:9.1f} min")

    print("\n'codes=' is packed-byte agreement with the exact path, not an error "
          "measure:\ndistinct paths of equal metric are equally good encodings.")

    if args.skip_paths:
        return

    # Where the cost vector lives, at fixed algorithm and fixed output. This is
    # what sets `_ACS_MAX_STATES`, so it is measured across the widths either
    # side of it rather than at the shipped one only.
    print(f"\n# cost vector in DRAM (_viterbi_codes) against in registers "
          f"(_acs_codes), d={d}\n# 'rows' is one chunk's worth at each width, "
          f"which is what the survivor table admits.")
    print(f"\n{'L':>3} {'states':>7} {'rows':>6} {'torch M w/s':>12} "
          f"{'kernel M w/s':>13} {'speedup':>8} {'codes=':>7} {'rho^2 torch':>12} "
          f"{'rho^2 kernel':>13}")
    for Lk in (8, 10, 12, 14, 16):
        if Lk % k:
            continue
        codec = TrellisCodec(d=d, bits=k, trellis_bits=Lk, device=DEV)
        rows = min(1024, codec._rows_per_chunk(d))
        v = u[:rows]
        x = v / codec.scale
        (ct, tt) = timed(lambda: codec._viterbi_codes(x), reps=5)
        w = rows * d
        # Deliberately run the kernel past `_ACS_MAX_STATES` too: the cap is a
        # claim about where the register file runs out, and a table that stops
        # at the cap cannot show it. Above it the kernel still returns the right
        # answer, just slowly, which is the whole reason the gate is a
        # performance gate and not a correctness one.
        try:
            (ck, tk) = timed(lambda: codec._acs_codes(x), reps=5)
        except Exception as exc:                     # noqa: BLE001 - reported
            print(f"{Lk:3d} {codec.S:7,d} {rows:6d} {w / tt / 1e6:12.2f} "
                  f"{'n/a':>13} {'--':>8} {'--':>7} "
                  f"{rho_sq(codec, v):12.6f}  {type(exc).__name__}")
            continue
        agree = (ct.to(torch.uint8) == ck).float().mean().item()
        # rho is the claim; the codes are only a diagnostic, so both are shown.
        rt = (v - codec._walk(ct) * codec.scale).pow(2).sum().item() / rows
        rk = ((v - codec._walk(ck.to(torch.int64)) * codec.scale)
              .pow(2).sum().item() / rows)
        gate = "" if codec.S <= ACS_MAX else "  (past the cap; torch path runs)"
        print(f"{Lk:3d} {codec.S:7,d} {rows:6d} {w / tt / 1e6:12.2f} "
              f"{w / tk / 1e6:13.2f} {tt / tk:7.1f}x {agree:6.1%} "
              f"{rt:12.6f} {rk:13.6f}{gate}")


if __name__ == "__main__":
    main()
