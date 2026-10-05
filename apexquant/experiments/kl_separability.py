"""Which rows of tab:kl are actually separated, and which are ties.

A mean KL over 32 windows with no spread beside it cannot support an ordering
claim, and at P=3 it turned out not to: trellis L=14 has the higher mean and is
nonetheless lower than L=12 on 19 of the 32 windows, its mean carried by a tail
twice as heavy. Reporting that as a regression would have been wrong, and
reporting the ordering as clean would have been wrong too.

Every tier is scored on the same windows, so the comparison is paired, which is
both the correct test and a much more powerful one than comparing standard
errors: the window-to-window variation is enormous relative to the differences
between stages, and pairing removes it.

The threshold is stated rather than discovered. Two stages count as separated
when the paired t exceeds T_CRITICAL, which is the two-sided 5% point for 31
degrees of freedom, and the direction is reported alongside the fraction of
windows agreeing with it so that a verdict resting on two outliers is visible as
one.

    uv run python experiments/kl_separability.py /path/to/perwin_*.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import statistics

#: Two-sided 5% point of Student's t at 31 degrees of freedom, for 32 windows.
T_CRITICAL = 2.0395

#: Printed label for each artifact directory name.
LABELS = {"scalar-p3": "Scalar Lloyd--Max", "e8p-p3": "E8P lattice",
          "trellis-p3": "Trellis, 1MAD (L=12)",
          "trellisL14-p3": "Trellis, 1MAD (L=14)"}


def paired(a: list[float], b: list[float]) -> tuple[float, float, int]:
    """Return ``(mean difference, paired t, windows where a is lower)``."""
    if len(a) != len(b):
        raise ValueError(f"unpaired samples: {len(a)} against {len(b)}")
    diff = [x - y for x, y in zip(a, b)]
    mean = statistics.mean(diff)
    sd = statistics.stdev(diff)
    t = mean / (sd / math.sqrt(len(diff))) if sd else math.inf
    return mean, t, sum(1 for d in diff if d < 0)


def load(directory: pathlib.Path, passes: int) -> dict[str, list[float]]:
    """Per-window KL for every stage at one tier, keyed by printed label."""
    out = {}
    for name, label in LABELS.items():
        suffix = "" if passes == 3 else f"_P{passes}"
        path = directory / f"perwin_{name}{suffix}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        out[label] = json.loads(path.read_text())[0]["per_window"]
    return out


def report(directory: pathlib.Path, passes: int) -> None:
    """Print the means and every pairwise verdict at one tier."""
    data = load(directory, passes)
    n = len(next(iter(data.values())))
    print(f"\n## P={passes}, {n} windows")
    for label, values in data.items():
        sd = statistics.stdev(values)
        print(f"  {label:24s} mean {statistics.mean(values):8.4f}  "
              f"sd {sd:7.4f}  sem {sd / math.sqrt(n):7.4f}  "
              f"max {max(values):7.4f}")
    for x, y in itertools.combinations(data, 2):
        mean, t, wins = paired(data[x], data[y])
        verdict = "separated" if abs(t) > T_CRITICAL else "TIE"
        lower = x if mean < 0 else y
        print(f"  {x:24s} vs {y:24s}  diff {mean:+8.4f}  t {t:+7.2f}  "
              f"{verdict:9s}  lower: {lower} on {wins if mean < 0 else n - wins}/{n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="directory holding the perwin_*.json files")
    ap.add_argument("--passes", nargs="*", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    print(f"# paired separability of tab:kl's rows, t_crit={T_CRITICAL} at 5%")
    for p in args.passes:
        report(pathlib.Path(args.directory), p)


if __name__ == "__main__":
    main()
