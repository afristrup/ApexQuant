"""``python -m ladder <step>``: the E1 rate ladder, one rung per Slurm array task."""

from __future__ import annotations

import argparse


def main() -> None:
    """Parse one step and run it."""
    ap = argparse.ArgumentParser(prog="ladder")
    ap.add_argument("--work", required=True, help="shared directory for every E1 output")
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("windows")
    for name in ("grid", "rung"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        if name == "rung":
            p.add_argument("--task", type=int, required=True, help="row of the model's grid")
            p.add_argument("--keep", action="store_true", help="keep the dense staging checkpoint")
    sub.add_parser("summarize")
    args = ap.parse_args()

    if args.step == "windows":
        from .step_windows import step_windows
        step_windows(args.work)
    elif args.step == "grid":
        from .rung_grid import rung_grid
        for i, row in enumerate(rung_grid(args.model)):
            print(i, *row)
    elif args.step == "rung":
        from .rung_grid import rung_grid
        from .step_rung import step_rung
        step_rung(args.work, *rung_grid(args.model)[args.task], keep=args.keep)
    elif args.step == "summarize":
        from .step_summary import step_summary
        step_summary(args.work)


if __name__ == "__main__":
    main()
