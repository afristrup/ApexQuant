"""``python -m clinical <step>``: the E3 pipeline, one step per Slurm job."""

from __future__ import annotations

import argparse

VARIANTS = ("base", "ft", "controlled", "constrained")


def main() -> None:
    """Parse one step and run it."""
    ap = argparse.ArgumentParser(prog="clinical")
    ap.add_argument("--work", required=True, help="shared directory for every E3 output")
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("windows")
    for name in ("moments", "references"):
        sub.add_parser(name).add_argument("--variant", choices=VARIANTS, required=True)
    sub.add_parser("controlled")
    lora = sub.add_parser("constrained")
    lora.add_argument("--steps", type=int, default=2000)
    lora.add_argument("--batch", type=int, default=8)
    lora.add_argument("--lr", type=float, default=5e-4)
    arm = sub.add_parser("arm")
    arm.add_argument("--task", type=int, required=True, help="row of the arm grid")
    arm.add_argument("--keep", action="store_true", help="keep the dense staging checkpoint")
    sub.add_parser("grid")
    sub.add_parser("summarize")
    args = ap.parse_args()

    if args.step == "windows":
        from .step_windows import step_windows
        step_windows(args.work)
    elif args.step == "moments":
        from .step_moments import step_moments
        step_moments(args.work, args.variant)
    elif args.step == "controlled":
        from .step_controlled import step_controlled
        step_controlled(args.work)
    elif args.step == "constrained":
        from .step_constrained import step_constrained
        step_constrained(args.work, args.steps, args.batch, args.lr)
    elif args.step == "references":
        from .step_references import step_references
        step_references(args.work, args.variant)
    elif args.step == "arm":
        from .arm_grid import arm_grid
        from .step_arm import step_arm
        step_arm(args.work, *arm_grid()[args.task], keep=args.keep)
    elif args.step == "grid":
        from .arm_grid import arm_grid
        for i, row in enumerate(arm_grid()):
            print(i, *row)
    elif args.step == "summarize":
        from .step_summarize import step_summarize
        step_summarize(args.work)


if __name__ == "__main__":
    main()
