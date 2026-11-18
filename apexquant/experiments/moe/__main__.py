"""``python -m moe <step>``: E6 expert allocation, one row per job."""

from __future__ import annotations

import argparse


def main() -> None:
    """Parse one step and run it."""
    ap = argparse.ArgumentParser(prog="moe")
    ap.add_argument("--work", required=True, help="shared directory for every E6 output")
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("windows")
    sub.add_parser("grid")
    row = sub.add_parser("policy")
    row.add_argument("--task", type=int, required=True, help="row of the policy grid")
    sub.add_parser("summarize")
    args = ap.parse_args()

    if args.step == "windows":
        from .step_windows import step_windows
        step_windows(args.work)
    elif args.step == "grid":
        from .policy_grid import policy_grid
        for i, r in enumerate(policy_grid()):
            print(i, *r)
    elif args.step == "policy":
        from .policy_grid import policy_grid
        from .step_policy import step_policy
        step_policy(args.work, *policy_grid()[args.task])
    elif args.step == "summarize":
        from .step_summary import step_summary
        step_summary(args.work)


if __name__ == "__main__":
    main()
