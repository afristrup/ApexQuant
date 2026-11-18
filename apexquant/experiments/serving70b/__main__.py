"""``python -m serving70b``: list the arms, or measure one arm's serving."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("arms")
    serve = sub.add_parser("serving")
    serve.add_argument("--arm", required=True)
    serve.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.step == "arms":
        from .settings import ARMS
        for name, (stage, passes, kv) in ARMS.items():
            print(name, stage or "fp16", passes, kv)
    else:
        from .measure_serving import measure_serving
        measure_serving(args.arm, args.out)


if __name__ == "__main__":
    main()
