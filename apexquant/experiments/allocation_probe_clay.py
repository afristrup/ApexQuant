"""Can one calibration sample choose which layers get the second pass?

`mixed_depth_trellis_clay.py` ranks layers by demoting each one and reading the
damage on a single synthetic chip, then buys the second pass back in that order.
That is a calibration procedure, even though the probe carries no real data, and a
procedure that needs a particular input is worth less than one that does not. So
this asks the three questions that decide whether the ranking is a property of the
model or of the draw.

  1. Do two probes agree on the ranking? Reported as Spearman between the
     per-layer damages measured on different chips, and as the overlap of the sets
     each one would select at a given budget.
  2. Does a selection made on one chip hold up on chips it never saw? Reported as
     held-out cosine for a selection made on chip 0, against an oracle that
     selects using the held-out chip's own ranking, and against a random selection
     of the same size -- the two bounds that say whether the ranking is doing
     work.
  3. Where does the budget go? Reported by layer kind and block depth, because
     "76 of 102" is only actionable if the 76 have a describable shape.

The probe is synthetic smooth-field input throughout, so nothing here reads the
Earth-observation distribution; what varies between chips is the draw, not the
domain. That keeps the question honest but narrow: it answers whether one draw is
enough, not whether synthetic draws stand in for real chips.

    uv run python experiments/allocation_probe_clay.py
    uv run python experiments/allocation_probe_clay.py --probes 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mixed_depth_trellis_clay import build_layer, set_submodule
from progressive_trellis_clay import DEV, datacube, eligible, load_encoder

from clay.inference.embed import embed  # noqa: E402
from apexquant.kernels.triton_gemv_trellis import (  # noqa: E402
    trellis_window_table,
)
from apexquant.vq.progressive_artifact import (  # noqa: E402
    group_size_for,
    quantize,
)
from apexquant.vq.trellis_codec import (  # noqa: E402
    DEFAULT_TRELLIS_BITS,
    TrellisCodec,
)


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, as Pearson on ranks."""
    x = torch.tensor(a).argsort().argsort().float()
    y = torch.tensor(b).argsort().argsort().float()
    x = x - x.mean()
    y = y - y.mean()
    return (x @ y / (x.norm() * y.norm())).item()


def layer_kind(name: str) -> str:
    """Group layers by the role they play, for reporting where budget goes."""
    if not name.startswith("transformer"):
        return "patch embedding"
    if name.endswith("to_qkv"):
        return "attention qkv"
    if name.endswith("to_out"):
        return "attention out"
    if name.endswith("net.1"):
        return "mlp expand"
    return "mlp project"


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=512)
    ap.add_argument("--trellis-bits", type=int, default=DEFAULT_TRELLIS_BITS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chip", type=int, default=224)
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--budgets", type=int, nargs="+", default=[13, 26, 51, 76])
    args = ap.parse_args()

    model = load_encoder()
    targets = eligible(model)
    chips = [datacube(args.chip, 1, seed) for seed in range(args.probes)]
    dense_reference = [embed(model, chip)[:, 0, :] for chip in chips]

    codecs, tables, variants = {}, {}, {}
    for name, module in targets:
        weight = module.weight.detach().to(DEV, torch.float32)
        gs = group_size_for(weight.shape[1], args.group_size)
        if gs not in codecs:
            codecs[gs] = TrellisCodec(d=gs, bits=args.bits,
                                      trellis_bits=args.trellis_bits,
                                      seed=0, device=DEV)
            tables[gs] = trellis_window_table(codecs[gs])
        codes, scale = quantize(weight, codecs[gs], 2, gs, args.seed)
        bias = None if module.bias is None else module.bias.detach()
        variants[name] = {
            p: build_layer(codes, scale, tables[gs], codecs[gs], weight.shape[1],
                           gs, args.seed, bias, p)
            for p in (1, 2)
        }
        del weight
    names = list(variants)
    for name in names:
        set_submodule(model, name, variants[name][2])

    # One ranking per chip: demote a layer, read the damage on that chip, restore.
    all_two = [embed(model, chip)[:, 0, :] for chip in chips]
    damage = {}
    for index, chip in enumerate(chips):
        per_layer = []
        for name in names:
            set_submodule(model, name, variants[name][1])
            tokens = embed(model, chip)[:, 0, :]
            per_layer.append(1.0 - F.cosine_similarity(tokens, all_two[index]).item())
            set_submodule(model, name, variants[name][2])
        damage[index] = per_layer
    rankings = {i: [names[j] for j in torch.tensor(d).argsort(descending=True)]
                for i, d in damage.items()}

    print(f"# Clay v1.5, {args.probes} synthetic chips, {len(names)} layers\n")
    print("## Do two probes agree?\n")
    print("| chip pair | Spearman on damage | top-13 overlap | top-76 overlap |")
    print("|---|---:|---:|---:|")
    for other in range(1, args.probes):
        top13 = len(set(rankings[0][:13]) & set(rankings[other][:13]))
        top76 = len(set(rankings[0][:76]) & set(rankings[other][:76]))
        print(f"| 0 vs {other} | {spearman(damage[0], damage[other]):.4f} | "
              f"{top13}/13 | {top76}/76 |")

    print("\n## Does a selection made on chip 0 hold up on chips it never saw?\n")
    print("| budget | chip-0 selection | per-chip oracle | random selection |")
    print("|---:|---:|---:|---:|")
    generator = torch.Generator().manual_seed(0)
    for budget in args.budgets:
        scores = {"chip0": [], "oracle": [], "random": []}
        shuffled = [names[i] for i in torch.randperm(len(names), generator=generator)]
        selections = {
            "chip0": set(rankings[0][:budget]),
            "random": set(shuffled[:budget]),
        }
        for index in range(1, args.probes):
            selections["oracle"] = set(rankings[index][:budget])
            for key, keep in selections.items():
                for name in names:
                    set_submodule(model, name, variants[name][2 if name in keep else 1])
                tokens = embed(model, chips[index])[:, 0, :]
                scores[key].append(
                    F.cosine_similarity(tokens, dense_reference[index]).item())
        cells = {k: torch.tensor(v) for k, v in scores.items()}
        print(f"| {budget}/{len(names)} | "
              + " | ".join(f"{cells[k].mean():.6f}" for k in
                           ("chip0", "oracle", "random")) + " |")

    print("\n## Where the budget goes, by chip-0 rank\n")
    kinds = sorted({layer_kind(n) for n in names})
    print("| budget | " + " | ".join(kinds) + " | median block depth |")
    print("|---:|" + "---:|" * (len(kinds) + 1))
    for budget in args.budgets:
        keep = rankings[0][:budget]
        counts = {k: sum(1 for n in keep if layer_kind(n) == k) for k in kinds}
        depths = [int(n.split(".")[2]) for n in keep if n.startswith("transformer")]
        median = (sorted(depths)[len(depths) // 2] if depths else -1)
        print(f"| {budget}/{len(names)} | "
              + " | ".join(str(counts[k]) for k in kinds)
              + f" | {median} |")


if __name__ == "__main__":
    main()
