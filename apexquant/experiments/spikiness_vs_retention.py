"""Does weight spikiness explain which models survive two bits?

\\Cref{app:llmtable} reports that the two-bit trellis arm keeps between 15.6% and
84.9% of full precision's above-chance margin, that Mistral-7B and Qwen2.5-7B are
the same size and sit 44 points apart on it, and that we have no account of the
variation. This tests the obvious account.

The paper's own instrument for "how far a direction is from flat" is the
spikiness d*S(v) = d * sum_j v_j^4 of a unit direction, which Proposition 1's
deployed-rotation analysis is already stated in terms of. Here it is measured on
the raw rows of every quantized linear layer, before any rotation, since that is
the state the first pass meets. A flat direction gives 1, a Gaussian-like one
about 3, a coordinate spike gives d.

Reads weights only: no forward pass, no GPU, one tensor resident at a time.

    uv run python experiments/spikiness_vs_retention.py
"""

from __future__ import annotations

import json
import math

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

GROUP_SIZE = 512
MAX_ROWS_PER_LAYER = 256
SEED = 0
CHANCE = 100.0 / 3.0
MODELS = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "Phi-1.5": "microsoft/phi-1_5",
    "Mistral-7B": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
}
# tab:zeroshot: full precision, the two-bit trellis arm, and the retained
# above-chance margin the appendix already prints.
RETAINED = {
    "TinyLlama-1.1B": 15.6,
    "Phi-1.5": 81.0,
    "Mistral-7B": 40.5,
    "Qwen2.5-7B": 84.9,
}


def layer_spikiness(weight: torch.Tensor, generator: torch.Generator) -> list[float]:
    """Spikiness d*sum v^4 of unit-normalised groups sampled from one weight matrix."""
    out_f, in_f = weight.shape
    if in_f < GROUP_SIZE:
        return []
    groups = weight[:, : (in_f // GROUP_SIZE) * GROUP_SIZE]
    groups = groups.reshape(-1, GROUP_SIZE).float()
    n = groups.shape[0]
    if n > MAX_ROWS_PER_LAYER:
        pick = torch.randperm(n, generator=generator)[:MAX_ROWS_PER_LAYER]
        groups = groups[pick]
    norms = groups.norm(dim=-1, keepdim=True)
    keep = norms.squeeze(-1) > 0
    unit = groups[keep] / norms[keep]
    return (GROUP_SIZE * unit.pow(4).sum(-1)).tolist()


def model_spikiness(repo: str) -> dict[str, float]:
    """Aggregate spikiness over every 2-D weight the quantizer would touch."""
    path = snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"])
    generator = torch.Generator().manual_seed(SEED)
    values: list[float] = []
    import pathlib

    for shard in sorted(pathlib.Path(path).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                if not key.endswith(".weight") or "embed" in key or "lm_head" in key:
                    continue
                tensor = f.get_tensor(key)
                if tensor.ndim != 2:
                    continue
                values.extend(layer_spikiness(tensor, generator))
    t = torch.tensor(values)
    return {
        "groups": float(t.numel()),
        "mean": float(t.mean()),
        "median": float(t.median()),
        "p99": float(t.quantile(0.99)),
        "frac_above_3": float((t > 3.0).float().mean()),
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def main() -> None:
    print("# Weight spikiness against two-bit retention")
    print(f"# d*S(v) on unit-normalised groups of {GROUP_SIZE}, raw weights, "
          f"<= {MAX_ROWS_PER_LAYER} groups per layer, seed {SEED}")
    print("# flat = 1, Gaussian-like = 3, coordinate spike = d\n")
    print("| model | groups | mean d*S | median | p99 | frac > 3 | retained margin |")
    print("|---|---|---|---|---|---|---|")
    stats = {}
    for name, repo in MODELS.items():
        s = model_spikiness(repo)
        stats[name] = s
        print(f"| {name} | {int(s['groups'])} | {s['mean']:.3f} | {s['median']:.3f} | "
              f"{s['p99']:.2f} | {s['frac_above_3']:.3f} | {RETAINED[name]:.1f}\\% |")
    names = list(MODELS)
    ys = [RETAINED[n] for n in names]
    print()
    for stat in ("mean", "median", "p99", "frac_above_3"):
        xs = [stats[n][stat] for n in names]
        print(f"# Pearson r({stat}, retained margin) = {pearson(xs, ys):+.3f}  (n = {len(names)})")
    print("\n# n = 4. Nothing here is a significant correlation at that size; the point is")
    print("# whether the ordering is even consistent, and it is reported either way.")
    with open("experiments/spikiness_vs_retention.json", "w") as fh:
        json.dump({"stats": stats, "retained": RETAINED}, fh, indent=1)


if __name__ == "__main__":
    main()
