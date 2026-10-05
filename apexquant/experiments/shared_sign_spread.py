"""What one sign pattern per pass costs against one per group.

Appendix B names this gap and does not measure it: the deployed rotation draws a
single sign vector for the whole model at each pass (random_sign_vector is a
function of (d, seed) alone, broadcast over every group), so Proposition 2's
expectation is realized once per pass rather than once per group. The question
that leaves open is whether the realized layer rate concentrates anyway. It can,
because the groups carry different directions, so one shared epsilon still
decorrelates the errors it induces across them.

This script answers it directly. For each of many sign-pattern draws it records
the layer mean of the per-group squared error under the deployed shared pattern
and under an independent pattern per group, and compares the spread of those two
layer means over draws. It also times the two rotations, since a per-group
stream replaces a broadcast row by a materialized (groups, d) sign tensor.

    uv run python experiments/shared_sign_spread.py path/to/clay-v1.5.ckpt
"""

from __future__ import annotations

import argparse
import time

import torch

from apexquant.codebook.apex_beta_levels import apex_beta_levels
from apexquant.quantize.quantize_to_indices import quantize_to_indices
from apexquant.rotation.random_rotation import random_rotation
from apexquant.rotation.walsh_hadamard_transform import walsh_hadamard_transform

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROUP_SIZE = 512
BITS = 2
DRAWS = 256
TIMING_REPEATS = 50
SEED_BASE = 90_000
LAYERS = {
    "layers.0 qkv (spikiest)": "model.encoder.transformer.layers.0.0.to_qkv.weight",
    "layers.11 fc2": "model.encoder.transformer.layers.11.1.net.3.weight",
    "layers.23 fc2": "model.encoder.transformer.layers.23.1.net.3.weight",
}


def unit_groups(weight: torch.Tensor) -> torch.Tensor:
    """The layer's first-pass unit residual directions, one row per group."""
    w = weight.to(DEV, torch.float32)
    out_f, in_f = w.shape
    pad = (-in_f) % GROUP_SIZE
    w = torch.nn.functional.pad(w, (0, pad)).view(out_f, -1, GROUP_SIZE).reshape(-1, GROUP_SIZE)
    return w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def per_group_rotation(v: torch.Tensor, seed: int) -> torch.Tensor:
    """random_rotation with an independent sign pattern for every group."""
    gen = torch.Generator(device=DEV).manual_seed(int(seed))
    signs = torch.randint(0, 2, v.shape, generator=gen, device=DEV).float() * 2 - 1
    return walsh_hadamard_transform(v * signs)


def group_errors(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Per-group squared quantization error of the rotated unit directions."""
    idx = quantize_to_indices(x, levels)
    return (x - levels[idx.long()]).pow(2).sum(-1)


def draw_means(v: torch.Tensor, levels: torch.Tensor):
    """Layer-mean rate per draw under the shared pattern and under per-group ones."""
    shared, split = [], []
    for k in range(DRAWS):
        shared.append(float(group_errors(random_rotation(v, SEED_BASE + k), levels).mean()))
        split.append(float(group_errors(per_group_rotation(v, SEED_BASE + k), levels).mean()))
    return torch.tensor(shared), torch.tensor(split)


def rotation_seconds(v: torch.Tensor, per_group: bool) -> float:
    """Wall-clock seconds per rotation of the whole layer, after a warm-up."""
    call = per_group_rotation if per_group else random_rotation
    call(v, SEED_BASE)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for k in range(TIMING_REPEATS):
        call(v, SEED_BASE + k)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / TIMING_REPEATS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--bits", type=int, default=BITS)
    args = ap.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    levels = apex_beta_levels(args.bits, GROUP_SIZE, device=DEV)

    print("# One sign pattern per pass against one per group")
    print(f"# Clay v1.5, {args.bits}-bit codebook, d={GROUP_SIZE}, {DRAWS} sign-pattern draws, "
          f"device {DEV.type}")
    print("# 'shared' is the deployed rotation: random_sign_vector(d, seed) broadcast over every")
    print("# group. 'per group' draws an independent pattern per group. Both are unbiased for the")
    print("# same rate; what differs is how tightly one draw's layer mean sits around it.\n")
    print("| layer | groups | mean (shared) | mean (per group) | sd (shared) | sd (per group) | "
          "sd ratio | rotation cost |")
    print("|---|---|---|---|---|---|---|---|")
    for label, key in LAYERS.items():
        v = unit_groups(state[key])
        shared, split = draw_means(v, levels)
        slow = rotation_seconds(v, per_group=True) / rotation_seconds(v, per_group=False)
        print(f"| {label} | {v.shape[0]} | {shared.mean():.6f} | {split.mean():.6f} | "
              f"{shared.std():.2e} | {split.std():.2e} | "
              f"{shared.std() / split.std():.2f} | {slow:.2f}x |")


if __name__ == "__main__":
    main()
