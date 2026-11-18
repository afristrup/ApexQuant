"""``python -m serving70b.layer_output_error OUT.json``: per-layer output error on real inputs.

For every decoder linear of the 70B model, the relative error ``|(W_hat - W) x| /
|W x|`` on the inputs that layer sees on held-out text, for the trellis
reconstruction at P = 1, 2, 3 (read from the vLLM code cache of a one-GPU run, so
nothing is re-encoded) and for 4-bit round-to-nearest at group 128 without a
rotation, the data-free baseline that holds up on this model. Also the share of
the input energy in its largest channel, which shows where activations are
concentrated. The layers where trellis is far worse than RTN are the collapse.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from apexquant.vq.progressive_artifact import dequantize, group_size_for
from apexquant.vq.trellis_codec import TrellisCodec

from .settings import MODEL, REVISION

FUSED = {"self_attn.qkv_proj": ("q_proj", "k_proj", "v_proj"),
         "mlp.gate_up_proj": ("gate_proj", "up_proj")}
TEXT = ("The history of the Roman Empire spans more than a thousand years, from the "
        "founding of the city to the fall of Constantinople. ") * 40


def rtn(w: torch.Tensor, bits: int = 4, group: int = 128) -> torch.Tensor:
    """Asymmetric min-max round-to-nearest per group of ``group`` inputs."""
    g = w.view(w.shape[0], -1, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    step = (hi - lo).clamp_min(1e-12) / (2 ** bits - 1)
    return (torch.round((g - lo) / step) * step + lo).view_as(w)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--codes", required=True, help="trellis tp1-rank0 cache directory")
    ap.add_argument("--tokens", type=int, default=1024)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16,
                                                 device_map="cuda:0").eval()
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :args.tokens].cuda()
    inputs: dict[str, torch.Tensor] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ".layers." in name:
            hooks.append(module.register_forward_hook(
                lambda m, a, o, name=name: inputs.__setitem__(name, a[0][0].float())))
    model(ids)
    for h in hooks:
        h.remove()
    codec = TrellisCodec(512, bits=2, trellis_bits=14, seed=42, device="cuda")
    rows = []
    for path in sorted(glob.glob(os.path.join(args.codes, "model.layers.*.pt")),
                       key=lambda p: (int(p.split(".layers.")[1].split(".")[0]), p)):
        prefix = os.path.basename(path)[:-3]
        layer, kind = prefix.split(".layers.")[1].split(".", 1)
        base = f"model.layers.{layer}"
        parts = FUSED.get(kind, (kind.split(".")[-1],))
        sub = "self_attn" if kind.startswith("self_attn") else "mlp"
        names = [f"{base}.{sub}.{p}" for p in parts]
        w = torch.cat([dict(model.named_modules())[n].weight.float() for n in names])
        x = inputs[names[0]]
        codes, scale = torch.load(path, map_location="cuda")
        gs = group_size_for(w.shape[1], 512)
        y = x @ w.T
        row = {"layer": int(layer), "kind": kind, "shape": list(w.shape),
               "max_channel_energy": float((x.square().sum(0).max() / x.square().sum()))}
        for p in (1, 2, 3):
            w_hat = dequantize(codes[:p], scale[:p], codec, w.shape[1], gs, 42)
            row[f"trellis_p{p}_weight"] = float((w_hat - w).norm() / w.norm())
            row[f"trellis_p{p}_output"] = float(((x @ w_hat.T) - y).norm() / y.norm())
        w_rtn = rtn(w)
        row["rtn4_weight"] = float((w_rtn - w).norm() / w.norm())
        row["rtn4_output"] = float(((x @ w_rtn.T) - y).norm() / y.norm())
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump({"model": MODEL, "tokens": args.tokens, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
