"""``python -m clinical.liftquant_extract``: dense weights of a LiftQuant checkpoint.

Runs in the LiftQuant environment with its repository on the path. Rebuilds the
model exactly as LiftQuant's own ``main.py --load_dir`` does, stock decoder layers
with ``FWTLinear`` in place of every linear, loads the saved state, and writes
each linear's ``get_weight()`` (which applies the learned transform) plus every
other parameter as one safetensors file in the original Hugging Face names.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


def main() -> None:
    """Load ``--pth`` into ``--model``'s architecture and write ``--out``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--expc", required=True)
    ap.add_argument("--wbits", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    from quantize.tmplinear import FWTLinear, TmpLinear

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="cpu")
    with torch.device("meta"):
        names = {n for n, _ in AutoModelForCausalLM.from_config(model.config).named_parameters()}
    for layer in model.model.layers:
        linears = {n: m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)}
        for name, module in linears.items():
            with torch.no_grad():
                fake = torch.nn.Linear(module.in_features, module.out_features, module.bias is not None,
                                       device="cuda", dtype=torch.float16)
                tmp = TmpLinear(fake, args.wbits, expc=args.expc, training_trans=True)
                tmp.find_params()
                tmp.quantizer.alpha = torch.nn.Parameter(torch.zeros_like(tmp.quantizer.scale))
                fwt = FWTLinear()
                fwt.convert_form_tmplinear(tmp, bits=args.wbits, expc=args.expc, training_trans=True,
                                           groupsize=-1)
                fwt.bit_channel_convert(True)
                fwt.pack_to_int8()
            parent = layer.get_submodule(name.rpartition(".")[0]) if "." in name else layer
            setattr(parent, name.rpartition(".")[2], fwt.to("cpu"))
    missing, unexpected = model.load_state_dict(torch.load(args.pth, map_location="cpu"),
                                                assign=True, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing {missing[:3]}, unexpected {unexpected[:3]}")

    out = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, FWTLinear):
                weight = mod.to("cuda").get_weight()[:, :mod.ic]
                out[f"{name}.weight"] = weight.to(torch.bfloat16).cpu().contiguous()
                mod.to("cpu")
        for name, p in model.named_parameters():
            if name in names and name not in out:
                out[name] = p.detach().to(torch.bfloat16).cpu().contiguous().clone()
    if not any(k.endswith("q_proj.weight") for k in out):
        raise RuntimeError("no FWTLinear weights found")
    save_file(out, args.out)


if __name__ == "__main__":
    main()
