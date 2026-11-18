"""Quantize with LiftQuant's released code and read its dense weights back."""

from __future__ import annotations

import os
import subprocess

import torch
from safetensors.torch import load_file

from .settings import LQ_EPOCHS, LQ_WBITS

RECIPE = ["--w_sym", "--abits", "16", "--kbits", "16", "--vbits", "16", "--true-sequential",
          "--act-order", "--use_fpinps", "--Rres_init", "Hadamard", "--batch_size", "2",
          "--usefullfp", "--training_trans", "--finetuning_weights", "--align", "1",
          "--lscale_lr", "5e-3", "--lexw_lr", "2e-2", "--lw_lr", "2e-5", "--la_lr", "2e-3",
          "--lt_lr", "2e-4"]


def run_liftquant(path: str, net: str, expc: str, windows: str, scratch: str
                  ) -> dict[str, torch.Tensor]:
    """LiftQuant on ``path`` with block correction, calibrated on our ``windows``.

    The recipe is the repository README's (its ``--nsamples``/``--epochs`` are
    ambiguous to the parser, so both numbered variants are set). LiftQuant's own
    RedPajama loader reads a path on the authors' machines, so the calibration
    set is written into the dataloader cache that ``main.py`` reads first, under
    the ``redpajama`` name its parser accepts; the content is ours. Runs
    under ``$LQ_PYTHON`` in ``$LQ_REPO``.
    """
    python, repo = os.environ.get("LQ_PYTHON"), os.environ.get("LQ_REPO")
    if not python or not repo:
        raise RuntimeError("set LQ_PYTHON and LQ_REPO for the LiftQuant environment")
    ids = torch.load(windows)
    family = "llama-3" if "llama-3" in net.lower() else "llama-2"
    cache = os.path.join(scratch, "cache")
    os.makedirs(cache, exist_ok=True)
    loader = []
    for w in ids:
        tar = w[None].clone()
        tar[:, :-1] = -100
        loader.append((w[None], tar))
    torch.save(loader, os.path.join(
        cache, f"dataloader_{family}_redpajama_{ids.shape[0]}_{ids.shape[1]}.cache"))
    n, epochs = str(ids.shape[0]), str(LQ_EPOCHS)
    subprocess.run([python, "main.py", "--model", path, "--net", net, "--save_dir", scratch,
                    "--cache_dir", cache, "--calib_dataset", "redpajama", "--wbits", str(LQ_WBITS),
                    "--expc", expc, "--seqlen", str(ids.shape[1]), "--nsamples1", n,
                    "--nsamples2", n, "--epochs1", epochs, "--epochs2", epochs, *RECIPE],
                   check=True, cwd=repo, env=dict(os.environ, PYTHONPATH=repo))
    pth = os.path.join(scratch, net, f"{net}+{expc}.pth")
    out = os.path.join(scratch, "dense.safetensors")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([python, "-m", "clinical.liftquant_extract", "--repo", repo, "--model", path,
                    "--pth", pth, "--expc", expc, "--wbits", str(LQ_WBITS), "--out", out],
                   check=True, env=dict(os.environ, PYTHONPATH=here))
    weights = load_file(out)
    os.remove(out)
    os.remove(pth)
    return weights
