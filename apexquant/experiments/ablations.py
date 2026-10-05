"""Controlled ablations for the progressive density-matched quantizer.

Every study changes ONE knob and holds the rest fixed, driven by a single unified
``reconstruct`` so the only difference between conditions is the variable under
test. Reconstruction rel-error is measured on many real Clay v1.5 layers; we report
mean +/- std across layers so a difference is only claimed when it clears the
spread. Studies:

  A1  fresh vs fixed rotation per residual pass  (tests the re-isotropization theorem)
  A2  codebook x passes: apex vs gaussian vs uniform
  A3  group size (rotation dimension)
  A4  rotation type: structured Hadamard vs dense random-orthogonal vs identity

    uv run python experiments/ablations.py path/to/clay-v1.5.ckpt
"""

from __future__ import annotations

import argparse
import functools

import torch
import torch.nn.functional as F

from apexquant.codebook.apex_beta_levels import apex_beta_levels
from apexquant.codebook.lloyd_max_levels import lloyd_max_levels
from apexquant.codebook.uniform_levels import uniform_levels
from apexquant.quantize.quantize_to_indices import quantize_to_indices
from apexquant.rotation.inverse_random_rotation import inverse_random_rotation
from apexquant.rotation.random_rotation import random_rotation

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@functools.lru_cache(maxsize=None)
def _dense_orth(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g))
    return (q * torch.sign(torch.diagonal(r))).to(DEV, torch.float32)


def _levels(codebook: str, bits: int, gs: int) -> torch.Tensor:
    if codebook == "apex":
        return apex_beta_levels(bits, gs, device=DEV)
    if codebook == "gaussian":
        # Unit-variance Lloyd-Max; coords are std-normalized to unit variance below.
        return lloyd_max_levels(bits, device=DEV)
    return uniform_levels(bits, device=DEV)


def _normalize(rot: torch.Tensor, codebook: str) -> torch.Tensor:
    if codebook == "apex":
        return rot.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)   # L2 sphere
    if codebook == "gaussian":
        return rot.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)  # std
    return rot.abs().amax(-1, keepdim=True).clamp_min(1e-12)              # absmax


def _rotate(x, seed, rotation, fwd):
    """x: (out, ng, gs). fwd=True forward rotation, False inverse."""
    if rotation == "identity":
        return x
    if rotation == "hadamard":
        return random_rotation(x, seed) if fwd else inverse_random_rotation(x, seed)
    M = _dense_orth(x.shape[-1], seed)          # dense random-orthogonal
    return x @ M.T if fwd else x @ M


def reconstruct(w, *, bits, passes, codebook, gs, rotation, fresh):
    """Unified progressive quantize->dequantize; returns reconstructed weight."""
    out_f, in_f = w.shape
    x = F.pad(w, (0, (-in_f) % gs)).view(out_f, -1, gs)
    levels = _levels(codebook, bits, gs)
    residual = x
    for p in range(passes):
        seed = 0 + (p if fresh else 0)
        rot = _rotate(residual, seed, rotation, True)
        scale = _normalize(rot, codebook)
        idx = quantize_to_indices(rot / scale, levels)
        residual = residual - _rotate(scale * levels[idx.long()], seed, rotation, False)
    return (x - residual).view(out_f, -1)[:, :in_f]


def rel_err(w, **kw) -> float:
    return float((reconstruct(w, **kw) - w).norm() / w.norm())


def load_layers(ckpt, n=8):
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    keys = [k for k, v in sd.items()
            if k.startswith("model.encoder.transformer") and v.ndim == 2 and v.shape[1] >= 1024]
    keys = keys[::max(1, len(keys) // n)][:n]
    return [(k.split("transformer.")[-1], sd[k].to(DEV, torch.float32)) for k in keys]


def summarize(vals):
    t = torch.tensor(vals)
    return f"{t.mean():.4f} +/- {t.std():.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    layers = load_layers(args.checkpoint)
    base = dict(bits=2, codebook="apex", gs=512, rotation="hadamard", fresh=True)
    print(f"# Ablations on {len(layers)} real Clay transformer layers "
          f"(2-bit; mean +/- std of reconstruction rel-error)\n")

    # A1: fresh vs fixed rotation per residual pass -- the theorem's mechanism.
    print("## A1  Fresh vs fixed rotation per residual pass\n")
    print("# Prediction: fresh decays geometrically; fixed stalls (residual not re-isotropized).\n")
    print("| passes | fresh rotation | fixed rotation |")
    print("|-------:|----------------|----------------|")
    for P in (1, 2, 3, 4):
        fresh = [rel_err(w, **{**base, "passes": P, "fresh": True}) for _, w in layers]
        fixed = [rel_err(w, **{**base, "passes": P, "fresh": False}) for _, w in layers]
        print(f"| {P} | {summarize(fresh)} | {summarize(fixed)} |")

    # A2: codebook x passes.
    print("\n## A2  Codebook x passes (fresh rotation, gs=512)\n")
    print("| passes | apex | gaussian | uniform |")
    print("|-------:|------|----------|---------|")
    for P in (1, 2, 3):
        cols = [summarize([rel_err(w, **{**base, "passes": P, "codebook": cb}) for _, w in layers])
                for cb in ("apex", "gaussian", "uniform")]
        print(f"| {P} | " + " | ".join(cols) + " |")

    # A3: group size (rotation dimension), apex 2-bit x2.
    print("\n## A3  Group size (apex, hadamard, 2 passes)\n")
    print("| group size | rel-error |")
    print("|-----------:|-----------|")
    for gs in (64, 128, 256, 512, 1024):
        print(f"| {gs} | {summarize([rel_err(w, **{**base, 'gs': gs, 'passes': 2}) for _, w in layers])} |")

    # A4: rotation type, apex 2-bit x2.
    print("\n## A4  Rotation type (apex, gs=512, 2 passes)\n")
    print("| rotation | rel-error |")
    print("|----------|-----------|")
    for rot in ("hadamard", "dense", "identity"):
        print(f"| {rot} | {summarize([rel_err(w, **{**base, 'rotation': rot, 'passes': 2}) for _, w in layers])} |")


if __name__ == "__main__":
    main()
