"""Does the deployed rotation cost anything on a *language* model?

appendix_RHT.tex answers this for Clay v1.5 three ways: the per-pass rate against
a dense Haar rotation, the end-to-end error under substitution, and the KS
distance of the post-rotation coordinates. All three are measured on one vision
encoder, and the paper's headline claims are on language models, so the
language-model case is left open. This closes it by
repeating the end-to-end substitution on decoder weights.

The recursion is written out here rather than imported, because
``progressive_artifact.quantize`` hardcodes the deployed rotation and the whole
point is to swap it. Duplicating a recursion is how the two silently drift
apart, so the Hadamard arm is checked against the library's own output before
either arm is reported: ``--verify`` asserts they agree, and the run refuses to
print numbers if they do not.

The Haar arm draws Q from the QR of a Gaussian matrix, which is Haar-distributed
on O(d) up to the sign convention of the factorization; the signs are fixed so
the draw is exactly Haar rather than nearly so.

    uv run python experiments/haar_substitution_llm.py --model microsoft/phi-1_5
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apexquant.rotation.inverse_random_rotation import inverse_random_rotation
from apexquant.rotation.random_rotation import random_rotation
from apexquant.vq.progressive_artifact import dequantize, quantize
from apexquant.vq.scalar_codec import ScalarCodec

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PASSES = 4
BITS = 2
#: Agreement required between the inlined Hadamard arm and the library's own
#: recursion, relative to the weight norm. Both are float32 over the same
#: matmuls, so anything above this means the recursion was copied wrong.
VERIFY_TOL = 1e-6


def haar_rotation(d: int, seed: int, device) -> torch.Tensor:
    """A Haar-distributed orthogonal matrix, drawn reproducibly."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(d, d, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    # QR's Q is Haar only after fixing the sign convention: R's diagonal must be
    # positive, otherwise the draw is biased by the factorization's choice.
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.to(device=device, dtype=torch.float32)


def recurse(w: torch.Tensor, codec, passes: int, gs: int, seed: int,
            haar: bool) -> list[torch.Tensor]:
    """Return the reconstruction after each of ``passes`` passes."""
    out_f, in_f = w.shape
    pad = (-in_f) % gs
    x = F.pad(w, (0, pad)).to(torch.float32).view(out_f, -1, gs)
    residual, recon, stages = x, torch.zeros_like(x), []
    for p in range(passes):
        s = seed + p
        if haar:
            q = haar_rotation(gs, s, x.device)
            rotated = residual @ q
        else:
            rotated = random_rotation(residual, s)
        scale = rotated.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        scale = scale.to(torch.float16).to(torch.float32)
        codes = codec.encode((rotated / scale).reshape(-1, gs))
        back = scale * codec.decode(codes.reshape(out_f, -1, codes.shape[-1])
                                    ).reshape(out_f, -1, gs)
        step = back @ q.T if haar else inverse_random_rotation(back, s)
        residual = residual - step
        recon = recon + step
        stages.append(recon.reshape(out_f, -1)[:, :in_f].clone())
    return stages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/phi-1_5")
    ap.add_argument("--group-size", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)
               and m.in_features % args.group_size == 0
               and "lm_head" not in n][:args.layers]
    if not linears:
        raise ValueError(f"no Linear with in_features divisible by {args.group_size}")

    gs = args.group_size
    codec = ScalarCodec(d=gs, bits=BITS, device=DEV)
    print(f"# {args.model}, {len(linears)} layers, d={gs}, b={BITS}, "
          f"fresh rotation per pass, seed {args.seed}")
    print(f"# verifying the inlined Hadamard arm against "
          f"progressive_artifact.quantize (tol {VERIFY_TOL:g})")

    num_h = [0.0] * MAX_PASSES
    num_q = [0.0] * MAX_PASSES
    den = 0.0
    for name, mod in linears:
        w = mod.weight.data.to(DEV).float()
        den += float(w.pow(2).sum())
        had = recurse(w, codec, MAX_PASSES, gs, args.seed, haar=False)
        haar = recurse(w, codec, MAX_PASSES, gs, args.seed, haar=True)
        codes, scale = quantize(w, codec, MAX_PASSES, gs, args.seed)
        for p in range(MAX_PASSES):
            ref = dequantize(codes[:p + 1], scale[:p + 1], codec, w.shape[1],
                             gs, args.seed, passes=p + 1)
            drift = float((had[p] - ref).norm() / w.norm())
            if drift > VERIFY_TOL:
                raise AssertionError(
                    f"{name} P={p + 1}: inlined Hadamard arm differs from the "
                    f"library recursion by {drift:.2e}, above {VERIFY_TOL:g}")
            num_h[p] += float((w - had[p]).pow(2).sum())
            num_q[p] += float((w - haar[p]).pow(2).sum())
        del w

    print("\n| P | Hadamard | dense random-orthogonal | relative gap |")
    print("|---|---------:|------------------------:|-------------:|")
    for p in range(MAX_PASSES):
        h = (num_h[p] / den) ** 0.5
        q = (num_q[p] / den) ** 0.5
        print(f"| {p + 1} | {h:.4f} | {q:.4f} | {abs(h - q) / q:.1e} |")


if __name__ == "__main__":
    main()
