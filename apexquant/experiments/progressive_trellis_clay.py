"""Progressive trellis quantisation of Clay v1.5, run through the folded kernel.

`bench_real_clay.py` reads 2-D tensors out of the checkpoint and reports what
quantising them costs in weight space. It stops there for a reason it states: the
`claymodel` package pins torch 2.4 and will not import in a CUDA-13 environment,
so there was no Clay to run. The sibling `clay-model/` package is that Clay,
ported off Lightning, so this script can close the loop -- quantise the encoder's linears,
swap in :class:`ProgressiveTrellisLinear`, and read the embedding the model
actually produces rather than the norm of a weight difference.

THE FOLDED KERNEL. Every quantised layer keeps its codes packed and decodes them
inside the matmul, against the permuted table `trellis_window_table` builds once
per codec, with the codec's scale folded into the table's values. No dense weight
is formed at any point on this path, which is the property being measured: the
resident footprint is the size of the codes.

WHERE THE ROWS FALL. `ProgressiveTrellisLinear` sends a layer wider than
``TRELLIS_GEMM_MAX_ROWS`` rows to a reconstruction instead, because past that the
``(P-1)`` extra dots cost more than the reconstruction they avoid. A Clay chip is
one row per patch plus the class token, so a 256 px chip at the shipped patch size
is 1025 rows and misses the packed path by exactly the class token. This runs at
224 px, 785 rows, so the folded kernel is what executes; ``--chip 256`` shows the
same model falling back, which is worth seeing rather than tuning around.

THE STAGE. The 2-bit trellis at the codec's default register width, which is the
stage every progressive artifact in this repo is written with. ``--shape`` moves
the codeword marginal off Gaussian onto the 1MAD-p axis; it is not swept here,
and an artifact written at any other value has to record it.

INPUT. Synthetic smooth fields, not real chips: the port has no dataloader and
this repo has no Sentinel-2 data. Cosine similarities below therefore measure how
much the quantised encoder moves relative to the same encoder in fp32 on the same
input, which is the quantity of interest, but they are not a retrieval result on
real imagery.

    uv run python experiments/progressive_trellis_clay.py
    uv run python experiments/progressive_trellis_clay.py --passes 4 --chip 256
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

PORT = Path(__file__).resolve().parent.parent.parent / "clay-model"
sys.path.insert(0, str(PORT / "src"))

from clay.inference.embed import embed  # noqa: E402
from clay.inference.load_checkpoint_state_dict import (  # noqa: E402
    load_checkpoint_state_dict,
)
from clay.inference.load_sensor_metadata import load_sensor_metadata  # noqa: E402
from clay.model.build_encoder import build_encoder  # noqa: E402
from apexquant.kernels.triton_gemm_trellis import (  # noqa: E402
    TRELLIS_GEMM_MAX_ROWS,
)
from apexquant.kernels.triton_gemv_trellis import (  # noqa: E402
    trellis_window_table,
)
from apexquant.vq.progressive_artifact import (  # noqa: E402
    dequantize,
    group_size_for,
    quantize,
    stored_bits_per_weight,
)
from apexquant.vq import progressive_trellis_linear  # noqa: E402
from apexquant.vq.progressive_trellis_linear import (  # noqa: E402
    ProgressiveTrellisLinear,
)
from apexquant.vq.trellis_codec import (  # noqa: E402
    DEFAULT_TRELLIS_BITS,
    TrellisCodec,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO = "made-with-clay/Clay"
CHECKPOINT = "v1.5/clay-v1.5.ckpt"
METADATA = PORT / "clayrepo/clay-foundation-model/configs/metadata.yaml"
PLATFORM = "sentinel-2-l2a"

# Below this width the resolved group is too narrow for the stored scale to pay
# for itself; the same threshold the language-model pipeline uses.
MIN_IN_FEATURES = 32


def eligible(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Every ``nn.Linear`` wide enough to be worth a group scale, and reachable.

    Two exclusions. Narrow layers, where the stored per-group scale costs more
    than the codes save. And the projections inside ``nn.MultiheadAttention``:
    torch's own fused attention reads ``out_proj.weight`` directly, so a layer
    that keeps its weight packed cannot stand in for one. In this model that is
    a single 128x128 projection in the wavelength transformer.
    """
    fused_attention = {
        id(child)
        for module in model.modules() if isinstance(module, nn.MultiheadAttention)
        for child in module.modules()
    }
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and module.in_features >= MIN_IN_FEATURES
            and id(module) not in fused_attention]


def load_encoder() -> nn.Module:
    """The published v1.5 encoder, in the ported architecture."""
    path = hf_hub_download(REPO, CHECKPOINT)
    hyper = torch.load(path, map_location="cpu", weights_only=False).get(
        "hyper_parameters", {})
    model = build_encoder(hyper.get("model_size", "large"),
                          patch_size=hyper.get("patch_size", 8)).eval()
    model.load_state_dict(load_checkpoint_state_dict(path))
    return model.to(DEV)


def datacube(chip: int, batch: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """A batch of smooth fields; real normalised chips are spatially correlated."""
    waves, gsd = load_sensor_metadata(METADATA, PLATFORM)
    torch.manual_seed(seed)
    coarse = torch.randn(batch, waves.shape[0], chip // 16, chip // 16)
    return {
        "pixels": F.interpolate(coarse, size=(chip, chip), mode="bicubic").to(DEV),
        "time": torch.randn(batch, 4, device=DEV),
        "latlon": torch.randn(batch, 4, device=DEV),
        "gsd": gsd,
        "waves": waves,
    }


def count_packed_calls(module) -> dict[str, int]:
    """Wrap the two fused entry points so the run reports which path it took.

    The dispatch inside :class:`ProgressiveTrellisLinear` turns on a row count,
    and a layer that quietly reconstructed would still produce the right numbers
    at a wrong footprint. Counting the launches is the only way to say the
    packed path ran rather than assume it from a shape.
    """
    counts = {"gemv": 0, "gemm": 0}

    def wrap(name: str, fn):
        def counted(*a, **kw):
            counts[name] += 1
            return fn(*a, **kw)
        return counted

    module.triton_gemv_trellis = wrap("gemv", module.triton_gemv_trellis)
    module.triton_gemm_trellis = wrap("gemm", module.triton_gemm_trellis)
    return counts


def latency_ms(model: nn.Module, cube: dict, repeats: int = 10) -> float:
    """Median-ish forward time, after warmup."""
    for _ in range(3):
        embed(model, cube)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        embed(model, cube)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=4,
                    help="Deepest tier to encode; shallower ones are prefixes.")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=512)
    ap.add_argument("--trellis-bits", type=int, default=DEFAULT_TRELLIS_BITS)
    ap.add_argument("--shape", type=float, default=2.0,
                    help="Codeword marginal exponent; 2.0 is QTIP's Gaussian.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chip", type=int, default=224)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    model = load_encoder()
    targets = eligible(model)
    total = sum(p.numel() for p in model.parameters())
    covered = sum(m.weight.numel() for _, m in targets)

    print(f"# {REPO} {CHECKPOINT}, ported encoder\n")
    print(f"{total / 1e6:.1f}M parameters, {covered / 1e6:.1f}M quantised "
          f"({100 * covered / total:.1f}%) over {len(targets)} linears, "
          f"stage = {args.bits}-bit trellis L={args.trellis_bits}, "
          f"g={args.group_size}, shape={args.shape}\n")

    codecs: dict[int, TrellisCodec] = {}
    tables: dict[int, torch.Tensor] = {}
    artifacts: dict[str, tuple] = {}
    weighted = {p: 0.0 for p in range(1, args.passes + 1)}
    bits_at = {p: 0.0 for p in range(1, args.passes + 1)}

    encode_start = time.perf_counter()
    for name, module in targets:
        weight = module.weight.detach().to(DEV, torch.float32)
        out_f, in_f = weight.shape
        gs = group_size_for(in_f, args.group_size)
        if gs not in codecs:
            codecs[gs] = TrellisCodec(d=gs, bits=args.bits,
                                      trellis_bits=args.trellis_bits,
                                      seed=0, device=DEV, shape=args.shape)
            tables[gs] = trellis_window_table(codecs[gs])
        codec = codecs[gs]

        codes, scale = quantize(weight, codec, args.passes, gs, args.seed)
        errs = {}
        for p in range(1, args.passes + 1):
            recon = dequantize(codes[:p], scale[:p], codec, in_f, gs, args.seed,
                               passes=p)
            errs[p] = float((weight - recon).norm() / weight.norm())
            weighted[p] += errs[p] * weight.numel()
            bits_at[p] += stored_bits_per_weight(
                out_f, in_f, p, args.group_size, args.bits) * weight.numel()

        artifacts[name] = (codes, scale, gs, in_f)
        print(f"  {name[-52:]:52s} {str(tuple(weight.shape)):14s} g={gs:<4d} "
              + " ".join(f"P{p}={errs[p]:.4f}" for p in errs), flush=True)
        del weight, recon
    print(f"\nencoded in {time.perf_counter() - encode_start:.1f}s")

    cube = datacube(args.chip, args.batch)
    rows = args.batch * ((args.chip // model.patch_size) ** 2 + 1)
    reference = embed(model, cube)[:, 0, :]
    dense_ms = latency_ms(model, cube)
    print(f"\n{rows} rows per linear at chip={args.chip}, batch={args.batch}; "
          f"packed GEMM covers up to {TRELLIS_GEMM_MAX_ROWS}\n")

    print("## Per-depth, parameter-weighted\n")
    print("| passes | stored bits/weight | weight rel. error | "
          "cls cosine vs fp32 | forward ms | layers on the packed path |")
    print("|---:|---:|---:|---:|---:|---:|")
    print(f"| fp32 | 32.00 | 0.0000 | 1.00000000 | {dense_ms:.1f} | 0/0 |")

    for p in range(1, args.passes + 1):
        quantised = copy.deepcopy(model)
        for name, (codes, scale, gs, in_f) in artifacts.items():
            parent_name, _, attr = name.rpartition(".")
            parent = quantised.get_submodule(parent_name)
            old = getattr(parent, attr)
            words = (codes[:p].reshape(p, codes.shape[1], -1)
                     .contiguous().view(torch.int32))
            setattr(parent, attr, ProgressiveTrellisLinear(
                words, scale[:p].to(torch.float32), tables[gs], codecs[gs],
                in_f, gs, args.seed,
                None if old.bias is None else old.bias.detach()))

        counts = count_packed_calls(progressive_trellis_linear)
        tokens = embed(quantised, cube)
        cosine = F.cosine_similarity(tokens[:, 0, :], reference).min().item()
        packed = counts["gemv"] + counts["gemm"]
        fused_ms = latency_ms(quantised, cube)
        print(f"| {p} | {bits_at[p] / covered:.2f} | {weighted[p] / covered:.4f} "
              f"| {cosine:.8f} | {fused_ms:.1f} | "
              f"{packed}/{len(artifacts)} |")
        del quantised, tokens
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
