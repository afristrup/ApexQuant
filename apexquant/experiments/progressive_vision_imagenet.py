"""ImageNet-1K: progressive quantization of CNN weights, including convolutions.

The CNN arm of Section 5. Unlike the Clay and UNI probes, this one cannot go
through the fuser: the fuser walks ``nn.Linear`` only, and ResNet-50 keeps 91.8%
of its parameters in ``Conv2d``. The quantizer itself rejects a 4-D tensor
outright (``turboquantize_weight``: "weight must be 2-D"), so a convolution is
handled here by the one reshape that makes it a weight matrix,

    (out_C, in_C, kh, kw) -> (out_C, in_C * kh * kw),

quantized at its natural fan-in ``in_C * kh * kw`` and reshaped back. This measures
the accuracy cost of the quantization; it is not a fused kernel path, and no claim
about convolution *speed* follows from it.

A group must be a power of two that DIVIDES the fan-in, since the pass reshapes a
row into whole groups. That excludes exactly one layer across the three models:
ResNet-50's 7x7 stem, whose fan-in is 3*49 = 147 and has no power-of-two divisor
above 1. It is left dense, as the literature conventionally does anyway.

    uv run python experiments/progressive_vision_imagenet.py /path/to/ImageNet1K-val

"""

from __future__ import annotations

import argparse
import glob
import io
import os

import torch

from apexquant.config import ApexQuantConfig
from apexquant.quantize.dequantize_weight import dequantize_weight
from apexquant.quantize.turboquantize_weight import turboquantize_weight

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ("resnet50", "convnext_tiny", "efficientnet_b0")
TIERS = (1, 2, 3)
BITS = 2
MAX_GROUP = 256
N_IMAGES = 10000          # fixed seeded subset of the 50k validation images
BATCH = 128
FP32_BITS = 32.0
MIN_FANIN_GROUP = 4       # a 2-bit group must pack 8 // 2 = 4 indices per byte
KEEP_ENDS = os.environ.get("KEEP_ENDS", "0") == "1"


def group_for_fanin(fan_in: int) -> int | None:
    """Largest power of two <= MAX_GROUP that divides ``fan_in``, else None."""
    gs = MAX_GROUP
    while gs >= MIN_FANIN_GROUP:
        if fan_in % gs == 0:
            return gs
        gs //= 2
    return None


def load_val(root: str, limit: int, transform, skip: int = 0):
    """Decode a seeded subset of the validation shards through ``transform``.

    The shards are ordered by class, so taking the first ``limit`` rows would
    cover only the first few hundred of the 1000 classes and the resulting top-1
    would be meaningless. Sampling is therefore per shard, which spreads the
    subset over every class.
    """
    from PIL import Image
    import pyarrow.parquet as pq

    shards = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {root}")
    per_shard = -(-limit // len(shards))
    gen = torch.Generator().manual_seed(0)

    xs, ys = [], []
    for shard in shards:
        table = pq.read_table(shard, columns=["image", "label"])
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        # One permutation per shard from the rolling generator, sliced at `skip`.
        # Two calls with the same seed draw the identical permutation per shard, so
        # non-overlapping [skip, skip+take) windows are disjoint by construction;
        # that is how the batch-norm calibration set is kept out of the evaluation
        # set. skip=0 reproduces the sampling the reported numbers were measured on.
        order = torch.randperm(len(images), generator=gen)
        take = min(per_shard, max(0, len(images) - skip))
        idx = order[skip:skip + take].tolist()
        for i in idx:
            img = Image.open(io.BytesIO(images[i]["bytes"])).convert("RGB")
            xs.append(transform(img))
            ys.append(labels[i])
    return torch.stack(xs), torch.tensor(ys)


def quantize_in_place(model: torch.nn.Module, passes: int, keep_ends: bool = False):
    """Replace every eligible weight by its ``passes``-pass reconstruction.

    ``keep_ends`` leaves the first convolution and the classifier dense, which is
    what the PTQ literature conventionally does: both sit at an interface where the
    representation is not yet (or no longer) redundant, and quantizing them is not
    what published CNN numbers measure. Quantizing everything is the harsher
    setting and we report both.

    Returns ``(quantized_params, dense_params, stored_bits_per_quantized_weight)``.
    """
    matrices = [(n, m) for n, m in model.named_modules()
                if getattr(m, "weight", None) is not None and m.weight.ndim in (2, 4)]
    skip = set()
    if keep_ends and matrices:
        skip.add(matrices[0][0])
        for n, m in reversed(matrices):
            if m.weight.ndim == 2:
                skip.add(n)
                break

    quant_n, dense_n, payload_bits = 0, 0, 0
    with torch.no_grad():
        for name, mod in matrices:
            w = mod.weight
            if name in skip:
                dense_n += w.numel()
                continue
            flat = w.reshape(w.shape[0], -1)
            gs = group_for_fanin(flat.shape[1])
            if gs is None:
                dense_n += w.numel()
                continue
            cfg = ApexQuantConfig(bits=BITS, passes=passes, group_size=gs,
                                   codebook="apex")
            qw = turboquantize_weight(flat.float(), cfg)
            w.copy_(dequantize_weight(qw).reshape(w.shape).to(w.dtype))
            quant_n += w.numel()
            payload_bits += qw.nbytes() * 8
    for _, p in model.named_parameters():
        if p.ndim not in (2, 4):
            dense_n += p.numel()
    bits = payload_bits / quant_n if quant_n else FP32_BITS
    return quant_n, dense_n, bits


@torch.no_grad()
def logits_of(model, x):
    outs = []
    for i in range(0, x.shape[0], BATCH):
        outs.append(model(x[i:i + BATCH].to(DEV)).float().cpu())
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagenet_val_dir")
    args = ap.parse_args()
    import timm

    print("# ImageNet-1K validation: progressive 2-bit quantization of CNN weights")
    print(f"# keep_ends={KEEP_ENDS} (first conv and classifier left dense when true)")
    print(f"# {N_IMAGES} images, seeded subset; KL is against the same model at fp32\n")
    print("| model | tier | quantized | stored bits | top-1 | KL vs fp32 |")
    print("|-------|------|----------:|------------:|------:|-----------:|")

    for name in MODELS:
        torch.manual_seed(0)
        ref = timm.create_model(name, pretrained=True).eval()
        cfg_data = timm.data.resolve_model_data_config(ref)
        transform = timm.data.create_transform(**cfg_data, is_training=False)
        x, y = load_val(args.imagenet_val_dir, N_IMAGES, transform)

        ref = ref.to(DEV)
        ref_logits = logits_of(ref, x)
        ref_logp = torch.log_softmax(ref_logits, dim=-1)
        acc = (ref_logits.argmax(1) == y).float().mean().item()
        total = sum(p.numel() for p in ref.parameters())
        print(f"| {name} | fp32 | --- | {FP32_BITS:>11.2f} | {acc * 100:>5.2f}% "
              f"| {0.0:>10.4f} |")

        for passes in TIERS:
            model = timm.create_model(name, pretrained=True).eval()
            qn, dn, bits = quantize_in_place(model, passes, keep_ends=KEEP_ENDS)
            model = model.to(DEV)
            lg = logits_of(model, x)
            a = (lg.argmax(1) == y).float().mean().item()
            kl = torch.nn.functional.kl_div(
                torch.log_softmax(lg, dim=-1), ref_logp,
                log_target=True, reduction="batchmean").item()
            print(f"| {name} | P={passes} | {qn / total * 100:>8.1f}% | {bits:>11.2f} "
                  f"| {a * 100:>5.2f}% | {kl:>10.4f} |", flush=True)
        print(f"#   {name}: {qn:,} of {total:,} params quantized, {dn:,} left dense\n",
              flush=True)


if __name__ == "__main__":
    main()
