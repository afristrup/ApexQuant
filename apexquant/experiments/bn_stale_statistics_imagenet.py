"""Why progressive quantization hurts BatchNorm CNNs but not LayerNorm ones.

On ImageNet the same 2-bit progressive scheme that leaves ConvNeXt-Tiny 0.6 points
below fp32 at three passes leaves ResNet-50 4.8 points below, and at two passes the
gap is 2.6 against 58 points. The two models have comparable fan-ins (weighted mean
1420 against 2218) and comparable fractions of their parameters in small groups, so
the spherical-marginal argument does not explain it. What differs is the
normalisation: ResNet-50 has 53 BatchNorm layers and no LayerNorm, ConvNeXt-Tiny has
23 LayerNorm layers and no BatchNorm.

BatchNorm applies running statistics estimated on the *unquantised* weights. Every
quantized layer shifts its own output distribution slightly, the stale mean and
variance no longer match what arrives, and the mismatch compounds with depth.
LayerNorm normalises per sample at inference and is therefore self-correcting.

This script separates the hypothesis from the alternatives with three arms:

    quantized            the harness number, stale statistics left alone
    + BN re-estimated    running statistics recomputed on unlabelled images; needs
                         input data, so it is NOT data-free, and is reported to
                         identify the mechanism rather than as our recipe
    + BN folded          BatchNorm folded into the convolution weights BEFORE
                         quantizing, so no running statistics survive to be stale.
                         This uses the weights alone and IS data-free.

    uv run python experiments/bn_stale_statistics_imagenet.py /path/to/ImageNet1K-val

"""

from __future__ import annotations

import argparse
import copy

import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from progressive_vision_imagenet import (DEV, N_IMAGES, load_val, logits_of,
                                         quantize_in_place)

MODELS = ("resnet50", "efficientnet_b0")
PASSES = (2, 3)
BN_BATCHES = 32           # unlabelled batches for re-estimation
BN_BATCH = 64


def fold_batchnorm(model: nn.Module) -> int:
    """Fold every Conv2d-BatchNorm2d pair into the convolution, in place.

    Data-free: the fold uses only the stored affine and running parameters. After
    it, the normalisation is gone, so there are no running statistics left for
    quantization to invalidate.

    timm's ``BatchNormAct2d`` subclasses ``nn.BatchNorm2d`` but also carries the
    block's activation and drop, so replacing it with an identity would silently
    delete the network's nonlinearity -- 49 SiLUs in EfficientNet-B0, which turns
    the model linear and makes its output diverge. The replacement therefore keeps
    whatever ``drop`` and ``act`` the module held.
    """
    folded = 0
    for parent in model.modules():
        prev_name, prev = None, None
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.BatchNorm2d) and isinstance(prev, nn.Conv2d):
                setattr(parent, prev_name, fuse_conv_bn_eval(prev.eval(), child.eval()))
                tail = [m for m in (getattr(child, "drop", None), getattr(child, "act", None))
                        if m is not None and not isinstance(m, nn.Identity)]
                setattr(parent, name, nn.Sequential(*tail) if tail else nn.Identity())
                folded += 1
                prev_name, prev = None, None
                continue
            prev_name, prev = name, child
    return folded


@torch.no_grad()
def reestimate_batchnorm(model: nn.Module, x: torch.Tensor) -> int:
    """Recompute BatchNorm running statistics from unlabelled images."""
    # The caller may still be holding the model on the host: the QBN arm builder
    # quantizes on CPU and only moves to the accelerator to score. Pushing the
    # model here rather than assuming makes both call sites work, and is a no-op
    # for the one that already passed reest.to(DEV).
    model.to(DEV)
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None          # cumulative average over the batches seen
        bn.train()
    for i in range(0, min(BN_BATCHES * BN_BATCH, x.shape[0]), BN_BATCH):
        model(x[i:i + BN_BATCH].to(DEV))
    model.eval()
    return len(bns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagenet_val_dir")
    args = ap.parse_args()
    import timm

    print("# BatchNorm stale statistics under progressive quantization\n")
    print("| model | tier | arm | top-1 | KL vs fp32 |")
    print("|-------|------|-----|------:|-----------:|")

    for name in MODELS:
        torch.manual_seed(0)
        ref = timm.create_model(name, pretrained=True).eval()
        transform = timm.data.create_transform(
            **timm.data.resolve_model_data_config(ref), is_training=False)
        x, y = load_val(args.imagenet_val_dir, N_IMAGES, transform)
        # Held-out. Re-estimating the running statistics on the images the
        # accuracy is then measured on would be transductive, and the recovery it
        # reports an upper bound rather than a result; this slice is disjoint
        # from x by construction (see load_val's `skip`).
        x_cal, _ = load_val(args.imagenet_val_dir, BN_BATCHES * BN_BATCH, transform,
                            skip=-(-N_IMAGES // 14))
        ref = ref.to(DEV)
        ref_logp = torch.log_softmax(logits_of(ref, x), dim=-1)

        def score(model, tier, arm):
            lg = logits_of(model.to(DEV), x)
            acc = (lg.argmax(1) == y).float().mean().item()
            kl = torch.nn.functional.kl_div(
                torch.log_softmax(lg, dim=-1), ref_logp,
                log_target=True, reduction="batchmean").item()
            print(f"| {name} | P={tier} | {arm} | {acc * 100:>5.2f}% | {kl:>10.4f} |",
                  flush=True)

        for passes in PASSES:
            plain = timm.create_model(name, pretrained=True).eval()
            quantize_in_place(plain, passes)
            score(plain, passes, "quantized")

            reest = copy.deepcopy(plain)
            n = reestimate_batchnorm(reest.to(DEV), x_cal)
            score(reest, passes, f"+ BN re-estimated ({n} layers, held-out, not data-free)")

            pre = timm.create_model(name, pretrained=True).eval()
            nf = fold_batchnorm(pre)
            quantize_in_place(pre, passes)
            score(pre, passes, f"+ BN folded before quantizing ({nf}, data-free)")
        print()


if __name__ == "__main__":
    main()
