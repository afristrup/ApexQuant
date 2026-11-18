"""``python -m clay_eurosat OUT.json``: probe accuracy and head KL at each trellis depth."""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch
from huggingface_hub import hf_hub_download

from .settings import CHECKPOINT, PASSES, PORT, REPO, SEED, STAGES, TRAIN_SHARE

sys.path.insert(0, str(PORT / "src"))

from clay.inference.load_checkpoint_state_dict import load_checkpoint_state_dict  # noqa: E402
from clay.model.build_encoder import build_encoder  # noqa: E402

from .embed_chips import embed_chips  # noqa: E402
from .linear_probe import linear_probe  # noqa: E402
from .load_eurosat import load_eurosat  # noqa: E402
from .quantize_encoder import quantize_encoder  # noqa: E402


def main() -> None:
    """FP32 encoder, then every stage at P = 1 to 4; the probe is refit per tier, and KL
    is measured with the FP32 probe on the FP32 against the quantized embedding."""
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    path = hf_hub_download(REPO, CHECKPOINT)
    hyper = torch.load(path, map_location="cpu", weights_only=False).get("hyper_parameters", {})
    encoder = build_encoder(hyper.get("model_size", "large"),
                            patch_size=hyper.get("patch_size", 8)).eval()
    encoder.load_state_dict(load_checkpoint_state_dict(path))
    encoder = encoder.cuda()
    pixels, labels, classes = load_eurosat()
    cut = int(TRAIN_SHARE * len(labels))
    train, test = slice(0, cut), slice(cut, None)
    reference = embed_chips(encoder, pixels)
    head = linear_probe(reference[train], labels[train], len(classes))
    p_ref = head(reference[test]).log_softmax(-1)
    rows = [{"tier": "fp32", "accuracy": float((p_ref.argmax(-1) == labels[test]).float().mean())}]
    print(rows[-1], flush=True)
    for stage, passes in ((s, p) for s in STAGES for p in PASSES):
        start = time.perf_counter()
        model, count = quantize_encoder(encoder, passes, stage)
        z = embed_chips(model, pixels)
        tier_head = linear_probe(z[train], labels[train], len(classes))
        acc = float((tier_head(z[test]).argmax(-1) == labels[test]).float().mean())
        p_q = head(z[test]).log_softmax(-1)
        kl = float((p_ref.exp() * (p_ref - p_q)).sum(-1).mean())
        cos = float(torch.nn.functional.cosine_similarity(z[test], reference[test]).mean())
        rows.append({"tier": f"{stage}_p{passes}", "stage": stage, "passes": passes,
                     "accuracy": acc,
                     "head_kl": kl, "cls_cosine": cos, "linears": count,
                     "seconds": time.perf_counter() - start})
        print(rows[-1], flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump({"model": f"{REPO}/{CHECKPOINT}", "chips": len(labels), "seed": SEED,
               "train_share": TRAIN_SHARE, "classes": classes, "rows": rows},
              open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
