"""Is a P=2 artifact the P=2 prefix of a P=3 one, on the published Gemma weights?

``quantise_hf_progressive`` claims it in its docstring, the paper claims it as
Corollary 1, and ``tab:kl`` depends on it: the P=1 and P=2 columns are read as
prefixes of the published P=3 artifact rather than quantised separately. If the
prefix and a standalone encode disagree, those two columns describe a model
nobody shipped.

Four checks, all bit-level, on one real weight of ``google/gemma-3-12b-it``:

    1 prefix        codes of a fresh P=3 encode, first two passes,
                    against a fresh standalone P=2 encode
    2 reproducible  the fresh P=3 encode against the *published* artifact's
                    stored codes, which is the stronger claim: it says the
                    artifact on the hub is the one these settings produce
    3 deterministic the same P=3 encode run twice
    4 row blocking  the whole slice at once against ``row_block``, the other
                    bit-identity the converter relies on

Check 2 is the one that can fail for an uninteresting reason: it compares against
whatever settings the artifact records, so a mismatch means the artifact was
written by different code or different settings, not that the recursion is wrong.
It is here because that distinction is exactly what a disagreeing KL number needs.

A row slice is enough, and is used so the test fits beside a running job: output
rows never interact under this recursion, which check 4 is the control for.

    HF_HOME=/mnt/bulk/hf uv run python experiments/prefix_identity_gemma.py \
        --artifact /mnt/bulk/gemma-kl/trellis-p3
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantise_hf_progressive import make_codec  # noqa: E402

from apexquant.vq.progressive_artifact import quantize  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Rows of the weight to encode. The recursion treats rows independently, so this
#: bounds VRAM without weakening any of the four checks; check 4 is the control.
DEFAULT_ROWS = 32

#: Row block for check 4, deliberately not a divisor of DEFAULT_ROWS so the last
#: block is short.
BLOCK_ROWS = 12

#: Only the source checkpoint has one; the converter shards without an index.
INDEX_FILE = "model.safetensors.index.json"


def source_weight(repo: str, param: str, rows: int) -> torch.Tensor:
    """The first ``rows`` rows of ``param`` from the unquantised checkpoint."""
    index = json.load(open(hf_hub_download(repo, INDEX_FILE)))["weight_map"]
    shard = hf_hub_download(repo, index[param])
    with safe_open(shard, framework="pt") as handle:
        return handle.get_slice(param)[:rows].to(DEV)


def stored_codes(artifact: str, base: str, rows: int) -> tuple:
    """The published ``(codes, scale)`` for ``base``, sliced to ``rows``.

    The converter writes sharded safetensors without an index file, so the
    shards are scanned rather than looked up. Raises if the key is absent, which
    is the only useful thing to do with an artifact that does not contain the
    tensor it is being asked about.
    """
    out = []
    for suffix in (".codes", ".scale"):
        key = base + suffix
        for shard in sorted(pathlib.Path(artifact).glob("*.safetensors")):
            with safe_open(shard, framework="pt") as handle:
                if key in handle.keys():
                    out.append(handle.get_slice(key)[:, :rows].to(DEV))
                    break
        else:
            raise KeyError(f"{key} is in no shard of {artifact}")
    return tuple(out)


def report(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    """Print one bit-level comparison and return whether it passed."""
    same = a.shape == b.shape and bool(torch.equal(a, b))
    if same:
        print(f"  {name:32s} IDENTICAL  {tuple(a.shape)}")
        return True
    if a.shape != b.shape:
        print(f"  {name:32s} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
        return False
    differing = int((a != b).sum())
    print(f"  {name:32s} DIFFERS    {differing}/{a.numel()} entries "
          f"({100.0 * differing / a.numel():.3f}%)")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="published P=3 artifact")
    ap.add_argument("--param", default="language_model.model.layers.0.self_attn.q_proj",
                    help="quantised tensor base name, without .codes/.weight")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.artifact, "progressive_config.json")))
    print(f"# artifact {args.artifact}")
    print(f"# stage {cfg['stage']}, P={cfg['passes']}, bits={cfg['bits']}, "
          f"g={cfg['group_size']}, seed={cfg['seed']}, "
          f"trellis_bits={cfg.get('trellis_bits')}, shape={cfg.get('trellis_shape')}")
    print(f"# {args.param}, first {args.rows} rows\n")

    w = source_weight(cfg["source_model"], args.param + ".weight", args.rows).float()
    codec = make_codec(cfg["stage"], cfg["group_size"], cfg["bits"], cfg["seed"],
                       cfg.get("trellis_bits", 12), cfg.get("trellis_shape", 2.0))
    encode = lambda passes, block=None: quantize(  # noqa: E731
        w, codec, passes, cfg["group_size"], cfg["seed"], row_block=block)

    codes3, scale3 = encode(cfg["passes"])
    codes2, scale2 = encode(2)
    again3, ascale3 = encode(cfg["passes"])
    blocked3, bscale3 = encode(cfg["passes"], BLOCK_ROWS)
    pub_codes, pub_scale = stored_codes(args.artifact, args.param, args.rows)

    checks = [
        ("1 prefix codes", codes3[:2], codes2),
        ("1 prefix scale", scale3[:2], scale2),
        ("2 published codes", codes3, pub_codes),
        ("2 published scale", scale3, pub_scale.to(scale3.dtype)),
        ("3 rerun codes", codes3, again3),
        ("3 rerun scale", scale3, ascale3),
        ("4 row-blocked codes", codes3, blocked3),
        ("4 row-blocked scale", scale3, bscale3),
    ]
    passed = [report(name, a, b) for name, a, b in checks]
    print(f"\n# {sum(passed)}/{len(passed)} bit-identical")


if __name__ == "__main__":
    main()
