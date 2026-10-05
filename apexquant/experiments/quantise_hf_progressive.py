"""Stream-quantise a Hugging Face causal LM with progressive quantisation.

Quantisation here is *data-free*: no calibration corpus, no Hessian, no forward
pass. Nothing but one weight matrix is ever resident, so a 70B model can be
converted on a consumer GPU -- measured peak is under 2 GiB on the largest
Llama-3.1-70B weight, against 141 GB for the dense fp16 model. That is the
practical dividend of the zero-data constraint, and it is why this script
streams shard by shard rather than loading a model.

Disk, not VRAM, is the binding constraint. Each shard is fetched, quantised and
deleted before the next is fetched, so peak disk is one shard plus the growing
output rather than the whole source model.

Encode once at the deepest tier you want. Prefixes are bit-identical to a fresh
shallower encode (`--passes 3` then slicing gives exactly `--passes 1` and
`--passes 2`), so shallower artifacts are produced by `slice_artifact.py` at
upload time rather than by re-running this.

    uv run python experiments/quantise_hf_progressive.py \
        --model meta-llama/Llama-3.1-70B-Instruct \
        --stage e8p --passes 3 --out /mnt/bulk/llama70b/e8p-p3
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from safetensors.torch import save_file

from apexquant.vq.e8p_codec import E8PCodec
from apexquant.vq.progressive_artifact import (
    DEFAULT_GROUP_SIZE,
    DEFAULT_SEED,
    effective_bits,
    group_size_for,
    quantize,
)
from apexquant.vq.scalar_codec import ScalarCodec
from apexquant.vq.trellis_codec import (
    DEFAULT_TRELLIS_BITS,
    TrellisCodec,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Files copied verbatim so the output repo is self-describing and loadable.
PASSTHROUGH_FILES = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
    # Multimodal and chat checkpoints carry more than the original list knew
    # about. chat_template.json is not cosmetic: upstream applies the chat
    # template to every zero-shot result, so an artifact without it cannot be
    # scored comparably.
    "added_tokens.json", "chat_template.json",
    "preprocessor_config.json", "processor_config.json",
)

MIN_IN_FEATURES = 32


def drop_shard(path: str) -> None:
    """Delete a downloaded shard, including the blob it points at.

    `hf_hub_download` returns a path inside `snapshots/`, which is a symlink
    into `blobs/`. Removing only the link frees nothing, so a long run would
    keep filling the disk while appearing to clean up after itself.
    """
    real = os.path.realpath(path)
    for p in (path, real):
        try:
            os.remove(p)
        except (FileNotFoundError, IsADirectoryError):
            pass


def _rotation_path(group_size: int) -> str:
    """``"matmul"`` or ``"wht"``: which rotation this machine actually applies.

    Not a preference. The two implementations agree only to about 1.4e-6, which is
    enough to flip a nearest-codeword decision at a cell boundary, and the peel
    then compounds it. Recorded so a decoder can check that it is replaying the
    same one.
    """
    from apexquant.rotation.hadamard_backend import matmul_rotation_max_dim
    from apexquant.rotation.rotation_matrix import MATMUL_ROTATION_MAX_DIM

    probe = torch.zeros(1, group_size, device=DEV)
    threshold = matmul_rotation_max_dim(probe, MATMUL_ROTATION_MAX_DIM)
    return "matmul" if group_size <= threshold else "wht"


def make_codec(stage: str, gs: int, bits: int, seed: int,
               trellis_bits: int = DEFAULT_TRELLIS_BITS,
               trellis_shape: float = 2.0):
    """Build the stage quantiser. All three are data-free by construction."""
    if stage == "scalar":
        return ScalarCodec(d=gs, bits=bits, device=DEV)
    if stage == "e8p":
        if bits != 2:
            raise ValueError("e8p is a 2-bit-per-coordinate construction")
        return E8PCodec(d=gs, seed=seed, device=DEV)
    if stage == "trellis":
        # The codeword hash is seeded separately from the rotation, and pinned to
        # 0 so the stage is the one whose rho the paper reports.
        return TrellisCodec(d=gs, bits=bits, trellis_bits=trellis_bits, seed=0,
                            shape=trellis_shape,
                            device=DEV)
    raise ValueError(f"unknown stage {stage!r}")


def _is_moe_router(name: str) -> bool:
    """The gate that picks which experts fire, kept dense.

    A router is tiny and decisive: on Qwen3.5-35B the forty ``mlp.gate.weight``
    tensors are 21M parameters, 0.06% of the model, and they choose 8 experts out
    of 256 per token. Quantizing them to two bits saves nothing measurable and
    risks changing which experts run at all, which is a different failure from
    the graded degradation the rest of the scheme produces. Matched on the exact
    leaf so ``gate_proj`` and ``gate_up_proj``, which are ordinary projections,
    are untouched.
    """
    return name.endswith("mlp.gate.weight") or name.endswith("mlp.router.weight")


def is_quantisable(name: str, shape, exclude: "re.Pattern | None" = None,
                   quantize_head: bool = False) -> bool:
    """Whether a stored tensor is a decoder linear weight we quantise.

    Mirrors the published `quantise_`: transformer-block Linear weights only,
    never embeddings, norms or the output head. Takes the shape rather than the
    tensor so the decision is made from the safetensors header, without
    materialising anything.

    ``exclude`` is a regex matched against the tensor name, and it is needed for
    any model that is not purely a language model. ``"layers."`` appears in a
    vision tower's blocks just as it does in a text decoder's, so on
    Gemma-3-12B-it the default rule would sweep up the 81 SigLIP projections
    alongside the 336 text ones. Quantising an image encoder is a different claim
    from quantising a language model, and the numbers it would be compared against
    are the language model's, so it has to be opted into rather than fallen into.
    """
    # 2D linears, and 3D stacked-expert tensors. A fused MoE checkpoint stores
    # all E experts of a projection in one (E, out, in) tensor with no ".weight"
    # suffix, and requiring 2D silently skipped 89.6% of Qwen3.5-35B: the routed
    # experts are the model, and what got quantized instead was the MTP head.
    # The contraction is still the last axis, so (E, out, in) flattens to
    # (E*out, in) and the scheme applies unchanged.
    if len(shape) == 3:
        if shape[-1] < MIN_IN_FEATURES:
            return False          # e.g. a width-4 causal conv, not a projection
        if "layers." not in name or any(k in name for k in ("norm", "embed", "lm_head")):
            return False
        if _is_moe_router(name):
            return False
        if exclude is not None and exclude.search(name):
            return False
        return True
    if not name.endswith(".weight") or len(shape) != 2:
        return False
    # The output head is a genuine linear: the contraction runs over the hidden
    # axis, so the rotation acts on the contracted axis exactly as it does in a
    # decoder layer and the fused identity is unchanged. It is excluded by
    # default only because it is the layer whose error lands directly on the
    # logits, so including it is a quality decision to be measured rather than a
    # free byte saving. The embedding table stays out regardless: it is looked
    # up, never contracted, so there is no activation to rotate.
    if quantize_head and "lm_head" in name and "embed" not in name:
        if exclude is not None and exclude.search(name):
            return False
        return shape[1] >= MIN_IN_FEATURES
    if "layers." not in name:
        return False
    if any(k in name for k in ("norm", "embed", "lm_head")):
        return False
    if _is_moe_router(name):
        return False
    if exclude is not None and exclude.search(name):
        return False
    return shape[1] >= MIN_IN_FEATURES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Hub repo id, or a directory holding an already "
                         "downloaded model. A directory is never deleted from, "
                         "whatever --keep-shards says.")
    ap.add_argument("--stage", required=True, choices=("scalar", "e8p", "trellis"))
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--trellis-bits", type=int, default=DEFAULT_TRELLIS_BITS,
                    help="shift-register width L for --stage trellis; 2**L "
                         "states. Encoding cost scales with it.")
    ap.add_argument("--trellis-shape", type=float, default=2.0,
                    help="Generalized-Gaussian exponent of the trellis codeword "
                         "marginal (the 1MAD-p axis). 2.0 is QTIP's Gaussian "
                         "target and the only value that reproduces previously "
                         "published artifacts; below 1 is HQQ's hyper-Laplacian.")
    ap.add_argument("--passthrough-dtype",
                    choices=("preserve", "float16", "bfloat16", "float32"),
                    default="float16",
                    help="dtype for tensors that are not quantised. Defaults to "
                         "float16 rather than the source dtype: a compression "
                         "artifact carrying fp32 embeddings is paying more for "
                         "two tensors than for every weight it compressed.")
    ap.add_argument("--quantize-head", action="store_true",
                    help="Also quantise lm_head. It is a real linear so the "
                         "scheme applies unchanged, but its error lands straight "
                         "on the logits, so measure before shipping it.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-shards", action="store_true",
                    help="do not delete each source shard after use")
    ap.add_argument("--exclude", default=None,
                    help="regex of tensor names to leave alone. Required for "
                         "multimodal checkpoints: 'layers.' matches a vision "
                         "tower's blocks too. For Gemma-3-*-it pass "
                         "'vision_tower|multi_modal_projector'.")
    ap.add_argument("--row-block", type=int, default=None,
                    help="quantise this many output rows at a time. Lowers peak "
                         "VRAM a long way (291->77 MiB on a 2048x4096 test) at a "
                         "small cost in speed. NOT bit-identical to unblocked: "
                         "differing batch shapes change cuBLAS tiling, flipping "
                         "~0.01%% of codes at cell boundaries. Reconstruction "
                         "error is unchanged to 6 significant figures, but keep "
                         "one setting for a whole model rather than mixing.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    exclude = re.compile(args.exclude) if args.exclude else None
    codec = make_codec(args.stage, args.group_size, args.bits, args.seed,
                       args.trellis_bits, args.trellis_shape)

    # A directory is taken as an already-downloaded model, which is the only way
    # to convert one whose transfer you want to run yourself: the in-process
    # fetch is one file at a time from inside the encode loop, so a slow or
    # stalling link becomes a stalled conversion. `hf download --local-dir` is
    # the same bytes with a progress bar and a retry you can watch.
    local = os.path.isdir(args.model)
    if local:
        files = sorted(os.listdir(args.model))
        def fetch(repo: str, fn: str) -> str:
            return os.path.join(repo, fn)
    else:
        files = list_repo_files(args.model)
        fetch = hf_hub_download

    shards = sorted(f for f in files if f.endswith(".safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors shards in {args.model}")
    print(f"# {args.model}: {len(shards)} shards, stage={args.stage}, "
          f"P={args.passes}, g={args.group_size}, seed={args.seed}"
          f"{', local' if local else ''}")
    print(f"# claimed storage: {effective_bits(args.passes, args.group_size, args.bits):.4f} bits/weight")

    for fn in PASSTHROUGH_FILES:
        if fn in files:
            src = fetch(args.model, fn)
            with open(src, "rb") as fh, open(os.path.join(args.out, fn), "wb") as out:
                out.write(fh.read())

    meta: dict[str, dict] = {}
    n_quant = n_pass = 0
    q_params = 0
    p_params = 0
    q_bytes = 0
    t0 = time.time()

    def sidecar(shard: str) -> str:
        return os.path.join(args.out, shard + ".meta.json")

    def already_done(shard: str) -> bool:
        """A shard counts as done only if its output *and* its metadata exist.

        The metadata is written per shard rather than only at the end so a run
        that dies partway can resume without re-encoding what it already did.
        """
        return (os.path.exists(os.path.join(args.out, shard))
                and os.path.exists(sidecar(shard)))

    # Fetch the next shard while the current one is being encoded. Encode is
    # minutes per shard and the download is bandwidth-bound, so serialising them
    # would roughly add the two times instead of hiding the smaller one.
    todo = [s for s in shards if not already_done(s)]
    for s in shards:
        if already_done(s):
            with open(sidecar(s)) as fh:
                done = json.load(fh)
            meta.update(done["meta"])
            n_quant += done["n_quant"]
            n_pass += done["n_pass"]
            q_params += done["q_params"]
            q_bytes += done["q_bytes"]
    if len(todo) < len(shards):
        print(f"# resuming: {len(shards) - len(todo)} shards already done, "
              f"{len(todo)} to go ({q_params/1e9:.2f}B params carried over)")
    if not todo:
        shards = []

    pool = ThreadPoolExecutor(max_workers=1)
    pending = pool.submit(fetch, args.model, todo[0]) if todo else None

    for si, shard in enumerate(todo):
        path = pending.result()
        if si + 1 < len(todo):
            pending = pool.submit(fetch, args.model, todo[si + 1])
        shard_meta: dict[str, dict] = {}
        shard_q = shard_p = 0
        shard_params = shard_bytes = 0
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                shape = f.get_slice(name).get_shape()
                if not is_quantisable(name, shape, exclude, args.quantize_head):
                    t = f.get_tensor(name)
                    # Cast what we do not quantise. A 2-bit artifact that stores
                    # its embedding table in fp32 because the source checkpoint
                    # was fp32 spends more on two tensors than on every weight it
                    # compressed: on TinyLlama the embedding and head were 51.6%
                    # of the artifact against 47.6% for all the codes. It also
                    # makes any disk comparison meaningless, since the methods we
                    # are measured against load at fp16 and therefore never pay
                    # it. "preserve" keeps the old behaviour.
                    if args.passthrough_dtype != "preserve" and t.is_floating_point():
                        t = t.to(getattr(torch, args.passthrough_dtype))
                    tensors[name] = t
                    n_pass += 1
                    shard_p += 1
                    n_el = 1
                    for _d in shape:
                        n_el *= _d
                    p_params += n_el
                    continue

                w = f.get_tensor(name).to(DEV)
                experts = None
                if w.ndim == 3:
                    # Flatten the expert axis into rows. Each expert keeps its own
                    # per-group scales because the grouping runs along the last
                    # axis, so this is E independent quantizations sharing one
                    # tensor rather than one quantization of a stacked thing.
                    experts, out_f, in_f = w.shape
                    w = w.reshape(experts * out_f, in_f)
                else:
                    out_f, in_f = w.shape
                codes, scale = quantize(w.float(), codec, args.passes,
                                        args.group_size, args.seed,
                                        row_block=args.row_block)
                base = name[: -len(".weight")] if name.endswith(".weight") else name
                tensors[base + ".codes"] = codes.cpu()
                tensors[base + ".scale"] = scale.cpu()
                entry = {
                    "in_features": in_f, "out_features": w.shape[0],
                    "group_size": group_size_for(in_f, args.group_size),
                    # Both recorded so the loader neither guesses the parameter
                    # name nor has to infer that the rows are stacked experts.
                    "experts": experts,
                    "param_name": name,
                }
                meta[base] = entry
                shard_meta[base] = entry
                nb = codes.numel() * codes.element_size() + scale.numel() * 2
                q_params += w.shape[0] * in_f
                q_bytes += nb
                n_quant += 1
                shard_params += w.shape[0] * in_f
                shard_bytes += nb
                shard_q += 1
                del w, codes, scale
                gc.collect()
                torch.cuda.empty_cache()

        out_shard = os.path.join(args.out, shard)
        save_file(tensors, out_shard, metadata={"format": "pt"})
        with open(sidecar(shard), "w") as fh:
            json.dump({"meta": shard_meta, "n_quant": shard_q, "n_pass": shard_p,
                       "q_params": shard_params, "q_bytes": shard_bytes}, fh)
        del tensors
        gc.collect()

        # Never for a local model: those bytes are the user's own download
        # sitting where they put it, not a cache entry this script created, and
        # deleting them would destroy the input rather than reclaim a copy of it.
        if not args.keep_shards and not local:
            drop_shard(path)
        el = time.time() - t0
        print(f"[{si+1}/{len(todo)}] {shard}: {n_quant} quantised, "
              f"{n_pass} passthrough, {q_params/1e9:.2f}B params, {el/60:.1f} min",
              flush=True)

    cfg = {
        "method": "progressive-density-matched-quantisation",
        "stage": args.stage,
        "passes": args.passes,
        "bits": args.bits,
        "group_size": args.group_size,
        "seed": args.seed,
        "row_block": args.row_block,
        "source_model": args.model,
        "exclude": args.exclude,
        "passthrough_dtype": args.passthrough_dtype,
        "quantize_head": bool(args.quantize_head),
        "trellis_bits": args.trellis_bits if args.stage == "trellis" else None,
        # The 1MAD-p exponent. Recorded for the same reason as codec_scale below:
        # the codeword table is rebuilt from parameters at decode time, so a
        # shape the decoder cannot see is a table it silently gets wrong, and
        # every weight is then scaled by a codebook that never encoded it.
        "trellis_shape": (float(args.trellis_shape)
                          if args.stage == "trellis" else None),
        # E8P's single global scale, recorded rather than left to be re-derived.
        # It is fitted against probes from `torch.Generator(device=...)`, and the
        # CPU and CUDA streams differ, so a decoder that rebuilds the codec on
        # the other device class gets a kappa 0.2% off and scales every weight by
        # that factor with nothing raising. Writing it here removes the
        # dependence on reproducing an RNG stream. `None` for the scalar stage,
        # whose levels are solved on a CPU float64 grid and are device-free.
        "codec_scale": getattr(codec, "scale", None),
        # Which rotation implementation actually ran. The matmul and Walsh-Hadamard
        # paths agree only to ~1.4e-6, and which one a group size takes depends on
        # whether the compiled kernel is importable, so it is a property of the
        # encoding machine rather than of the artifact. Recording it lets a decoder
        # check instead of assume.
        "rotation_path": _rotation_path(args.group_size),
        "effective_bits": effective_bits(args.passes, args.group_size, args.bits),
        "measured_bits_per_weight": (q_bytes * 8 / q_params) if q_params else None,
        "quantised_tensors": meta,
    }
    with open(os.path.join(args.out, "progressive_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"# done in {(time.time()-t0)/60:.1f} min")
    total = q_params + p_params
    cover = 100.0 * q_params / total if total else 0.0
    print(f"# quantised {q_params/1e9:.2f}B of {total/1e9:.2f}B params "
          f"({cover:.1f}%) over {n_quant} tensors")
    # A quantizer that silently compresses a small fraction of the model reports
    # a perfect bits-per-weight for the part it touched and says nothing about
    # the rest. On Qwen3.5-35B a 2D-only rule covered 6.3% -- the multi-token
    # prediction head -- while the stacked-expert tensors that are 90% of the
    # model passed straight through, and every downstream number looked fine.
    if cover < 80.0:
        print(f"### WARNING: only {cover:.1f}% of parameters were quantised.")
        print("###   Expect embeddings and norms to pass through (a few percent);")
        print("###   anything more means a tensor layout the rule did not match.")
    if q_params:
        print(f"# measured {q_bytes*8/q_params:.4f} bits/weight "
              f"(claimed {cfg['effective_bits']:.4f})")


if __name__ == "__main__":
    main()
