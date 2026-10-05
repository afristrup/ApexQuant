"""Load a progressive artifact back into a Hugging Face model.

This reconstructs dense weights from the stored codes and evaluates from those,
which is exactly how every perplexity in the paper was measured. It is *not* the
packed-form execution path: that needs a fused kernel for the stage's code
format, which exists for the scalar stage but not yet for real E8P (the existing
`triton_gemv_e8p` benchmarks a deliberately simplified code and says so).

The `passes` argument decodes a prefix. Because prefixes are bit-identical to a
fresh shallower encode, loading a P=3 artifact with `passes=2` gives exactly the
4.06-bit tier -- one stored artifact, three operating points.

    uv run python experiments/load_progressive.py \
        --artifact /mnt/bulk/llama70b/e8p-p3 --passes 2 --ppl
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from apexquant.kernels.triton_gemv_trellis import trellis_window_table
from apexquant.nn.progressive_scalar_linear import ProgressiveScalarLinear
from apexquant.vq.e8p_codec import E8PCodec
from apexquant.vq.progressive_artifact import dequantize
from apexquant.vq.progressive_trellis_linear import ProgressiveTrellisLinear
from apexquant.vq.scalar_codec import ScalarCodec
from apexquant.vq.trellis_codec import TrellisCodec

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_codec(cfg: dict, device=DEV):
    """Rebuild the stage quantiser described by a `progressive_config.json`."""
    gs, stage = cfg["group_size"], cfg["stage"]
    if stage == "scalar":
        return ScalarCodec(d=gs, bits=cfg["bits"], device=device)
    if stage == "e8p":
        return E8PCodec(d=gs, seed=cfg["seed"], device=device)
    if stage == "trellis":
        # seed 0 for the codeword hash, which is the stage whose rho the paper
        # reports; the config's `seed` is the rotation's and is unrelated.
        return TrellisCodec(d=gs, bits=cfg["bits"],
                            trellis_bits=cfg.get("trellis_bits") or 12,
                            shape=cfg.get("trellis_shape") or 2.0,
                            seed=0, device=device)
    raise ValueError(f"unknown stage {stage!r}")


def _slot_resolver(slots):
    """Map artifact tensor names onto this transformers version's parameter names.

    An artifact records the names the model had when it was written, and those
    move. Gemma-3 is the case in hand: the transformers 5 refactor made two
    independent changes, and between them not one name in 1401 survives, so the
    loader refuses the artifact outright rather than loading it wrongly --
    correct, but it means nine of the published checkpoints cannot be read at all
    by a current install.

        language_model.model.layers.0.mlp.down_proj
     -> model.language_model.layers.0.mlp.down_proj          `model` moved up

        vision_tower.vision_model.embeddings.patch_embedding
     -> model.vision_tower.embeddings.patch_embedding        `vision_model` went

    What moved in both cases is a container level, not the module tree under it,
    and the container levels of a nested architecture are exactly the components
    called `model` or `<something>_model`. Dropping those from both sides puts
    the two vocabularies in the same terms. A plain Llama normalises to itself
    and matches exactly anyway, so nothing here is Gemma-specific.

    The rule is deliberately loose, and what makes that safe is that a wrong
    answer is impossible rather than unlikely. The normalisation is consulted
    only when the exact name misses; it is used only for keys whose normal form
    names exactly one parameter, so any ambiguity falls back to exact matching
    rather than a guess; and the unfilled-slot check downstream still accounts
    for every slot the model declares. The worst case is a refusal to load.

    Args:
        slots: The model's ``named_parameters()`` keys.

    Returns:
        ``(resolve, n_ambiguous)`` where ``resolve(name)`` gives the parameter
        name to write, or None if the artifact tensor has no home.
    """
    by_norm: dict[str, list[str]] = {}
    for s in slots:
        by_norm.setdefault(_drop_model_levels(s), []).append(s)
    unique = {k: v[0] for k, v in by_norm.items() if len(v) == 1}

    def resolve(name: str) -> str | None:
        return name if name in slots else unique.get(_drop_model_levels(name))

    return resolve, sum(1 for v in by_norm.values() if len(v) > 1)


def _drop_model_levels(name: str) -> str:
    """Strip container levels: `model` and any `<something>_model`."""
    return ".".join(p for p in name.split(".")
                    if p != "model" and not p.endswith("_model"))


def load_progressive(artifact: str, passes: int | None = None,
                     device: str | torch.device | None = None,
                     dtype=torch.float16, fused: bool = False):
    """Materialise the model with progressively-quantised weights.

    The skeleton is built from config and every tensor comes from the artifact,
    so the dense source checkpoint is never downloaded or materialised. Decoding
    runs on the GPU regardless; only the assembled model lands on `device`.

    Args:
        artifact: Directory written by `quantise_hf_progressive.py`.
        passes: Decode depth; defaults to the artifact's full depth.
        device: Where to place the assembled model. Defaults to CUDA when the
            model fits a single card. Pass `"cpu"` to keep it in host memory,
            which is what a 70B model needs before an accelerate dispatch.
        dtype: Weight dtype after reconstruction.
        fused: Keep the codes packed and swap each quantised `nn.Linear` for a
            `ProgressiveTrellisLinear`, which decodes inside the mat-vec. Trellis
            artifacts only -- the scalar and E8P stages have no packed-form
            kernel with the pass axis folded in. Resident memory becomes the size
            of the codes rather than of the fp16 model, which is the difference
            between a 12B fitting on one 24 GB card and not.

    Returns:
        `(model, tokenizer, config)`.
    """
    with open(os.path.join(artifact, "progressive_config.json")) as fh:
        cfg = json.load(fh)
    P = cfg["passes"] if passes is None else passes
    if P > cfg["passes"]:
        raise ValueError(f"artifact stores {cfg['passes']} passes, asked for {P}")
    if fused and cfg["stage"] not in ("trellis", "scalar"):
        raise ValueError(f"--fused covers the trellis and scalar stages; this "
                         f"artifact is {cfg['stage']!r}. E8P has a packed kernel "
                         "(`triton_gemv_e8p_real`) but no pass axis folded into "
                         "it yet, so there is nothing to dispatch to.")
    codec = load_codec(cfg)

    src = cfg["source_model"]
    # Build the skeleton from config, never from the source weights. The
    # artifact holds every tensor the model needs -- passthrough ones copied
    # verbatim, the rest as codes -- so pulling the dense fp16 checkpoint just to
    # overwrite it would cost 141 GB on a 70B model for no benefit.
    # `init_empty_weights` puts *parameters* on the meta device but leaves
    # buffers real, which matters: RoPE's `inv_freq` is computed from config,
    # not stored, and `to_empty()` would silently replace it with garbage.
    hf_cfg = AutoConfig.from_pretrained(artifact)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(hf_cfg)
    model.eval()
    tok = AutoTokenizer.from_pretrained(artifact)

    target = str(DEV if device is None else device)
    shards = sorted(f for f in os.listdir(artifact) if f.endswith(".safetensors"))
    # Only parameters live on meta and need populating; buffers are already real.
    slots = dict(model.named_parameters())
    resolve, n_ambiguous = _slot_resolver(slots)
    filled: set[str] = set()
    n_q = n_p = 0
    n_renamed = 0
    renamed_eg = ""
    missing: list[str] = []
    packed: dict[str, tuple] = {}
    for shard in shards:
        with safe_open(os.path.join(artifact, shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(".scale"):
                    continue                      # consumed with its .codes
                if key.endswith(".codes"):
                    base = key[: -len(".codes")]
                    info = cfg["quantised_tensors"][base]
                    # Stacked-expert tensors carry no ".weight" suffix, so the
                    # writer records the parameter name rather than leaving the
                    # loader to reconstruct it.
                    want = info.get("param_name") or (base + ".weight")
                    name = resolve(want)
                    if info.get("experts") and fused:
                        raise ValueError(
                            f"{want} is a stacked-expert tensor; --fused swaps "
                            "nn.Linear modules and there is none to swap. Run "
                            "the reconstructing path for MoE checkpoints.")
                    if name is not None and name != want:
                        n_renamed += 1
                        renamed_eg = renamed_eg or f"{want} -> {name}"
                    if fused:
                        # Stash rather than swap now: the layer's bias arrives
                        # through the passthrough branch below, possibly from a
                        # later shard, and it has to be read off the original
                        # module before that module is replaced.
                        if name is None:
                            missing.append(want)
                            continue
                        packed[name[: -len(".weight")]] = (
                            base,
                            f.get_tensor(key)[:P].to(target),
                            f.get_tensor(base + ".scale")[:P].to(target).float(),
                        )
                        filled.add(name)
                        n_q += 1
                        continue
                    codes = f.get_tensor(key)[:P].to(DEV)
                    scale = f.get_tensor(base + ".scale")[:P].to(DEV)
                    # Blocked: bit-identical, and it keeps the accumulator off
                    # the peak. It is not enough to make a 12B fit dense on a
                    # 24 GB card -- 20.0 GiB of reconstructed weights plus 2.8
                    # of passthrough leaves under a gigabyte, and the concat
                    # still materialises each layer in full before the cast --
                    # but it does buy back the intermediate.
                    w = dequantize(codes, scale, codec, info["in_features"],
                                   cfg["group_size"], cfg["seed"], passes=P,
                                   row_block=1024)
                    if info.get("experts"):
                        w = w.reshape(info["experts"], -1, info["in_features"])
                    if name is None:
                        missing.append(want)
                    else:
                        # `dtype` must be named. Left implicit, accelerate casts
                        # the value back to the *existing* parameter's dtype, and
                        # the skeleton was built by `from_config` at the default
                        # dtype (fp32), so every `.to(dtype)` here was silently
                        # undone and the whole model ran in fp32.
                        set_module_tensor_to_device(
                            model, name, target, value=w, dtype=dtype)
                        filled.add(name)
                        n_q += 1
                    del codes, scale, w
                    torch.cuda.empty_cache()
                else:
                    slot = resolve(key)
                    if slot is None:
                        missing.append(key)
                        continue
                    if slot != key:
                        n_renamed += 1
                        renamed_eg = renamed_eg or f"{key} -> {slot}"
                    set_module_tensor_to_device(
                        model, slot, target, value=f.get_tensor(key), dtype=dtype)
                    filled.add(slot)
                    n_p += 1

    # Re-tie before checking. The check below already excuses `lm_head.weight`
    # on a tied model, on the grounds that it shares storage with the embedding
    # and so is never written directly -- but `set_module_tensor_to_device`
    # *replaces* the embedding tensor rather than filling it in place, which
    # breaks the sharing it was excused for. The head is then left holding
    # whatever the empty skeleton had, and the model returns logits two to three
    # orders of magnitude too small instead of failing. Nothing catches that
    # downstream, so re-establish the tie explicitly.
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()

    # The skeleton is built empty, so an unfilled slot holds uninitialised
    # memory and would evaluate as garbage rather than raise. Account for every
    # slot the model declares. Tied weights legitimately share storage and so
    # may never be written directly.
    unfilled = sorted(set(slots) - filled)
    tied = set(getattr(model.config, "tie_word_embeddings", False)
               and {"lm_head.weight"} or set())
    real_unfilled = [k for k in unfilled if k not in tied]
    if real_unfilled:
        raise RuntimeError(
            f"{len(real_unfilled)} model tensors were not populated from the "
            f"artifact, e.g. {real_unfilled[:5]}. Refusing to return a model "
            "with uninitialised weights.")

    # Swap in the fused layers only now that every slot is accounted for: the
    # check above is what catches a bias that never arrived, and replacing the
    # module first would hide it by removing the slot.
    if fused:
        trellis = cfg["stage"] == "trellis"
        table = trellis_window_table(codec) if trellis else None
        for mod_path, (base, codes, sc) in packed.items():
            info = cfg["quantised_tensors"][base]
            gs = info.get("group_size", cfg["group_size"])
            parent_name, _, attr = mod_path.rpartition(".")
            parent = model.get_submodule(parent_name)
            old_bias = getattr(getattr(parent, attr), "bias", None)
            bias = None if old_bias is None else old_bias.detach().to(dtype)
            if trellis:
                words = (codes.reshape(P, codes.shape[1], -1)
                         .contiguous().view(torch.int32))
                layer = ProgressiveTrellisLinear(
                    words, sc, table, codec, info["in_features"], gs,
                    cfg["seed"], bias)
            else:
                layer = ProgressiveScalarLinear.from_artifact(
                    codes, sc, codec.levels, info["in_features"], gs,
                    cfg["seed"], cfg["bits"], bias)
            setattr(parent, attr, layer)
        packed.clear()

    print(f"# loaded {n_q} quantised{' (fused)' if fused else ''} + {n_p} "
          f"passthrough tensors at P={P} "
          f"({P * (cfg['bits'] + 16.0 / cfg['group_size']):.4f} bits/weight) "
          f"on {target}")
    if n_renamed:
        # Never silent: the artifact was written against a different module tree,
        # and which one it was is the sort of thing that has to be visible in a
        # log when a number later looks off.
        print(f"# NOTE: {n_renamed} artifact tensors were matched by name after "
              f"dropping `model` path levels,\n#   e.g. {renamed_eg}. The "
              "artifact predates this transformers layout; every tensor still "
              "went\n#   to exactly one slot, and the unfilled-slot check above "
              "covers the rest.")
    if n_ambiguous:
        print(f"# NOTE: {n_ambiguous} model slots share a normalised name and "
              "so were matched by exact name only.")
    if missing:
        print(f"# WARNING: {len(missing)} artifact tensors had no model slot, "
              f"e.g. {missing[:3]}")
    return model, tok, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--ppl", action="store_true", help="WikiText-2 perplexity")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device", default=None,
                    help="where to place the assembled model; 'cpu' for models "
                         "too large for one card")
    ap.add_argument("--fused", action="store_true",
                    help="keep trellis codes packed and decode inside the "
                         "kernels instead of reconstructing dense weights")
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "bfloat16"),
                    help="Gemma needs bfloat16: its hidden states pass 6e5, an "
                         "order of magnitude beyond fp16, so the default "
                         "overflows to NaN part-way up the stack")
    args = ap.parse_args()

    model, tok, cfg = load_progressive(args.artifact, args.passes, args.device,
                                       dtype=getattr(torch, args.dtype),
                                       fused=args.fused)
    if args.ppl:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from datasets import load_dataset  # noqa: E402
        from rerun_apexquant_llm import perplexity  # noqa: E402
        test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        enc = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
        print(f"wikitext2 ppl = {perplexity(model, (enc,), args.seq_len):.3f}")


if __name__ == "__main__":
    main()
