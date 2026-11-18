"""Every remaining LLM number the ICLR table set needs, in one resumable run.

It regenerates `tab:llm-full` and the non-Gemma rows of `tab:kl` with this
paper's harness, so every number in those tables comes from one protocol.

Seven arms, at 4 and 2 bits, all through the SAME protocol so the columns are
comparable: each target Linear weight is quantized and reconstructed in place, and
the model is then scored as a dense model. The arms:

    rtn       per-row absmax, uniform grid, no rotation
    quarot    random orthogonal rotation, absmax, uniform grid
    apex      same rotation, per-row L2 norm, density-matched codebook  (P=1)
    apex-lp   the same stage fitted under HQQ's l_p objective, p=0.7
    apex-e8   the same recursion with a finite E8 lattice stage instead
    apex-tcq  the same recursion with the bitshift trellis stage, L=14
    apex-P2   the progressive construction, 2 passes at 2 bits          (this paper)
    hqq       real HQQ, half-quadratic splitting                        (data-free)
    sinq      real SINQ, Sinkhorn dual-scale                            (data-free)
    asinq     real A-SINQ, SINQ plus AWQ activation-aware scales        (CALIBRATED)

`asinq` is the only arm that reads data, and its budget is recorded in the output
so the comparison against our calibration pass can be stated at matched cost.

Metrics: WikiText-2 raw test perplexity at sequence length 2048, and KL against the
unquantized model on the same windows, reported as the per-token mean and the
99.9th percentile. KL is the paper's primary metric; perplexity cannot separate the
arms because the gaps sit inside its noise.

Sizing. KL needs the unquantized reference resident alongside the reconstructed
arm, so two instances of the model are held at once: about 28 GiB for a 7B at fp16,
plus a transient logit tensor per window. One H200 is ample; a 24 GiB card is not,
which is why this is a pod job.

Every (model, arm, bits) result is written as it completes and skipped if already
present, so a killed pod resumes by re-running.

    HF_HOME=/workspace/hf python experiments/iclr_llm_table.py --out results/iclr_llm

"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerun_apexquant_llm import apex_levels, bos_token_id, rand_orthogonal  # noqa: E402
from progressive_vector_stage_llm import (  # noqa: E402
    make_e8_stage_fast, reconstruct_stage)

from apexquant.vq.trellis_codec import TrellisCodec  # noqa: E402

from apexquant.codebook.uniform_levels import uniform_levels  # noqa: E402
from apexquant.quantize.quantize_to_indices import quantize_to_indices  # noqa: E402

from apexquant.config import ApexQuantConfig  # noqa: E402
from apexquant.quantize.dequantize_weight import dequantize_weight  # noqa: E402
from apexquant.quantize.turboquantize_weight import turboquantize_weight  # noqa: E402

DEV = torch.device("cuda")
SEQ_LEN = 2048
MIN_FANIN = 512           # matches rerun_apexquant_llm's layer selection exactly
GROUP = 512
KL_TAIL = 0.999
# Tokens per slice of the KL reduction; bounds the peak temporary, not the result.
KL_SLICE = 256
# Each baseline's own published default. These are NOT our group size, so the
# effective bits differ per arm and are recorded per arm: at 4 bits, g=64 stores
# 4 + 16/64 = 4.25 bits against our 4 + 16/512 = 4.03. Comparing them as if equal
# would flatter whichever arm carries the smaller group. `--match-group` reruns
# them at our group size for the like-for-like column.
HQQ_GROUP = 64
SINQ_GROUP = 64
ASINQ_SAMPLES = 16        # sinq.awq.get_calib_dataset defaults
ASINQ_SEQLEN = 512
ASINQ_TOKENS = 128       # tokens kept per sample for the AWQ statistic
LP_NORM = 0.7            # HQQ's own exponent, for the apex-lp arm
# Blocks per chunk in the E8 nearest-codeword search. The scoring matrix is
# chunk x 65536 in fp32, so 4096 costs about 1 GiB, which coexists with a
# resident 7B where the module's own default of 8192 does not.
E8_CHUNK = 4096
# Shift-register width of the trellis stage. tab:stages reports rho^2=0.0720
# at L=14, and the fused ACS kernel tops out at 2**14 states, so this is both
# the paper's setting and the widest the encoder accelerates.
TRELLIS_BITS = 14

MODELS = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "Phi-1.5": "microsoft/phi-1_5",
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    # The two models MoBiQuant's main table reports, carried so that a
    # cross-paper row can be read against a baseline both papers implement.
    # Our QuaRot number on these models is the anchor: whatever it differs
    # from their published QuaRot by is the implementation gap, and no
    # ApexQuant-vs-MoBiQuant difference smaller than that is interpretable.
    "LLaMA2-7B": "meta-llama/Llama-2-7b-hf",
    # The base (not Instruct) checkpoint, which is what the published table
    # reports. meta-llama/Meta-Llama-3-8B is gated; the token gained access on
    # 2026-09-14, after this run started, and the mirror is kept because its
    # four safetensors shards have the same LFS sha256 as the official repo's,
    # so nothing needs re-running and nobody needs a gate to reproduce it.
    # Separately: api.model_info() succeeds on a gated repo the token cannot
    # download from, so a visibility check is not an access check.
    "LLaMA3-8B": "NousResearch/Meta-Llama-3-8B",
}
ARMS = ("rtn", "quarot", "apex", "apex-P2", "apex-lp",
        "apex-e8", "apex-e8-P2", "apex-tcq", "apex-tcq-P2",
        "hqq", "sinq", "asinq")


def quantize_weight_grouped(w, method, bits, group=None):
    """Reconstruct a weight the way each baseline is actually reported.

    `rerun_apexquant_llm.quantize_weight` scales by absmax over the entire row.
    At 2048 columns one outlier then sets the grid for the whole row, and 4-bit
    RTN collapses to a perplexity in the tens of thousands. No paper reports RTN
    that way; absmax is reported per group. Scaling per group of `group`
    coordinates instead puts every arm on the same footing as ours, which stores
    one scale per 512 coordinates, so a win here is a win over the baseline as it
    is normally run rather than over a handicapped version of it.
    """
    group = group or GROUP
    orig_dtype, dev = w.dtype, w.device
    r = w.to(DEV, torch.float32)
    out_f, in_f = r.shape
    if method != "rtn":
        M = rand_orthogonal(in_f)
        r = r @ M.T
    g = group if in_f >= group else in_f
    pad = (-in_f) % g
    x = torch.nn.functional.pad(r, (0, pad)).view(out_f, -1, g)
    if method == "apex":
        scale = x.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        levels = apex_levels(bits, g)
    else:
        scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        levels = uniform_levels(bits, device=DEV)
    idx = quantize_to_indices(x / scale, levels)
    recon = (scale * levels[idx.long()]).view(out_f, -1)[:, :in_f]
    if method != "rtn":
        recon = recon @ M
    return recon.to(dev, orig_dtype)


def stored_bits_per_weight(arm, bits, out_f, in_f, group):
    """Exact stored bits per weight for one layer, by each arm's own structure.

    Assuming every arm costs `bits` would flatter whichever one carries the most
    side information, and the differences are large enough to change the ranking.
    Measured, not assumed:

      ours      one fp16 scale per group of `group`, once per pass
      hqq       one fp16 scale AND one fp16 zero per group, so 32/group; at its
                default group of 64 that is 0.50 bits, giving 4.50 not 4.03
      sinq      two vectors, a row and a column scale in fp32, so 32(out+in)/(out*in);
                negligible, which is the overhead claim its paper makes
    """
    passes = 2 if arm.endswith("-P2") else 1
    if arm in ("sinq", "asinq"):
        # Two things, not one. The Sinkhorn dual vectors are the overhead SINQ's
        # own paper quotes and are negligible; the scale and zero that
        # `quantize_rtn` keeps per tile are not, and at the published block of 64
        # they are 32/64 = 0.5 bits, which puts SINQ at 4.50 exactly as HQQ is.
        # Charging only the first is what made this arm look free.
        return (bits + 32.0 / group
                + 32.0 * (out_f + in_f) / (out_f * in_f))
    if arm == "hqq":
        return bits + 32.0 / group
    return passes * (bits + 16.0 / group)


def target_linears(model):
    """The layers every arm quantizes. One rule, so the columns are comparable."""
    out = []
    for name, mod in model.named_modules():
        if (isinstance(mod, torch.nn.Linear) and "layers." in name
                and mod.weight.shape[1] >= MIN_FANIN):
            out.append((name, mod))
    return out


def apply_progressive(mod, bits, passes, codebook="apex"):
    """This paper's arm: P residual passes, reconstructed for evaluation.

    `codebook="lp"` swaps the stage for the same density-matched levels fitted
    under HQQ's l_p objective at p=0.7 instead of squared error. Nothing else
    changes: same rotation, same scale, same recursion, same stored bits. It is
    worse in MSE by construction, so the design chart of tab:stages cannot rank
    it, and an end-to-end score is the only way to learn whether HQQ's claim that
    squared error is the wrong loss survives a random rotation.
    """
    w = mod.weight.data
    qw = turboquantize_weight(
        w.to(DEV, torch.float32),
        ApexQuantConfig(bits=bits, passes=passes, group_size=GROUP,
                         codebook=codebook, lp_norm=LP_NORM))
    mod.weight.data = dequantize_weight(qw).to(w.dtype)


def apply_vector_stage(mod, bits, passes, stage):
    """The same recursion with a vector stage in place of the scalar one.

    tab:stages says the E8 lattice cuts the per-pass distortion from 0.1171 to
    0.0906 at two bits, and cor:optimal is stage-agnostic, so the recursion is
    unchanged: fresh rotation per pass, per-group L2 normalization, peel. Only
    the one line that assigns a codeword differs. The stage is built once per
    bit-width and reused, since its scale search reads the spherical marginal
    rather than any weight.
    """
    w = mod.weight.data
    recon = reconstruct_stage(w.to(DEV, torch.float32), stage, passes,
                              group_size=GROUP)
    mod.weight.data = recon.to(w.dtype)


def apply_hqq(mod, bits, group=HQQ_GROUP):
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    cfg = BaseQuantizeConfig(nbits=bits, group_size=group, axis=1)
    shell = torch.nn.Linear(mod.in_features, mod.out_features, bias=False)
    shell.weight.data = mod.weight.data.clone()
    hq = HQQLinear(shell, cfg, del_orig=True, compute_dtype=torch.float16, device="cuda")
    mod.weight.data = hq.dequantize().to(mod.weight.dtype)
    del hq


def apply_sinq(mod, bits, awq_scales=None, group=SINQ_GROUP):
    """Real SINQ, and real A-SINQ when awq_scales is given.

    `sinq_fake_quant` does not tile. It reduces with `amax(dim=1)`, so calling it
    on a whole matrix gives one scale and one zero per row across the entire
    fan-in, which is not how SINQ is published: its own default is a block of 64,
    reached through `tiled_fake_quant_rectangle`. The difference is 2.1x in
    squared error at four bits and 1.9x at two, measured on real weights, so
    running it untiled and reporting the result as SINQ would be reporting a
    baseline this harness had weakened. An earlier version of this function did
    exactly that.

    The tiled path vmaps over `scales`, so the AWQ vector is reshaped per tile
    rather than passed whole, and a plain SINQ run passes ones rather than None.
    """
    from sinq.awq import sinq_fake_quant, tiled_fake_quant_rectangle

    w = mod.weight.data.to(DEV, torch.float32)
    scales = torch.ones(w.shape[1], device=DEV) if awq_scales is None else awq_scales.to(DEV)
    # The library asserts the block divides the fan-in. Every target layer here
    # has a power-of-two fan-in, so this only guards the pathological case.
    g = group if w.shape[1] % group == 0 else w.shape[1]
    recon = tiled_fake_quant_rectangle(w, fakequant=sinq_fake_quant,
                                       min_max=[0, 2 ** bits - 1], block=g,
                                       scales=scales)
    mod.weight.data = recon.to(mod.weight.dtype).to(mod.weight.device)


def c4_calibration(tokenizer):
    """A-SINQ's calibration set: C4, not the evaluation corpus."""
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    out, it = [], iter(ds)
    while len(out) < ASINQ_SAMPLES:
        text = next(it)["text"]
        ids = tokenizer(text, return_tensors="pt").input_ids
        if ids.shape[1] < ASINQ_SEQLEN:
            continue
        out.append(ids[:, :ASINQ_SEQLEN])
    return torch.cat(out, dim=0)


def collect_awq_scales(model, tokenizer, bits):
    """Per-input-channel AWQ scales, the only place any arm reads data.

    SINQ ships its own `collect_activations`, but it drives the model in a way
    transformers 5.12 rejects (a rotary-embedding shape error inside the attention
    forward), so the activations are collected here with forward hooks instead and
    handed to SINQ's own `compute_awq_scale` unchanged. The scale search is
    therefore theirs; only the plumbing that feeds it is ours.

    `compute_awq_scale` reduces over the first two axes, so each layer's
    activations are stacked as (samples, tokens, in_features). Tokens are
    subsampled to ASINQ_TOKENS per sample to bound memory: the statistic it needs
    is a per-channel mean and standard deviation, which a few thousand tokens
    already pin down, and keeping all 512 would cost several GiB per model for no
    change in the scales.
    """
    from sinq.awq import compute_awq_scale

    calib = c4_calibration(tokenizer)
    targets = dict(target_linears(model))
    store = {n: [] for n in targets}
    handles = []

    def make_hook(name):
        def hook(_mod, inputs):
            x = inputs[0].detach()
            if x.dim() == 3:
                x = x[:, :ASINQ_TOKENS, :]
            else:
                x = x[:ASINQ_TOKENS, :].unsqueeze(0)
            store[name].append(x.float().cpu())
        return hook

    for name, mod in targets.items():
        handles.append(mod.register_forward_pre_hook(make_hook(name)))
    try:
        for i in range(calib.shape[0]):
            model(calib[i:i + 1].to(DEV))
    finally:
        for h in handles:
            h.remove()

    scales = {}
    for name, mod in targets.items():
        if not store[name]:
            continue
        acts = torch.cat(store[name], dim=0)
        scales[name] = compute_awq_scale(
            mod.weight.data.float(), acts, [0, 2 ** bits - 1], method="sinq").detach().cpu()
        store[name] = []
    del store
    gc.collect(); torch.cuda.empty_cache()
    print(f"#   A-SINQ: {len(scales)} layers calibrated on "
          f"{ASINQ_SAMPLES}x{ASINQ_TOKENS} C4 tokens", flush=True)
    return scales


@torch.no_grad()
def score(model_q, model_ref, ids, max_windows=0, ref_cache=None):
    """Perplexity of the arm, plus KL against the reference on the same windows.

    With `ref_cache` the reference log-probs are read from disk rather than from a
    second resident model. Two 7B instances at fp16 need about 28 GiB and a 24 GiB
    card holds one, so caching the reference pass is what lets a 7B be scored here
    at all. The arithmetic is identical either way, which
    `--ref-cache` on TinyLlama checks against the resident-reference numbers.
    """
    bos = bos_token_id(model_ref if model_ref is not None else model_q)
    n = ids.numel() // SEQ_LEN
    if max_windows:
        n = min(n, max_windows)
    nll, ntok, kls, per_window = 0.0, 0, [], []
    for i in range(n):
        chunk = ids[0, i * SEQ_LEN:(i + 1) * SEQ_LEN].unsqueeze(0).to(DEV)
        if bos is not None:
            chunk = torch.cat([torch.full((1, 1), bos, dtype=chunk.dtype, device=DEV),
                               chunk[:, :-1]], dim=1)
        out = model_q(chunk, labels=chunk)
        nll += out.loss.float().item() * (SEQ_LEN - 1)
        ntok += SEQ_LEN - 1
        lq = torch.log_softmax(out.logits.float(), dim=-1)
        if ref_cache is not None:
            lp = torch.load(os.path.join(ref_cache, f"w{i:04d}.pt"),
                            map_location=DEV).float()
        else:
            lp = torch.log_softmax(model_ref(chunk).logits.float(), dim=-1)
        # Per-token KL(p || q), p the unquantized model. Kept per token so the
        # tail can be reported: the mean hides the tokens that actually break.
        #
        # Summed over slices of the sequence rather than the whole window at
        # once. Written flat, `lp.exp() * (lp - lq)` holds three tensors the
        # size of the logits simultaneously, which at Qwen2.5's 152k vocabulary
        # is 1.16 GiB each and overflows a 24 GiB card on top of a resident 7B.
        # Slicing bounds the temporary to KL_SLICE tokens and changes nothing
        # about the result.
        first = len(kls)
        for t0 in range(0, lp.shape[1], KL_SLICE):
            a = lp[:, t0:t0 + KL_SLICE]
            b = lq[:, t0:t0 + KL_SLICE]
            kls.append((a.exp() * (a - b)).sum(-1).flatten().cpu())
        # The window mean, kept alongside the per-token values. Pooling every
        # token into one vector throws away the block structure, and without it
        # a difference between two arms has no interval around it: the table's
        # closest pair differ by 14%, which is a claim only if the windows say so.
        per_window.append(float(torch.cat(kls[first:]).mean()))
        del lq, lp, out
    k = torch.cat(kls)
    return {
        "ppl": float(torch.tensor(nll / ntok).exp()),
        "kl_mean": float(k.mean()),
        "kl_p999": float(k.quantile(KL_TAIL)),
        "kl_windows": per_window,
        "windows": n,
    }


def write_reference_cache(model, ids, cache_dir, max_windows=0):
    """Store the reference log-probs for every window, fp16, one file per window."""
    os.makedirs(cache_dir, exist_ok=True)
    bos = bos_token_id(model)
    n = ids.numel() // SEQ_LEN
    if max_windows:
        n = min(n, max_windows)
    for i in range(n):
        dest = os.path.join(cache_dir, f"w{i:04d}.pt")
        if os.path.exists(dest):
            continue
        chunk = ids[0, i * SEQ_LEN:(i + 1) * SEQ_LEN].unsqueeze(0).to(DEV)
        if bos is not None:
            chunk = torch.cat([torch.full((1, 1), bos, dtype=chunk.dtype, device=DEV),
                               chunk[:, :-1]], dim=1)
        lp = torch.log_softmax(model(chunk).logits.float(), dim=-1)
        torch.save(lp.half().cpu(), dest)
        del lp
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/iclr_llm")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--bits", nargs="*", type=int, default=[4, 2])
    # Smoke-testing only. A capped run is NOT comparable to the paper's protocol
    # and is written to a separate directory so it can never be mistaken for one.
    ap.add_argument("--max-windows", type=int, default=0)
    # Run HQQ and SINQ at OUR group size as well, so one column of the table is
    # matched on stored bits rather than on each method's own default.
    ap.add_argument("--match-group", action="store_true")
    # The grouped baselines (rtn, quarot, apex) default to OUR group so stored
    # bits match. Published RTN is usually reported at a finer group, where it
    # costs more side information but survives outlier channels, so this lets
    # the same harness produce that column instead of only the matched one.
    ap.add_argument("--baseline-group", type=int, default=GROUP)
    # Cache the reference pass here and hold one model at a time. Required for 7B
    # on a 24 GiB card; optional and numerically identical below that.
    ap.add_argument("--ref-cache", default="")
    args = ap.parse_args()
    if args.max_windows:
        args.out = f"{args.out}-smoke{args.max_windows}"
        print(f"# SMOKE TEST: {args.max_windows} windows only, not a paper number")
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    for mname in args.models:
        repo = MODELS[mname]
        path = os.path.join(args.out, f"{mname.replace('/', '_')}.json")
        done = json.load(open(path)) if os.path.exists(path) else {}

        # Every arm that spends its rate as residual passes is defined at two
        # bits per coordinate and reaches the four-bit class with two passes, so
        # asking for it at b=4 is a category error rather than a configuration.
        # It used to be one too: a default --bits 4 2 run reached apex-e8@4 and
        # died inside the stage constructor, after checkpointing four arms.
        two_bit_only = ("apex-P2", "apex-e8", "apex-e8-P2", "apex-tcq", "apex-tcq-P2")
        want = [(a, b) for a in args.arms for b in args.bits
                if not (a in two_bit_only and b != 2)]
        if all(f"{a}@{b}" in done for a, b in want):
            print(f"# {mname}: complete, skipping", flush=True)
            continue

        print(f"\n### {mname} ({repo})", flush=True)
        tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
        ids = tok("\n\n".join(raw["text"]), return_tensors="pt").input_ids

        def load_fp16():
            return AutoModelForCausalLM.from_pretrained(
                repo, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()

        cache_dir = os.path.join(args.ref_cache, mname) if args.ref_cache else None
        if cache_dir:
            # Phase one: the reference alone, its log-probs to disk, then release
            # it so the arm has the whole card to itself.
            ref = load_fp16()
            write_reference_cache(ref, ids, cache_dir, args.max_windows)
            if "fp16" not in done:
                done["fp16"] = score(ref, ref, ids, args.max_windows)
                json.dump(done, open(path, "w"), indent=2)
                print(f"#   fp16: ppl {done['fp16']['ppl']:.3f}", flush=True)
            del ref
            gc.collect(); torch.cuda.empty_cache()
            ref = None
        else:
            ref = load_fp16()

        arm_model = load_fp16()
        pristine = {n: m.weight.data.detach().clone().cpu()
                    for n, m in target_linears(arm_model)}
        print(f"#   {len(pristine)} target linears, {ids.numel() // SEQ_LEN} windows",
              flush=True)

        if "fp16" not in done:
            done["fp16"] = score(ref, ref, ids, args.max_windows)
            json.dump(done, open(path, "w"), indent=2)
            print(f"#   fp16: ppl {done['fp16']['ppl']:.3f}", flush=True)

        awq_cache = {}
        e8_cache = {}
        tcq_cache = {}
        for arm, bits in want:
            key = f"{arm}@{bits}"
            if key in done:
                continue
            # The "-P2" suffix is the whole difference between the one-pass and
            # two-pass forms of an arm, and stored_bits_per_weight reads it the
            # same way, so the two cannot disagree about what was run.
            passes = 2 if arm.endswith("-P2") else 1
            for n, m in target_linears(arm_model):
                m.weight.data = pristine[n].to(m.weight.device, m.weight.dtype)

            if arm == "asinq" and bits not in awq_cache:
                awq_cache[bits] = collect_awq_scales(arm_model, tok, bits)
            if arm.startswith("apex-tcq") and bits not in tcq_cache:
                # Encoding is Viterbi and therefore dear, but the codeword table
                # is a hash of the state rather than a fitted codebook, so the
                # stage itself is built once and carries no data.
                tcq_cache[bits] = TrellisCodec(
                    d=GROUP, bits=bits, trellis_bits=TRELLIS_BITS, device=DEV)
                print(f"#   trellis stage b={bits}, L={TRELLIS_BITS}", flush=True)
            if arm.startswith("apex-e8") and bits not in e8_cache:
                # The codebook is 2**(8*bits) points in 8 dimensions, so 2 bits
                # per coordinate is 65536 codewords and 4 would be 2**32. The
                # 4-bit class is reached by two 2-bit passes, not by a wider
                # stage, which is also the only form that stays nestable.
                if bits != 2:
                    raise ValueError("the E8 stage is defined at 2 bits per "
                                     f"coordinate, not {bits}")
                e8_cache[bits] = make_e8_stage_fast(bits, GROUP,
                                                    chunk_blocks=E8_CHUNK)
                print(f"#   E8 stage b={bits}: scale {e8_cache[bits].scale:.5f}",
                      flush=True)

            # --match-group now reaches SINQ as well as HQQ. It did not, on the
            # mistaken belief that SINQ's fake-quant path tiles internally; it
            # does not tile at all unless it is asked to, so the group we pass is
            # the group it runs at and the matched budget is real.
            if arm == "hqq":
                arm_group = GROUP if args.match_group else HQQ_GROUP
            elif arm in ("sinq", "asinq"):
                # Same treatment as HQQ: its own published block by default, and
                # ours under --match-group, so the table can show both the arm as
                # its authors run it and the arm at our storage.
                arm_group = GROUP if args.match_group else SINQ_GROUP
            elif arm in ("rtn", "quarot"):
                arm_group = args.baseline_group
            else:
                arm_group = GROUP
            for n, m in target_linears(arm_model):
                if arm in ("rtn", "quarot"):
                    m.weight.data = quantize_weight_grouped(m.weight.data, arm, bits,
                                                            group=arm_group)
                elif arm == "apex":
                    # Through turboquantize_weight, not quantize_weight_grouped.
                    # The latter rotates with a dense Haar matrix over the whole
                    # fan-in, shared with the QuaRot row, where our construction
                    # applies a fresh randomized Hadamard per group of GROUP. The
                    # two agree numerically to within the seed spread, but only
                    # one of them is the recursion the paper describes, and the
                    # E8 and trellis rows already use it. Reporting all three
                    # under one label requires all three to be one pipeline.
                    apply_progressive(m, bits, passes=1)
                elif arm == "apex-P2":
                    apply_progressive(m, bits, passes=2)
                elif arm == "apex-lp":
                    apply_progressive(m, bits, passes=1, codebook="lp")
                elif arm.startswith("apex-e8"):
                    apply_vector_stage(m, bits, passes, e8_cache[bits])
                elif arm.startswith("apex-tcq"):
                    apply_vector_stage(m, bits, passes, tcq_cache[bits])
                elif arm == "hqq":
                    apply_hqq(m, bits, group=arm_group)
                elif arm == "sinq":
                    apply_sinq(m, bits, group=arm_group)
                elif arm == "asinq":
                    apply_sinq(m, bits, awq_cache[bits].get(n), group=arm_group)
            gc.collect(); torch.cuda.empty_cache()

            r = score(arm_model, ref, ids, args.max_windows, ref_cache=cache_dir)
            r["calibration"] = ({"samples": ASINQ_SAMPLES, "seqlen": ASINQ_SEQLEN}
                                if arm == "asinq" else None)
            r["group_size"] = arm_group
            num = sum(m.weight.numel() for _, m in target_linears(arm_model))
            r["effective_bits"] = sum(
                stored_bits_per_weight(arm, bits, m.weight.shape[0], m.weight.shape[1],
                                       arm_group) * m.weight.numel()
                for _, m in target_linears(arm_model)) / num
            done[key] = r
            json.dump(done, open(path, "w"), indent=2)
            print(f"#   {key:12s} {r['effective_bits']:.3f}b  ppl {r['ppl']:>10.3f}  "
                  f"KL {r['kl_mean']:.5f}  KL99.9 {r['kl_p999']:.5f}", flush=True)

        del ref, arm_model, pristine
        gc.collect(); torch.cuda.empty_cache()

    print("\n### done ->", args.out)


if __name__ == "__main__":
    main()
