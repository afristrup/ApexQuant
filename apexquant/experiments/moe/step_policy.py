"""One E6 row: encode once, allocate, then KL, envelope, controller time and accuracy."""

from __future__ import annotations

import json
import os
import time

import torch

from apexquant.vq.trellis_codec import TrellisCodec
from clinical.grouped_axis import grouped_axis
from clinical.load_causal_lm import load_causal_lm
from clinical.reconstruct_apex import reconstruct_apex
from clinical.target_weights import target_weights
from clinical.token_kl import token_kl
from ladder.model_dir import model_dir

from .allocate_static import allocate_static
from .allocated_experts import AllocatedExperts
from .encode_experts import encode_experts
from .experts_modules import experts_modules
from .gate_statistics import gate_statistics
from .harness_accuracy import harness_accuracy
from .settings import DENSE_PASSES, ENVELOPE_EVERY, ENVELOPE_WINDOWS, GROUP, MODELS, PMAX, TRELLIS_BITS


@torch.no_grad()
def step_policy(work: str, model: str, policy: str, bits: int) -> None:
    """Write ``results/{model}_{policy}_{bits}.json``; dense layers always read two passes.

    The controller share is allocation time over the wall time of the KL pass,
    which also runs the reference model, so it is an upper bound.
    """
    tag = f"{model}_{policy}_{bits}"
    dest = os.path.join(work, "results", f"{tag}.json")
    if os.path.exists(dest):
        print(f"{tag}: done already", flush=True)
        return
    path = model_dir(model, MODELS)
    windows = os.path.join(work, "windows", model)
    candidate, reference = load_causal_lm(path), load_causal_lm(path)
    stage = TrellisCodec(GROUP, bits=2, trellis_bits=TRELLIS_BITS, device="cuda")
    start = time.perf_counter()
    for name, p in target_weights(candidate).items():
        if p.dim() == 2:
            p.data.copy_(grouped_axis(p.data.cuda(), lambda m: reconstruct_apex(
                m, [stage] * DENSE_PASSES, GROUP, seed=0), GROUP))
    modules = experts_modules(candidate)
    reference_modules = experts_modules(reference)
    encoded = {n: encode_experts(m, stage, seed=0) for n, m in modules.items()}
    encode_s = time.perf_counter() - start

    passes = bits / 2
    stats = None
    if policy in ("usage", "oracle"):
        source = "usage" if policy == "usage" else "heldout"
        stats = gate_statistics(reference, torch.load(os.path.join(windows, f"{source}.pt")))
    allocations = {}
    for name, mod in modules.items():
        gate_up, down, u2 = encoded[name]
        count = u2.shape[0]
        kwargs = {}
        if policy == "uniform":
            kwargs["static"] = torch.full((count,), int(passes), dtype=torch.long)
        elif policy in ("usage", "oracle"):
            weight = stats[name]["usage" if policy == "usage" else "energy"].to(u2.device)
            kwargs["static"] = allocate_static(weight, u2, round(passes * count))
        else:
            kwargs["budget"] = round(passes * candidate.config.num_experts_per_tok)
        allocations[name] = kwargs["static"].tolist() if "static" in kwargs else None
        parent, _, leaf = name.rpartition(".")
        setattr(candidate.get_submodule(parent), leaf,
                AllocatedExperts(gate_up, down, u2, mod.act_fn, **kwargs))
    kl_windows = torch.load(os.path.join(windows, "wikitext.pt"))
    forward_start = time.perf_counter()
    kl = token_kl(reference, candidate, kl_windows)
    forward_ms = 1000 * (time.perf_counter() - forward_start)
    controller_ms = sum(m.controller_ms for m in candidate.modules() if isinstance(m, AllocatedExperts))

    probe_names = [n for i, n in enumerate(modules) if i % ENVELOPE_EVERY == 0]
    probed = [candidate.get_submodule(n) for n in probe_names]
    for m, name in zip(probed, probe_names):
        m.reference = (reference_modules[name].gate_up_proj, reference_modules[name].down_proj)
    token_kl(reference, candidate, kl_windows[:ENVELOPE_WINDOWS])
    envelope = torch.cat([torch.cat(m.envelope, dim=1) for m in probed], dim=1)
    for m in probed:
        m.reference, m.envelope = None, []
    del reference
    torch.cuda.empty_cache()

    accuracy = harness_accuracy(candidate, path, os.path.join(work, "lm_eval", tag))
    u2_all = torch.cat([encoded[n][2].cpu() for n in modules])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump({
        "model": model, "policy": policy, "average_bits": bits, "encode_seconds": encode_s,
        "kl": kl, "accuracy": accuracy, "forward_ms": forward_ms, "controller_ms": controller_ms,
        "envelope": {"realized_mean": float(envelope[0].mean()), "triangle_mean": float(envelope[1].mean()),
                     "realized_over_triangle": float((envelope[0] / envelope[1].clamp_min(1e-12)).mean()),
                     "cross_term_ratio": float((envelope[0].square() / envelope[2].clamp_min(1e-24)).mean())},
        "stored_bits_per_expert_weight": (2 + 16 / GROUP) * (
            PMAX if policy == "gate" else sum(sum(a) for a in allocations.values())
            / sum(len(a) for a in allocations.values())),
        "read_bits_per_expert_weight": (2 + 16 / GROUP) * bits / 2,
        "u2_per_pass_mean": u2_all.mean(0).tolist(),
        "allocations": allocations,
    }, open(dest, "w"), indent=1)
