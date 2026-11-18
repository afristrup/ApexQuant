"""Every (model, arm, level, group) row of the E1 arrays."""

from __future__ import annotations

from clinical.settings import RUN_LIFTQUANT

from .settings import (ABLATION_GROUP, GROUP, LLMC_GROUP, LQ_RATIOS, MODELS, MOE, MOE_GROUP,
                       WEIGHT_ONLY_GROUPS)


def rung_grid(model: str) -> list[tuple]:
    """``level`` is passes for apex, mixed and the ablation, the lifting ratio for
    LiftQuant, and bits otherwise.

    Mixture-of-experts models get the data-free arms only, at group 64 for the
    grouped baselines since PhiMoE's expert fan-in is 448. GPTQ and AWQ through
    llm-compressor and LiftQuant's Llama layer code have not been checked against
    fused expert tensors, so they are left out rather than run on an untested path.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    if model in MOE:
        rows = [(model, "apex", p, GROUP) for p in (1, 2, 3)]
        rows += [(model, "residual_rtn", p, MOE_GROUP) for p in (1, 2, 3)]
        rows += [(model, "dash", b, GROUP) for b in (2, 3, 4)]
        rows += [(model, a, b, MOE_GROUP) for a in ("hqq", "sinq") for b in (2, 3, 4)]
        return rows + [(model, a, p, GROUP) for a in ("apex_sinkhorn", "apex_l0") for p in (1, 2, 3)]
    rows = [(model, "apex", p, 512) for p in (1, 2, 3, 4)]
    rows += [(model, "mixed", p, 512) for p in (1, 2, 3)]
    rows += [(model, "residual_rtn", p, ABLATION_GROUP) for p in (1, 2, 3, 4)]
    rows += [(model, a, b, g) for a in ("hqq", "sinq") for g in WEIGHT_ONLY_GROUPS
             for b in (2, 3, 4)]
    rows += [(model, "dash", b, GROUP) for b in (2, 3, 4)]
    rows += [(model, a, b, LLMC_GROUP) for a in ("gptq", "awq") for b in (3, 4)]
    if RUN_LIFTQUANT:
        rows += [(model, "liftquant", r, 0) for r in LQ_RATIOS]
    # Appended last, so the indices of every earlier rung are unchanged.
    return rows + [(model, a, p, GROUP) for a in ("apex_sinkhorn", "apex_l0") for p in (1, 2, 3, 4)]
