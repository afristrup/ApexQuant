"""Stored bits per weight of one quantized linear layer, side information included."""

from __future__ import annotations

from .settings import ABLATION_GROUP, GROUP


def linear_bits(arm: str, level, group: int, out_f: int, in_f: int) -> float:
    """Bits per weight by each arm's own storage.

    ApexQuant keeps one fp16 norm per group of 512 per pass, so each pass costs
    exactly ``2 + 16 / 512 = 2.03125`` bits (groups run along the fan-out when the
    fan-in does not divide 512); HQQ keeps an fp16 scale and
    zero per group, as does each pass of the residual RTN ablation; SINQ adds its fp32 row and column vectors; GPTQ is symmetric
    (scale only) and AWQ asymmetric (scale plus a ``b``-bit zero). DASHQuant keeps
    one fp16 norm per group, its signs and dither coming from the seed. LiftQuant
    stores one lifted bit per code coordinate plus per-channel scales, counted
    here as one fp16 value per row and per column, which approximates its
    side information.
    """
    side = 16.0 / GROUP
    if arm in ("apex", "apex_l0", "apex_sinkhorn", "mixed", "dash") and in_f % GROUP and out_f % GROUP:
        raise ValueError(f"a {out_f}x{in_f} matrix has no side divisible by {GROUP}")
    if arm in ("apex", "apex_l0"):
        return level * (2 + side)
    if arm == "apex_sinkhorn":
        return level * (2 + side) + 16.0 * (out_f + in_f) / (out_f * in_f)
    if arm == "mixed":
        return 4 + side + (level - 1) * (2 + side)
    if arm == "residual_rtn":
        return level * (2 + 32.0 / ABLATION_GROUP)
    if arm == "dash":
        return level + side
    if arm == "liftquant":
        lifted, kept = (int(x) for x in level.split("to"))
        return lifted / kept + 16.0 * (out_f + in_f) / (out_f * in_f)
    if arm == "hqq":
        return level + 32.0 / group
    if arm == "sinq":
        return level + 32.0 / group + 32.0 * (out_f + in_f) / (out_f * in_f)
    if arm == "gptq":
        return level + 16.0 / group
    if arm == "awq":
        return level + (16.0 + level) / group
    raise ValueError(f"unknown arm {arm!r}")
