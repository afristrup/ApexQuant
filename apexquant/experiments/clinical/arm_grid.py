"""Every (variant, arm, level, calibration, seed) the E3 array runs."""

from __future__ import annotations

from .settings import APEX_SEEDS, ARM_VARIANTS, CALIB_DRAWS, RUN_LIFTQUANT

CALIBRATIONS = [f"c4_{d}" for d in range(CALIB_DRAWS)] + ["oracle"]


def arm_grid() -> list[tuple]:
    """Apex at one to three passes over five seeds, the rest at three and four bits.

    LiftQuant's lattices stop at three bits (``24to8``); it is calibrated on its
    own 4096-sequence generic set and on a clinical oracle of the same size.
    """
    rows = []
    for variant in ARM_VARIANTS:
        rows += [(variant, "apex", p, "none", s) for p in (1, 2, 3) for s in range(APEX_SEEDS)]
        rows += [(variant, a, b, "none", 0) for a in ("hqq", "sinq", "dash") for b in (3, 4)]
        if RUN_LIFTQUANT:
            rows += [(variant, "liftquant", "24to8", c, 0) for c in ("c4_lq", "oracle_lq")]
        rows += [(variant, a, b, c, 0) for a in ("gptq", "awq") for b in (3, 4)
                 for c in CALIBRATIONS]
    # Appended after every variant, so the indices of the rows above are unchanged.
    rows += [(variant, "apex_sinkhorn", p, "none", 0) for variant in ARM_VARIANTS
             for p in (1, 2, 3)]
    return rows
