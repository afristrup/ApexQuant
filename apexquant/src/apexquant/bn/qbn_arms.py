"""The seven arms of the accuracy experiment of section 7.3, and their labels."""

from __future__ import annotations

from .qbn_modes import MODE_MEAN, MODE_MEAN_VAR, MODE_VAR

ARM_FP32 = 1
ARM_QUANTIZED = 2
ARM_QBN_MEAN = 3
ARM_QBN_MEAN_VAR = 4
ARM_QBN_VAR = 5
ARM_REESTIMATED = 6
ARM_FOLDED = 7

ARM_NUMBERS = (ARM_FP32, ARM_QUANTIZED, ARM_QBN_MEAN, ARM_QBN_MEAN_VAR,
               ARM_QBN_VAR, ARM_REESTIMATED, ARM_FOLDED)

ARM_NAMES = {
    ARM_FP32: "fp32",
    ARM_QUANTIZED: "quantized",
    ARM_QBN_MEAN: "+ QBN mean",
    ARM_QBN_MEAN_VAR: "+ QBN mean + variance",
    ARM_QBN_VAR: "+ QBN variance only (control)",
    ARM_REESTIMATED: "+ BN re-estimated, 2048 unlabelled images (ceiling)",
    ARM_FOLDED: "+ BN folded before quantizing (predicted no-op)",
}

#: Arm 6 reads images and is the ceiling, not the method. The distinction is the
#: one section 9 insists on keeping in its own column.
ARM_DATA_FREE = {
    ARM_FP32: "n/a",
    ARM_QUANTIZED: "yes",
    ARM_QBN_MEAN: "yes",
    ARM_QBN_MEAN_VAR: "yes",
    ARM_QBN_VAR: "yes",
    ARM_REESTIMATED: "no",
    ARM_FOLDED: "yes",
}

#: Which BatchNorm buffers each QBN arm is allowed to write. The three arms are
#: one code path with a flag so they cannot drift apart.
ARM_MODES = {
    ARM_QBN_MEAN: MODE_MEAN,
    ARM_QBN_MEAN_VAR: MODE_MEAN_VAR,
    ARM_QBN_VAR: MODE_VAR,
}

#: ``passes`` of the fp32 arm, which has no ``P``. It is a table coordinate, not
#: a pass count, and no arm may be run with it.
FP32_PASSES = 0
