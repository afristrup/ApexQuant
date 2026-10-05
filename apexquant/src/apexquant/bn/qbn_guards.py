"""The three guards of QBN section 4.7, and the reasons a layer is left alone."""

from __future__ import annotations

#: ``running_var <- max(v + d_var, VAR_FLOOR_RATIO * v)``. At 0.25 the divisor's
#: standard deviation can never change by more than 2x. 19 ResNet-50 channels and
#: 696 EfficientNet-B0 channels carry ``running_var < 1e-3``, the regime HFP8
#: flags as having vastly magnified output errors, and an additive term there is
#: dangerous.
VAR_FLOOR_RATIO = 0.25

#: ``|d_mean_c| <= MAX_MEAN_SHIFT * sqrt(v_c)``. At ``P=2`` the measured maximum
#: is 0.97 standard deviations, so this guard should never fire. If it does, the
#: input-moment case assignment for that layer is wrong, which is why the count
#: is reported rather than swallowed.
MAX_MEAN_SHIFT = 3.0

#: Guard 3: skip the layer outright. Reported, never silent.
SKIP_NONE = ""
SKIP_UNCORRECTABLE_CASE = "uncorrectable case"
SKIP_NO_RESIDUAL = "not quantized"
SKIP_NO_BATCHNORM = "no following batchnorm"
SKIP_LAYOUT = "group layout assertion failed"
