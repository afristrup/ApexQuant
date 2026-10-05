"""Which of the two BatchNorm buffers a QBN pass is allowed to write."""

from __future__ import annotations

#: Arm 3 of section 7.3: the analytic bias correction of Nagel et al., moved into
#: ``running_mean``. It is the larger half, a median 5.2% of a BatchNorm output
#: standard deviation at ``P=2`` against the variance term's 0.88%.
MODE_MEAN = "mean"

#: Arm 4: both halves. The two variance terms partly cancel, so the variance half
#: ships whole or not at all.
MODE_MEAN_VAR = "mean+var"

#: Arm 5: the variance half alone. This is a control, not a product. If it helps
#: materially on its own, the mechanism is the coherent contraction rather than
#: the mean shift and the paper should be rewritten around ``alpha_c``.
MODE_VAR = "var"

QBN_MODES = (MODE_MEAN, MODE_MEAN_VAR, MODE_VAR)
