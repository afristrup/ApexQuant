"""The four input-moment cases of the QBN design document, section 4.4."""

from __future__ import annotations

CASE_A = "A"        # input is act(BN_p(.)), a single preceding BatchNorm
CASE_B = "B"        # input is act(sum of branches), a residual sum
CASE_C = "C"        # input is the image
CASE_D = "D"        # pooling, squeeze-excitation gates, anything else: uncorrected

CASE_LETTERS = (CASE_A, CASE_B, CASE_C, CASE_D)
CORRECTABLE_CASES = (CASE_A, CASE_B)
