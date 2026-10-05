/-! ## §17  Block stages: the vector quadrature for `ρ`

Proposition 2 (§3, §11) never used coordinates.  What *is* tied to the scalar stage is the
evaluation of `ρ²` as `d · E_{f_d}[(t − q t)²]` (§3b).  This section is the same computation for
a stage that quantises `m` blocks of a space `F` at once: `F = ℝ` is the scalar stage,
`F = EuclideanSpace ℝ (Fin 8)` the `E₈` stage, `F = EuclideanSpace ℝ (Fin T)` a trellis stage of
length `T`.  The answer has the same shape, `ρ² = m · E_ν‖x − q_B x‖²` with `ν` the common law of
one block, and `E_ν‖x‖² = 1/m` is again forced by `‖u‖ = 1`.  Only linearity of expectation is
used: the blocks need not be independent, only identically distributed, and §17b derives that
from invariance of the direction law whenever the rotation group contains the block swaps. -/

section Block

variable {m : ℕ} {F : Type*} [NormedAddCommGroup F] [MeasurableSpace F]
