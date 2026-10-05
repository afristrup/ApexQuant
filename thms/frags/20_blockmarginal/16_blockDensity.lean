
/-- The density of the `U`-block marginal, before normalisation: `(1 − ‖x‖²)^{(m−2)/2}`. -/
noncomputable def blockDensity (m : ℕ) (x : U) : ℝ≥0∞ :=
  ENNReal.ofReal ((1 - ‖x‖ ^ 2) ^ (((m : ℝ) - 2) / 2))
