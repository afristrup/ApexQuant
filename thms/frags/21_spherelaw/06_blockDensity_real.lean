
/-! ### The scalar case `U = ℝ`: the paper's `f_d` -/

theorem blockDensity_real (m : ℕ) (t : ℝ) :
    blockDensity m t = ENNReal.ofReal ((1 - t ^ 2) ^ (((m : ℝ) - 2) / 2)) := by
  rw [blockDensity, Real.norm_eq_abs, sq_abs]
