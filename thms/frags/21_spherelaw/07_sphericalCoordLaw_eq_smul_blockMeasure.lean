
/-- `f_{m+1}` is a positive constant times the block density at `U = ℝ`. -/
theorem sphericalCoordLaw_eq_smul_blockMeasure (m : ℕ) (hm : 1 ≤ m) :
    sphericalCoordLaw (m + 1)
      = ENNReal.ofReal (Real.Gamma (((m + 1 : ℕ) : ℝ) / 2) /
          (Real.sqrt Real.pi * Real.Gamma ((((m + 1 : ℕ) : ℝ) - 1) / 2))) • blockMeasure ℝ m := by
  rw [sphericalCoordLaw, blockMeasure, Real.ball_eq_Ioo, zero_sub, zero_add,
    ← Measure.restrict_congr_set Ioo_ae_eq_Icc, ← withDensity_smul _ (measurable_blockDensity m)]
  congr 1
  funext t
  have hm' : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hc : 0 ≤ Real.Gamma (((m + 1 : ℕ) : ℝ) / 2) /
      (Real.sqrt Real.pi * Real.Gamma ((((m + 1 : ℕ) : ℝ) - 1) / 2)) :=
    div_nonneg (Real.Gamma_pos_of_pos (by positivity)).le
      (mul_nonneg (Real.sqrt_nonneg _) (Real.Gamma_pos_of_pos (by push_cast; linarith)).le)
  rw [Pi.smul_apply, smul_eq_mul, blockDensity_real, sphericalPdf,
    ← ENNReal.ofReal_mul hc]
  congr 2
  push_cast
  ring_nf
