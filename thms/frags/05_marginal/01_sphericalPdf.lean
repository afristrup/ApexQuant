
/-- The claimed marginal density `f_d`. -/
noncomputable def sphericalPdf (d : ℕ) (t : ℝ) : ℝ :=
  Real.Gamma ((d : ℝ) / 2) / (Real.sqrt π * Real.Gamma (((d : ℝ) - 1) / 2))
    * (1 - t ^ 2) ^ (((d : ℝ) - 3) / 2)
