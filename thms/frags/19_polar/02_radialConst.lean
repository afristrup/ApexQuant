
/-- `∫₀¹ rⁿ dr`, in the form the polar formula produces it. -/
noncomputable def radialConst (n : ℕ) : ℝ≥0∞ :=
  ∫⁻ r in Set.Ioi (0 : ℝ), ENNReal.ofReal (r ^ n) * (Set.Iio (1 : ℝ)).indicator 1 r
