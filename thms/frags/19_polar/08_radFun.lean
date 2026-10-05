
/-- The radial factor after the change of variables. -/
noncomputable def radFun (n : ℕ) (R : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (R ^ n) * (Set.Iio (1 : ℝ)).indicator 1 R
