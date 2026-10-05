
/-- The closed-form normalising mass of the block density. -/
noncomputable def blockMass (B m : ℕ) : ℝ :=
  Real.sqrt π ^ B * Real.Gamma ((m : ℝ) / 2) / Real.Gamma (((B : ℝ) + m) / 2)
