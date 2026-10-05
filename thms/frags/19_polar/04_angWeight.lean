
/-- The angular weight `cos^a θ · sin^b θ`. -/
noncomputable def angWeight (a b : ℕ) (θ : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (Real.cos θ ^ a * Real.sin θ ^ b)
