
/-- The quadrant integrand: `ρ^a s^b · g(ρ/√(ρ²+s²)) · 1[ρ²+s²<1]`. -/
noncomputable def quadFun (a b : ℕ) (g : ℝ → ℝ≥0∞) (p : ℝ × ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (p.1 ^ a) * (ENNReal.ofReal (p.2 ^ b) *
    (g ((√(p.1 ^ 2 + p.2 ^ 2))⁻¹ * p.1) * (Set.Iio (1 : ℝ)).indicator 1 (p.1 ^ 2 + p.2 ^ 2)))
