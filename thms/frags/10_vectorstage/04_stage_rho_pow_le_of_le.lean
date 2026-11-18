
/-- …and a smaller stage distortion gives a smaller `P`-pass envelope, for every `P`, the
vector-valued form of `rho_pow_le_of_distortion_le`. -/
theorem stage_rho_pow_le_of_le {Q₁ Q₂ : E → E} {σ : Measure E}
    (h : stageDistortion Q₂ σ ≤ stageDistortion Q₁ σ) (P : ℕ) :
    Real.sqrt (stageDistortion Q₂ σ) ^ P ≤ Real.sqrt (stageDistortion Q₁ σ) ^ P :=
  pow_le_pow_left₀ (Real.sqrt_nonneg _) (Real.sqrt_le_sqrt h) P
