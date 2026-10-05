
/-- **Corollary 1, per pass.**  A codebook with smaller per-coordinate distortion has a smaller
decay constant. -/
theorem rho_le_of_distortion_le {cb₁ cb₂ : ℝ → ℝ} (h : distortion cb₁ ν ≤ distortion cb₂ ν) :
    rho d cb₁ ν ≤ rho d cb₂ ν := by
  unfold rho rhoSq
  exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left h (Nat.cast_nonneg d))
