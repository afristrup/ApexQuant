
/-- **Corollary 1, whole envelope.**  …and therefore a smaller error at *every* pass count:
optimality per pass upgrades to optimality of the geometric envelope `ρ^P`.  This is the exact
sense in which "the fixed density-matched codebook is optimal not only per pass but for the
entire geometric envelope". -/
theorem rho_pow_le_of_distortion_le {cb₁ cb₂ : ℝ → ℝ}
    (h : distortion cb₁ ν ≤ distortion cb₂ ν) (P : ℕ) :
    rho d cb₁ ν ^ P ≤ rho d cb₂ ν ^ P := by
  exact pow_le_pow_left₀ (Real.sqrt_nonneg _) (rho_le_of_distortion_le h) P
