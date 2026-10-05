
/-- **Proposition 2.**  `E‖r_P‖² = ρ^{2P} · E‖w‖²` for every `P`. -/
theorem integral_norm_resid_sq [IsProbabilityMeasure μ]
    (h : IsotropicPasses S W ρ μ) (P : ℕ) :
    ∫ ω, ‖resid (S ω) (W ω) P‖ ^ 2 ∂μ = ρ ^ (2 * P) * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  induction P with
  | zero => simp
  | succ P ih =>
      rw [integral_norm_resid_sq_succ h P, ih, Nat.mul_succ, pow_add]
      ring
