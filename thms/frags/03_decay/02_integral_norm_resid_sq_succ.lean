
/-- **Proposition 2, one pass.**  `E‖r_{p+1}‖² = ρ² · E‖r_p‖²`. -/
theorem integral_norm_resid_sq_succ [IsProbabilityMeasure μ]
    (h : IsotropicPasses S W ρ μ) (p : ℕ) :
    ∫ ω, ‖resid (S ω) (W ω) (p + 1)‖ ^ 2 ∂μ = ρ ^ 2 * ∫ ω, ‖resid (S ω) (W ω) p‖ ^ 2 ∂μ := by
  -- The deterministic per-pass identity `‖r_{p+1}‖² = ‖r_p‖² · ‖u_p - q u_p‖²` (§1),
  -- then independence to split the expectation of the product, then `distortion p`.
  have hpt : ∀ ω, ‖resid (S ω) (W ω) (p + 1)‖ ^ 2
      = ‖resid (S ω) (W ω) p‖ ^ 2 * ‖unitVec (S ω) (W ω) p - (S ω).q (unitVec (S ω) (W ω) p)‖ ^ 2 :=
    fun ω => norm_resid_succ_sq (S ω) (W ω) p
  simp_rw [hpt]
  rw [(h.indep p).integral_fun_mul_eq_mul_integral
      (h.integrable p).aestronglyMeasurable (h.integrable_err p).aestronglyMeasurable,
    h.distortion p, mul_comm]
