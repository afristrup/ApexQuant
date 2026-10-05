
/-! ### The pass-count design rule -/

/-- **`P = ⌈ln ε / ln ρ⌉` passes suffice** to reach relative error `ε`.  With `ρ` computable by
quadrature (§4), this is the paper's data-free design rule: the number of passes is known before
any weight is touched. -/
theorem rho_pow_ceil_le {ρ ε : ℝ} (hρ0 : 0 < ρ) (hρ1 : ρ < 1) (hε0 : 0 < ε) (hε1 : ε ≤ 1) :
    ρ ^ (⌈Real.log ε / Real.log ρ⌉₊) ≤ ε := by
  have hL : Real.log ρ < 0 := Real.log_neg hρ0 hρ1
  set n : ℕ := ⌈Real.log ε / Real.log ρ⌉₊ with hn
  -- `n * log ρ ≤ log ε`, the inequality flipping because `log ρ < 0`.
  have hstep : (n : ℝ) * Real.log ρ ≤ Real.log ε := by
    have := mul_le_mul_of_nonpos_right (Nat.le_ceil (Real.log ε / Real.log ρ)) hL.le
    rwa [div_mul_cancel₀ _ hL.ne] at this
  rw [← Real.log_le_log_iff (pow_pos hρ0 n) hε0, Real.log_pow]
  exact hstep
