
/-- …and it is the least such `P`. -/
theorem le_ceil_of_rho_pow_le {ρ ε : ℝ} {P : ℕ} (hρ0 : 0 < ρ) (hρ1 : ρ < 1) (hε0 : 0 < ε)
    (h : ρ ^ P ≤ ε) : Real.log ε / Real.log ρ ≤ P := by
  have hL : Real.log ρ < 0 := Real.log_neg hρ0 hρ1
  rw [div_le_iff_of_neg hL, ← Real.log_pow]
  exact Real.log_le_log (pow_pos hρ0 P) h
