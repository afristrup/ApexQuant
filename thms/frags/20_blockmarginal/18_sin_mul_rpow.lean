
/-- `sin θ · (1 − cos²θ)^{(m−2)/2} = sin^{m−1} θ` on `(0, π/2)`. -/
theorem sin_mul_rpow (m : ℕ) (hm : 1 ≤ m) {θ : ℝ} (hθ : θ ∈ Set.Ioo 0 (Real.pi / 2)) :
    Real.sin θ * (1 - Real.cos θ ^ 2) ^ (((m : ℝ) - 2) / 2) = Real.sin θ ^ (m - 1) := by
  have hs : 0 < Real.sin θ := Real.sin_pos_of_pos_of_lt_pi hθ.1 (by linarith [hθ.2, Real.pi_pos])
  have h1 : (1 - Real.cos θ ^ 2) ^ (((m : ℝ) - 2) / 2) = Real.sin θ ^ ((m : ℝ) - 2) := by
    rw [← Real.sin_sq, ← Real.rpow_two, ← Real.rpow_mul hs.le, mul_div_cancel₀ _ two_ne_zero]
  rw [h1, show Real.sin θ * Real.sin θ ^ ((m : ℝ) - 2)
      = Real.sin θ ^ (1 : ℝ) * Real.sin θ ^ ((m : ℝ) - 2) by rw [Real.rpow_one],
    ← Real.rpow_add hs, show (1 : ℝ) + ((m : ℝ) - 2) = ((m - 1 : ℕ) : ℝ) by
      rw [Nat.cast_sub hm, Nat.cast_one]; ring,
    Real.rpow_natCast]
