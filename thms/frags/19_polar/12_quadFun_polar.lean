
/-- **Pointwise polar identity in the first quadrant.** -/
theorem quadFun_polar (a b : ℕ) (g : ℝ → ℝ≥0∞) {R θ : ℝ} (hR : 0 < R)
    (hθ : θ ∈ Set.Ioo (-Real.pi) Real.pi) :
    ENNReal.ofReal R * (Set.Ioi (0 : ℝ) ×ˢ Set.Ioi (0 : ℝ)).indicator (quadFun a b g)
        (R * Real.cos θ, R * Real.sin θ)
      = radFun (a + b + 1) R * angFun a b g θ := by
  by_cases hq : θ ∈ Set.Ioo 0 (Real.pi / 2)
  · have hcos : 0 < Real.cos θ := Real.cos_pos_of_mem_Ioo ⟨by linarith [hq.1, Real.pi_pos], hq.2⟩
    have hsin : 0 < Real.sin θ :=
      Real.sin_pos_of_pos_of_lt_pi hq.1 (by linarith [hq.2, Real.pi_pos])
    have hmem : (R * Real.cos θ, R * Real.sin θ) ∈ Set.Ioi (0 : ℝ) ×ˢ Set.Ioi (0 : ℝ) :=
      ⟨mul_pos hR hcos, mul_pos hR hsin⟩
    rw [Set.indicator_of_mem hmem, angFun, Set.indicator_of_mem hq, quadFun, radFun, angWeight]
    simp only
    have hsq : (R * Real.cos θ) ^ 2 + (R * Real.sin θ) ^ 2 = R ^ 2 := by
      have := Real.cos_sq_add_sin_sq θ
      nlinarith [this]
    have hsqrt : √((R * Real.cos θ) ^ 2 + (R * Real.sin θ) ^ 2) = R := by
      rw [hsq, Real.sqrt_sq hR.le]
    have hind : (Set.Iio (1 : ℝ)).indicator (1 : ℝ → ℝ≥0∞) ((R * Real.cos θ) ^ 2 + (R * Real.sin θ) ^ 2)
        = (Set.Iio (1 : ℝ)).indicator (1 : ℝ → ℝ≥0∞) R := by
      rw [hsq]
      by_cases h1 : R < 1
      · rw [Set.indicator_of_mem (Set.mem_Iio.mpr (by nlinarith)),
          Set.indicator_of_mem (Set.mem_Iio.mpr h1)]
        rfl
      · have h1' : 1 ≤ R := not_lt.mp h1
        rw [Set.indicator_of_notMem
            (fun h' => absurd (Set.mem_Iio.mp h') (not_lt.mpr (by nlinarith))),
          Set.indicator_of_notMem (fun h' => h1 (Set.mem_Iio.mp h'))]
    rw [hsqrt, hind, inv_mul_cancel_left₀ hR.ne', mul_pow, mul_pow,
      ENNReal.ofReal_mul (pow_nonneg hR.le _), ENNReal.ofReal_mul (pow_nonneg hR.le _),
      ENNReal.ofReal_mul (pow_nonneg hcos.le _), pow_succ, pow_add,
      ENNReal.ofReal_mul (mul_nonneg (pow_nonneg hR.le _) (pow_nonneg hR.le _)),
      ENNReal.ofReal_mul (pow_nonneg hR.le _)]
    ring
  · have hnot : (R * Real.cos θ, R * Real.sin θ) ∉ Set.Ioi (0 : ℝ) ×ˢ Set.Ioi (0 : ℝ) := by
      rintro ⟨h1, h2⟩
      simp only [Set.mem_Ioi] at h1 h2
      have hcos : 0 < Real.cos θ := pos_of_mul_pos_right h1 hR.le
      have hsin : 0 < Real.sin θ := pos_of_mul_pos_right h2 hR.le
      apply hq
      constructor
      · by_contra h
        exact absurd hsin (not_lt.mpr
          (Real.sin_nonpos_of_nonpos_of_neg_pi_le (not_lt.mp h) hθ.1.le))
      · by_contra h
        exact absurd hcos (not_lt.mpr
          (Real.cos_nonpos_of_pi_div_two_le_of_le (not_lt.mp h) (by linarith [hθ.2, Real.pi_pos])))
    rw [Set.indicator_of_notMem hnot, angFun, Set.indicator_of_notMem hq, mul_zero, mul_zero]
