
private theorem aux_J (a : ℝ) (ha : -1 < a) :
    ∫ t in (-1:ℝ)..1, t ^ 2 * (1 - t ^ 2) ^ a
      = Real.sqrt π * Real.Gamma (a + 1) / Real.Gamma (a + 3/2)
        - Real.sqrt π * Real.Gamma (a + 2) / Real.Gamma (a + 5/2) := by
  have hcongr : ∫ t in (-1:ℝ)..1, t ^ 2 * (1 - t ^ 2) ^ a
      = ∫ t in (-1:ℝ)..1, ((1 - t ^ 2) ^ a - (1 - t ^ 2) ^ (a + 1)) := by
    refine intervalIntegral.integral_congr fun t ht => ?_
    rw [Set.uIcc_of_le (by norm_num : (-1:ℝ) ≤ 1)] at ht
    have hs : (0:ℝ) ≤ 1 - t ^ 2 := by nlinarith [ht.1, ht.2]
    rcases eq_or_lt_of_le hs with h | h
    · have ht2 : t ^ 2 = 1 := by linarith
      rw [← h, Real.zero_rpow (by linarith : a + 1 ≠ 0), ht2]
      ring
    · rw [Real.rpow_add h a 1, Real.rpow_one]
      ring
  have hI2 := aux_I (a + 1) (by linarith)
  rw [show a + 1 + 1 = a + 2 by ring, show a + 1 + 3/2 = a + 5/2 by ring] at hI2
  rw [hcongr, intervalIntegral.integral_sub (aux_ii a ha) (aux_ii (a+1) (by linarith)),
    aux_I a ha, hI2]
