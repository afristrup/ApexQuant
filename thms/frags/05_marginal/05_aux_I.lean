
private theorem aux_I (a : ℝ) (ha : -1 < a) :
    ∫ t in (-1:ℝ)..1, (1 - t ^ 2) ^ a
      = Real.sqrt π * Real.Gamma (a + 1) / Real.Gamma (a + 3/2) := by
  have hsub := intervalIntegral.smul_integral_comp_mul_add (a := (0:ℝ)) (b := 1)
    (f := fun t => (1 - t^2) ^ a) 2 (-1)
  simp only [smul_eq_mul] at hsub
  norm_num at hsub
  have hB : ∫ x in (0:ℝ)..1, x ^ a * (1 - x) ^ a
      = Real.Gamma (a+1) * Real.Gamma (a+1) / Real.Gamma (2*a+2) := by
    have := aux_beta_real (a+1) (a+1) (by linarith) (by linarith)
    simpa [add_sub_cancel_right, show a+1+(a+1) = 2*a+2 by ring] using this
  have hcong : ∫ x in (0:ℝ)..1, (1 - (2*x + -1)^2) ^ a
      = (4:ℝ)^a * ∫ x in (0:ℝ)..1, x^a * (1-x)^a := by
    rw [← intervalIntegral.integral_const_mul]
    refine intervalIntegral.integral_congr fun x hx => ?_
    rw [Set.uIcc_of_le zero_le_one] at hx
    have h1 : (0:ℝ) ≤ x := hx.1
    have h2 : (0:ℝ) ≤ 1 - x := by linarith [hx.2]
    rw [show 1 - (2*x + -1)^2 = 4 * (x * (1-x)) by ring,
      Real.mul_rpow (by norm_num) (by positivity), Real.mul_rpow h1 h2]
    try ring
  rw [← hsub, hcong, hB]
  have hg1 : 0 < Real.Gamma (a+1) := Real.Gamma_pos_of_pos (by linarith)
  have hg2 : 0 < Real.Gamma (a+3/2) := Real.Gamma_pos_of_pos (by linarith)
  have hg3 : 0 < Real.Gamma (2*a+2) := Real.Gamma_pos_of_pos (by linarith)
  have hdup : Real.Gamma (a+1) * Real.Gamma (a+3/2)
      = Real.Gamma (2*a+2) * (2:ℝ)^(-1-2*a) * Real.sqrt π := by
    have h := Real.Gamma_mul_Gamma_add_half (a+1)
    rw [show a+1+1/2 = a+3/2 by ring, show (1:ℝ)-2*(a+1) = -1-2*a by ring,
      show 2*(a+1) = 2*a+2 by ring] at h
    exact h
  have h4 : 2 * (4:ℝ)^a * (2:ℝ)^(-1-2*a) = 1 := by
    rw [show (4:ℝ) = 2 * 2 by norm_num, Real.mul_rpow (by norm_num) (by norm_num),
      mul_assoc, mul_assoc, ← Real.rpow_add (by norm_num : (0:ℝ) < 2),
      ← Real.rpow_add (by norm_num : (0:ℝ) < 2),
      show a + (a + (-1 - 2*a)) = -1 by ring, Real.rpow_neg_one]
    norm_num
  have hkey : 2 * (4:ℝ)^a * (Real.Gamma (a+1) * Real.Gamma (a+1)) * Real.Gamma (a+3/2)
      = Real.sqrt π * Real.Gamma (a+1) * Real.Gamma (2*a+2) := by
    calc 2 * (4:ℝ)^a * (Real.Gamma (a+1) * Real.Gamma (a+1)) * Real.Gamma (a+3/2)
        = 2 * (4:ℝ)^a * Real.Gamma (a+1) * (Real.Gamma (a+1) * Real.Gamma (a+3/2)) := by ring
      _ = 2 * (4:ℝ)^a * Real.Gamma (a+1)
            * (Real.Gamma (2*a+2) * (2:ℝ)^(-1-2*a) * Real.sqrt π) := by rw [hdup]
      _ = (2 * (4:ℝ)^a * (2:ℝ)^(-1-2*a))
            * (Real.sqrt π * Real.Gamma (a+1) * Real.Gamma (2*a+2)) := by ring
      _ = _ := by rw [h4, one_mul]
  rw [show 2 * ((4:ℝ)^a * (Real.Gamma (a+1) * Real.Gamma (a+1) / Real.Gamma (2*a+2)))
      = (2 * (4:ℝ)^a * (Real.Gamma (a+1) * Real.Gamma (a+1))) / Real.Gamma (2*a+2) by ring,
    div_eq_div_iff hg3.ne' hg2.ne']
  linear_combination hkey
