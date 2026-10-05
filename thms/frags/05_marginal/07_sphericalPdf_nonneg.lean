
theorem sphericalPdf_nonneg {d : ℕ} {t : ℝ} (ht : t ∈ Set.Icc (-1 : ℝ) 1) (hd : 2 ≤ d) :
    0 ≤ sphericalPdf d t := by
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have h1 : 0 < Real.Gamma ((d:ℝ)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h2 : 0 < Real.Gamma (((d:ℝ)-1)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h3 : 0 < Real.sqrt π := Real.sqrt_pos.mpr Real.pi_pos
  obtain ⟨h5, h6⟩ := ht
  have h4 : (0:ℝ) ≤ 1 - t ^ 2 := by nlinarith
  exact mul_nonneg (by positivity) (Real.rpow_nonneg h4 _)
