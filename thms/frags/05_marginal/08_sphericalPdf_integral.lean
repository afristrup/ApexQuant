
/-- `f_d` is a probability density: a Beta integral. -/
theorem sphericalPdf_integral (hd : 2 ≤ d) :
    ∫ t in (-1 : ℝ)..1, sphericalPdf d t = 1 := by
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have ha : (-1:ℝ) < ((d:ℝ) - 3) / 2 := by linarith
  have h1 : 0 < Real.Gamma ((d:ℝ)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h2 : 0 < Real.Gamma (((d:ℝ)-1)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h3 : 0 < Real.sqrt π := Real.sqrt_pos.mpr Real.pi_pos
  have hI := aux_I (((d:ℝ) - 3) / 2) ha
  rw [show ((d:ℝ) - 3) / 2 + 1 = ((d:ℝ) - 1) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 3/2 = (d:ℝ) / 2 by ring] at hI
  simp only [sphericalPdf]
  rw [intervalIntegral.integral_const_mul, hI]
  field_simp
