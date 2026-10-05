
private theorem aux_ii_pdf (hd : 2 ≤ d) :
    IntervalIntegrable (sphericalPdf d) volume (-1) 1 := by
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have ha : (-1:ℝ) < ((d:ℝ) - 3) / 2 := by linarith
  unfold sphericalPdf
  exact (aux_ii (((d:ℝ) - 3) / 2) ha).const_mul
      (Real.Gamma ((d : ℝ) / 2) / (Real.sqrt π * Real.Gamma (((d : ℝ) - 1) / 2)))
