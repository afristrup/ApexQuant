
private theorem aux_integrable_pdf (hd : 2 ≤ d) :
    Integrable (sphericalPdf d) (volume.restrict (Set.Icc (-1:ℝ) 1)) := by
  rw [aux_restrict_Icc_eq]
  exact (intervalIntegrable_iff_integrableOn_Ioc_of_le (by norm_num)).mp (aux_ii_pdf hd)
