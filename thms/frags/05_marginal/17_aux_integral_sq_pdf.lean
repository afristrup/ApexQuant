
private theorem aux_integral_sq_pdf (hd : 2 ≤ d) :
    ∫ t, t ^ 2 * sphericalPdf d t ∂(volume.restrict (Set.Icc (-1:ℝ) 1)) = 1 / (d : ℝ) := by
  rw [aux_restrict_Icc_eq, ← intervalIntegral.integral_of_le (by norm_num : (-1:ℝ) ≤ 1)]
  exact sphericalPdf_second_moment hd
