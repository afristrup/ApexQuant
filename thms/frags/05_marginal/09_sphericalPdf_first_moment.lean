
/-- `f_d` has mean zero (it is even). -/
theorem sphericalPdf_first_moment (hd : 2 ≤ d) :
    ∫ t in (-1 : ℝ)..1, t * sphericalPdf d t = 0 := by
  have hodd : ∀ t : ℝ, (fun s : ℝ => s * sphericalPdf d s) (-t)
      = -((fun s : ℝ => s * sphericalPdf d s) t) := by
    intro t
    simp only [sphericalPdf, neg_mul, neg_inj]
    rw [show (-t) ^ 2 = t ^ 2 by ring]
  have h := intervalIntegral.integral_comp_neg (a := (-1:ℝ)) (b := 1)
    (fun s : ℝ => s * sphericalPdf d s)
  simp only [hodd, intervalIntegral.integral_neg] at h
  norm_num at h
  linarith
