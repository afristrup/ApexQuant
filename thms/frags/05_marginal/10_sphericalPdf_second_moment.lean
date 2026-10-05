
/-- `f_d` has variance `1/d`, matching `§3b`'s `second_moment_eq_inv_card`.  That these two
routes agree is the consistency check between Lemma 1 and Proposition 2. -/
theorem sphericalPdf_second_moment (hd : 2 ≤ d) :
    ∫ t in (-1 : ℝ)..1, t ^ 2 * sphericalPdf d t = 1 / (d : ℝ) := by
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have ha : (-1:ℝ) < ((d:ℝ) - 3) / 2 := by linarith
  have h1 : 0 < Real.Gamma ((d:ℝ)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h2 : 0 < Real.Gamma (((d:ℝ)-1)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h3 : 0 < Real.sqrt π := Real.sqrt_pos.mpr Real.pi_pos
  have h5 : 0 < Real.Gamma (((d:ℝ)+2)/2) := Real.Gamma_pos_of_pos (by linarith)
  -- the two Gamma recurrences that make the answer collapse to `1/d`
  have e1 : Real.Gamma (((d:ℝ)+1)/2) = (((d:ℝ)-1)/2) * Real.Gamma (((d:ℝ)-1)/2) := by
    have := Real.Gamma_add_one (s := ((d:ℝ)-1)/2) (by intro h; linarith [h])
    rw [show ((d:ℝ)-1)/2 + 1 = ((d:ℝ)+1)/2 by ring] at this
    exact this
  have e2 : Real.Gamma (((d:ℝ)+2)/2) = ((d:ℝ)/2) * Real.Gamma ((d:ℝ)/2) := by
    have := Real.Gamma_add_one (s := (d:ℝ)/2) (by intro h; linarith [h])
    rw [show (d:ℝ)/2 + 1 = ((d:ℝ)+2)/2 by ring] at this
    exact this
  have hJ := aux_J (((d:ℝ) - 3) / 2) ha
  rw [show ((d:ℝ) - 3) / 2 + 1 = ((d:ℝ) - 1) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 3/2 = (d:ℝ) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 2 = ((d:ℝ) + 1) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 5/2 = ((d:ℝ) + 2) / 2 by ring] at hJ
  have hmul : ∀ t : ℝ, t ^ 2 * sphericalPdf d t
      = (Real.Gamma ((d : ℝ) / 2) / (Real.sqrt π * Real.Gamma (((d : ℝ) - 1) / 2)))
        * (t ^ 2 * (1 - t ^ 2) ^ (((d : ℝ) - 3) / 2)) := by
    intro t; simp only [sphericalPdf]; ring
  simp only [hmul]
  rw [intervalIntegral.integral_const_mul, hJ, e1, e2]
  have hdne : (d:ℝ) ≠ 0 := by linarith
  field_simp
  ring
