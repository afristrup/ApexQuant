/-! ## §37  The fourth moment of `f_d`, and the exact Hadamard gap

§5 computed the second moment of the coordinate law `f_d`; the same Beta integrals give the
fourth, `E_{f_d} t⁴ = 3/(d(d+2))` (`integral_pow_four_sphericalCoordLaw`), from
`t⁴(1 − t²)^a = (1 − t²)^a − 2(1 − t²)^{a+1} + (1 − t²)^{a+2}` and three Gamma recurrences.
With §36 this makes the fourth-moment comparison between the deployed rotation and the Haar
theory exact: for a unit direction `v`,
`E (H D_ε v)ᵢ⁴ − E_{f_d} t⁴ = (6/(d+2) − 2 ∑ⱼ vⱼ⁴)/d²` (`Hadamard.signAvg_coord_pow_four_sub_sphere`),
which lies between `−2/d²` and `4/d³` because `1/d ≤ ∑ⱼ vⱼ⁴ ≤ 1` on the sphere
(`Hadamard.abs_signAvg_coord_pow_four_sub_sphere`).  The quoted value of §36 is now a theorem. -/

variable {d : ℕ}

/-! ### The fourth moment of `f_d` -/

private theorem aux_J4 (a : ℝ) (ha : -1 < a) :
    ∫ t in (-1:ℝ)..1, t ^ 4 * (1 - t ^ 2) ^ a
      = Real.sqrt π * Real.Gamma (a + 1) / Real.Gamma (a + 3/2)
        - 2 * (Real.sqrt π * Real.Gamma (a + 2) / Real.Gamma (a + 5/2))
        + Real.sqrt π * Real.Gamma (a + 3) / Real.Gamma (a + 7/2) := by
  have hcongr : ∫ t in (-1:ℝ)..1, t ^ 4 * (1 - t ^ 2) ^ a
      = ∫ t in (-1:ℝ)..1, ((1 - t ^ 2) ^ a - 2 * (1 - t ^ 2) ^ (a + 1) + (1 - t ^ 2) ^ (a + 2)) := by
    refine intervalIntegral.integral_congr fun t ht => ?_
    rw [Set.uIcc_of_le (by norm_num : (-1:ℝ) ≤ 1)] at ht
    have hs : (0:ℝ) ≤ 1 - t ^ 2 := by nlinarith [ht.1, ht.2]
    rcases eq_or_lt_of_le hs with h | h
    · have ht2 : t ^ 2 = 1 := by linarith
      have ht4 : t ^ 4 = 1 := by rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, ht2]; norm_num
      rw [← h, Real.zero_rpow (by linarith : a + 1 ≠ 0), Real.zero_rpow (by linarith : a + 2 ≠ 0),
        ht4]
      ring
    · rw [Real.rpow_add h a 1, Real.rpow_add h a 2, Real.rpow_one, Real.rpow_two]
      ring
  have hI1 := aux_I (a + 1) (by linarith)
  have hI2 := aux_I (a + 2) (by linarith)
  rw [show a + 1 + 1 = a + 2 by ring, show a + 1 + 3/2 = a + 5/2 by ring] at hI1
  rw [show a + 2 + 1 = a + 3 by ring, show a + 2 + 3/2 = a + 7/2 by ring] at hI2
  rw [hcongr, intervalIntegral.integral_add ((aux_ii a ha).sub ((aux_ii (a+1) (by linarith)).const_mul 2))
    (aux_ii (a+2) (by linarith)), intervalIntegral.integral_sub (aux_ii a ha)
    ((aux_ii (a+1) (by linarith)).const_mul 2), intervalIntegral.integral_const_mul, aux_I a ha,
    hI1, hI2]

/-- `f_d` has fourth moment `3/(d(d+2))`. -/
theorem sphericalPdf_fourth_moment (hd : 2 ≤ d) :
    ∫ t in (-1 : ℝ)..1, t ^ 4 * sphericalPdf d t = 3 / ((d : ℝ) * ((d : ℝ) + 2)) := by
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have ha : (-1:ℝ) < ((d:ℝ) - 3) / 2 := by linarith
  have h1 : 0 < Real.Gamma ((d:ℝ)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h2 : 0 < Real.Gamma (((d:ℝ)-1)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h3 : 0 < Real.sqrt π := Real.sqrt_pos.mpr Real.pi_pos
  have e1 : Real.Gamma (((d:ℝ)+1)/2) = (((d:ℝ)-1)/2) * Real.Gamma (((d:ℝ)-1)/2) := by
    have := Real.Gamma_add_one (s := ((d:ℝ)-1)/2) (by intro h; linarith [h])
    rw [show ((d:ℝ)-1)/2 + 1 = ((d:ℝ)+1)/2 by ring] at this
    exact this
  have e2 : Real.Gamma (((d:ℝ)+2)/2) = ((d:ℝ)/2) * Real.Gamma ((d:ℝ)/2) := by
    have := Real.Gamma_add_one (s := (d:ℝ)/2) (by intro h; linarith [h])
    rw [show (d:ℝ)/2 + 1 = ((d:ℝ)+2)/2 by ring] at this
    exact this
  have e3 : Real.Gamma (((d:ℝ)+3)/2) = (((d:ℝ)+1)/2) * Real.Gamma (((d:ℝ)+1)/2) := by
    have := Real.Gamma_add_one (s := ((d:ℝ)+1)/2) (by intro h; linarith [h])
    rw [show ((d:ℝ)+1)/2 + 1 = ((d:ℝ)+3)/2 by ring] at this
    exact this
  have e4 : Real.Gamma (((d:ℝ)+4)/2) = (((d:ℝ)+2)/2) * Real.Gamma (((d:ℝ)+2)/2) := by
    have := Real.Gamma_add_one (s := ((d:ℝ)+2)/2) (by intro h; linarith [h])
    rw [show ((d:ℝ)+2)/2 + 1 = ((d:ℝ)+4)/2 by ring] at this
    exact this
  have hJ := aux_J4 (((d:ℝ) - 3) / 2) ha
  rw [show ((d:ℝ) - 3) / 2 + 1 = ((d:ℝ) - 1) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 3/2 = (d:ℝ) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 2 = ((d:ℝ) + 1) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 5/2 = ((d:ℝ) + 2) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 3 = ((d:ℝ) + 3) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 7/2 = ((d:ℝ) + 4) / 2 by ring] at hJ
  have hmul : ∀ t : ℝ, t ^ 4 * sphericalPdf d t
      = (Real.Gamma ((d : ℝ) / 2) / (Real.sqrt π * Real.Gamma (((d : ℝ) - 1) / 2)))
        * (t ^ 4 * (1 - t ^ 2) ^ (((d : ℝ) - 3) / 2)) := by
    intro t; simp only [sphericalPdf]; ring
  simp only [hmul]
  rw [intervalIntegral.integral_const_mul, hJ, e4, e3, e2, e1]
  have hdne : (d:ℝ) ≠ 0 := by linarith
  have hd2 : (d:ℝ) + 2 ≠ 0 := by linarith
  field_simp
  ring

private theorem aux_integral_pow_four_pdf (hd : 2 ≤ d) :
    ∫ t, t ^ 4 * sphericalPdf d t ∂(volume.restrict (Set.Icc (-1:ℝ) 1))
      = 3 / ((d : ℝ) * ((d : ℝ) + 2)) := by
  rw [aux_restrict_Icc_eq, ← intervalIntegral.integral_of_le (by norm_num : (-1:ℝ) ≤ 1)]
  exact sphericalPdf_fourth_moment hd

/-- **The fourth moment of the coordinate law is `3/(d(d+2))`.** -/
theorem integral_pow_four_sphericalCoordLaw (hd : 2 ≤ d) :
    ∫ t, t ^ 4 ∂(sphericalCoordLaw d) = 3 / ((d : ℝ) * ((d : ℝ) + 2)) := by
  rw [sphericalCoordLaw,
    integral_withDensity_eq_integral_toReal_smul
      (f := fun t => ENNReal.ofReal (sphericalPdf d t))
      (ENNReal.measurable_ofReal.comp (aux_meas_pdf d))
      (Filter.Eventually.of_forall fun t => ENNReal.ofReal_lt_top)]
  rw [show (∫ t, (ENNReal.ofReal (sphericalPdf d t)).toReal • t ^ 4
        ∂(volume.restrict (Set.Icc (-1:ℝ) 1)))
      = ∫ t, t ^ 4 * sphericalPdf d t ∂(volume.restrict (Set.Icc (-1:ℝ) 1)) from
    setIntegral_congr_fun measurableSet_Icc fun t ht => by
      rw [ENNReal.toReal_ofReal (sphericalPdf_nonneg ht hd), smul_eq_mul, mul_comm]]
  exact aux_integral_pow_four_pdf hd

namespace Hadamard

open Orthogonal

/-- **The fourth-moment gap between the Hadamard and the spherical coordinate law**, exactly:
for a unit direction `v`, `E (H D_ε v)ᵢ⁴ − E_{f_d} t⁴ = (6/(d+2) − 2 ∑ⱼ vⱼ⁴)/d²`. -/
theorem signAvg_coord_pow_four_sub_sphere {d : ℕ} (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) :
    signAvg (fun s => (hadRot H s • v) i ^ 4) - ∫ t, t ^ 4 ∂(sphericalCoordLaw d)
      = (6 / ((d : ℝ) + 2) - 2 * ∑ j, v j ^ 4) / (d : ℝ) ^ 2 := by
  rw [signAvg_coord_pow_four hH, integral_pow_four_sphericalCoordLaw hd, hv]
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have hdne : (d:ℝ) ≠ 0 := by linarith
  have hd2 : (d:ℝ) + 2 ≠ 0 := by linarith
  field_simp
  ring

/-- For a unit vector, `1/d ≤ ∑ⱼ vⱼ⁴ ≤ 1`. -/
theorem sum_pow_four_bounds {d : ℕ} (hd : 0 < d) {v : Ed d} (hv : ‖v‖ = 1) :
    1 / (d : ℝ) ≤ ∑ j, v j ^ 4 ∧ ∑ j, v j ^ 4 ≤ 1 := by
  have h2 : ∑ j, v j ^ 2 = 1 := by rw [← norm_sq_eq_sum_sq, hv, one_pow]
  have hdR : (0:ℝ) < d := by exact_mod_cast hd
  constructor
  · have hcs := sq_sum_le_card_mul_sum_sq (s := Finset.univ) (f := fun j => v j ^ 2)
    simp only [Finset.card_univ, Fintype.card_fin] at hcs
    rw [h2, one_pow] at hcs
    have : ∀ j, (v j ^ 2) ^ 2 = v j ^ 4 := fun j => by ring
    simp only [this] at hcs
    rw [div_le_iff₀ hdR]; linarith
  · calc ∑ j, v j ^ 4 = ∑ j, v j ^ 2 * v j ^ 2 := by
          refine Finset.sum_congr rfl fun j _ => by ring
      _ ≤ ∑ j, v j ^ 2 * 1 := Finset.sum_le_sum fun j _ =>
          mul_le_mul_of_nonneg_left (by
            rw [← h2]
            exact Finset.single_le_sum (f := fun k => v k ^ 2) (fun k _ => sq_nonneg _)
              (Finset.mem_univ j)) (sq_nonneg _)
      _ = 1 := by simp [h2]

/-- **The gap is at most `4/d³` above and `2/d²` below**: for a spread direction the Hadamard
coordinate law matches `f_d` to fourth order, for a spiky one it falls short at order `1/d²`. -/
theorem abs_signAvg_coord_pow_four_sub_sphere {d : ℕ} (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) :
    signAvg (fun s => (hadRot H s • v) i ^ 4) - ∫ t, t ^ 4 ∂(sphericalCoordLaw d) ≤ 4 / (d : ℝ) ^ 3
    ∧ -(2 / (d : ℝ) ^ 2) ≤ signAvg (fun s => (hadRot H s • v) i ^ 4) - ∫ t, t ^ 4 ∂(sphericalCoordLaw d) := by
  rw [signAvg_coord_pow_four_sub_sphere hd hH hv i]
  obtain ⟨hlo, hhi⟩ := sum_pow_four_bounds (by omega) hv
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have hdpos : (0:ℝ) < d := by linarith
  constructor
  · rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have h6 : 6 / ((d:ℝ) + 2) ≤ 3 := by rw [div_le_iff₀ (by linarith)]; linarith
    have hl : 2 / (d:ℝ) ≤ 2 * ∑ j, v j ^ 4 := by
      have := mul_le_mul_of_nonneg_left hlo (by norm_num : (0:ℝ) ≤ 2)
      rw [show 2 * (1 / (d:ℝ)) = 2 / d by ring] at this; exact this
    have h6' : 6 / ((d:ℝ) + 2) - 2 / d ≤ 4 / d := by
      rw [div_sub_div _ _ (by linarith) hdpos.ne', div_le_div_iff₀ (by positivity) hdpos]
      nlinarith
    calc (6 / ((d:ℝ) + 2) - 2 * ∑ j, v j ^ 4) * (d:ℝ) ^ 3
        ≤ (6 / ((d:ℝ) + 2) - 2 / d) * (d:ℝ) ^ 3 := by gcongr
      _ ≤ 4 / d * (d:ℝ) ^ 3 := by gcongr
      _ = 4 * (d:ℝ) ^ 2 := by field_simp; try ring
  · rw [show -(2 / (d:ℝ) ^ 2) = (-2) / (d:ℝ) ^ 2 by ring,
      div_le_div_iff_of_pos_right (by positivity)]
    have : 0 ≤ 6 / ((d:ℝ) + 2) := by positivity
    linarith

end Hadamard
