
/-- **The mass of the block density**: `∫_{ball} (1 − ‖x‖²)^{(m−2)/2} dx = √π^B Γ(m/2)/Γ((B+m)/2)`,
read off from the block marginal of the sphere measure (§20) applied to the whole space, with
the volumes of balls from Mathlib. -/
theorem blockMeasure_univ (m : ℕ) (hm : 1 ≤ m) :
    blockMeasure U m Set.univ = ENNReal.ofReal (blockMass (Module.finrank ℝ U) m) := by
  haveI : Nonempty (Fin m) := ⟨⟨0, by omega⟩⟩
  have hmV : Module.finrank ℝ (EuclideanSpace ℝ (Fin m)) = m := finrank_euclideanSpace_fin
  have h := congrArg (fun ν : Measure U => ν Set.univ)
    (toSphere_map_fst (U := U) (V := EuclideanSpace ℝ (Fin m)))
  rw [Measure.map_apply measurable_sphere_fst MeasurableSet.univ, Set.preimage_univ,
    Measure.smul_apply, smul_eq_mul, hmV, Measure.toSphere_apply_univ, Measure.toSphere_apply_univ,
    InnerProductSpace.volume_ball, InnerProductSpace.volume_ball, finrank_withLp_prod, hmV] at h
  set B := Module.finrank ℝ U with hB
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hB' : (0 : ℝ) < B := by exact_mod_cast Module.finrank_pos
  have hΓm : 0 < Real.Gamma ((m : ℝ) / 2) := Real.Gamma_pos_of_pos (by positivity)
  have hΓBm : 0 < Real.Gamma (((B : ℝ) + m) / 2) := Real.Gamma_pos_of_pos (by positivity)
  have hΓm1 : 0 < Real.Gamma ((m : ℝ) / 2 + 1) := Real.Gamma_pos_of_pos (by positivity)
  have hΓBm1 : 0 < Real.Gamma (((B + m : ℕ) : ℝ) / 2 + 1) := Real.Gamma_pos_of_pos (by positivity)
  have hc : (m : ℝ≥0∞) * (ENNReal.ofReal 1 ^ m
      * ENNReal.ofReal (Real.sqrt π ^ m / Real.Gamma ((m : ℝ) / 2 + 1)))
      = ENNReal.ofReal (m * (Real.sqrt π ^ m / Real.Gamma ((m : ℝ) / 2 + 1))) := by
    rw [ENNReal.ofReal_one, one_pow, one_mul, ENNReal.ofReal_mul (by positivity),
      ENNReal.ofReal_natCast]
  have ha : ((B + m : ℕ) : ℝ≥0∞) * (ENNReal.ofReal 1 ^ (B + m)
      * ENNReal.ofReal (Real.sqrt π ^ (B + m) / Real.Gamma (((B + m : ℕ) : ℝ) / 2 + 1)))
      = ENNReal.ofReal ((B + m : ℕ) * (Real.sqrt π ^ (B + m)
          / Real.Gamma (((B + m : ℕ) : ℝ) / 2 + 1))) := by
    rw [ENNReal.ofReal_one, one_pow, one_mul, ENNReal.ofReal_mul (by positivity),
      ENNReal.ofReal_natCast]
  rw [ha, hc] at h
  have hc0 : ENNReal.ofReal (m * (Real.sqrt π ^ m / Real.Gamma ((m : ℝ) / 2 + 1))) ≠ 0 := by
    rw [ne_eq, ENNReal.ofReal_eq_zero, not_le]; positivity
  refine (ENNReal.mul_right_inj hc0 ENNReal.ofReal_ne_top).mp ?_
  rw [← h, ← ENNReal.ofReal_mul (by positivity)]
  congr 1
  unfold blockMass
  rw [Real.Gamma_add_one (by positivity : ((m : ℝ) / 2) ≠ 0),
    Real.Gamma_add_one (by positivity : (((B + m : ℕ) : ℝ) / 2) ≠ 0)]
  push_cast
  field_simp
  ring
