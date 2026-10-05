
/-- **The 2D polar step.**  The quadrant double integral factors as the radial constant times
the angular integral over `(0, π/2)`. -/
theorem lintegral_quadFun (a b : ℕ) {g : ℝ → ℝ≥0∞} (hg : Measurable g) :
    ∫⁻ ρ in Set.Ioi (0 : ℝ), ∫⁻ s in Set.Ioi (0 : ℝ), quadFun a b g (ρ, s)
      = radialConst (a + b + 1) *
          ∫⁻ θ in Set.Ioo 0 (Real.pi / 2), angWeight a b θ * g (Real.cos θ) := by
  have hmq := measurable_quadFun a b hg
  have h1 : ∫⁻ ρ in Set.Ioi (0 : ℝ), ∫⁻ s in Set.Ioi (0 : ℝ), quadFun a b g (ρ, s)
      = ∫⁻ p, (Set.Ioi (0 : ℝ) ×ˢ Set.Ioi (0 : ℝ)).indicator (quadFun a b g) p := by
    rw [← lintegral_prod _ hmq.aemeasurable, Measure.prod_restrict,
      lintegral_indicator (measurableSet_Ioi.prod measurableSet_Ioi), volume_eq_prod]
  have h2 : ∫⁻ p, (Set.Ioi (0 : ℝ) ×ˢ Set.Ioi (0 : ℝ)).indicator (quadFun a b g) p
      = ∫⁻ p in polarCoord.target, radFun (a + b + 1) p.1 * angFun a b g p.2 := by
    rw [← lintegral_comp_polarCoord_symm, polarCoord_target]
    refine setLIntegral_congr_fun (measurableSet_Ioi.prod measurableSet_Ioo) fun p hp => ?_
    rw [polarCoord_symm_apply, smul_eq_mul]
    exact quadFun_polar a b g hp.1 hp.2
  have h3 : ∫⁻ p in polarCoord.target, radFun (a + b + 1) p.1 * angFun a b g p.2
      = radialConst (a + b + 1) * ∫⁻ θ in Set.Ioo (-Real.pi) Real.pi, angFun a b g θ := by
    rw [polarCoord_target, volume_eq_prod, ← Measure.prod_restrict,
      lintegral_prod_mul (measurable_radFun _).aemeasurable (measurable_angFun a b hg).aemeasurable]
    rfl
  have h4 : ∫⁻ θ in Set.Ioo (-Real.pi) Real.pi, angFun a b g θ
      = ∫⁻ θ in Set.Ioo 0 (Real.pi / 2), angWeight a b θ * g (Real.cos θ) := by
    unfold angFun
    rw [lintegral_indicator measurableSet_Ioo, Measure.restrict_restrict measurableSet_Ioo,
      Set.inter_eq_left.mpr (Set.Ioo_subset_Ioo (by linarith [Real.pi_pos]) (by linarith [Real.pi_pos]))]
  rw [h1, h2, h3, h4]

end Polar2D
