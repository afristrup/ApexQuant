
/-- **The density side.**  The candidate density integrated against `ψ` over the unit ball of
`U` is the same angular integral as `lintegral_sphere_fst` produces. -/
theorem lintegral_ball_density {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) (m : ℕ) (hm : 1 ≤ m) :
    ∫⁻ x in ball (0 : U) 1, blockDensity m x * ψ x
      = ∫⁻ v : sphere (0 : U) 1, (∫⁻ θ in Set.Ioo 0 (Real.pi / 2),
          angWeight (Module.finrank ℝ U - 1) (m - 1) θ * ψ (Real.cos θ • (v : U)))
          ∂volume.toSphere := by
  rw [← lintegral_indicator measurableSet_ball,
    lintegral_eq_polar (fun x => (ball (0 : U) 1).indicator (fun x => blockDensity m x * ψ x) x)
      (by exact ((measurable_blockDensity m).mul hψ).indicator measurableSet_ball)]
  refine lintegral_congr fun v => ?_
  have hv : ‖(v : U)‖ = 1 := mem_sphere_zero_iff_norm.mp v.2
  have hpt : ∀ ρ ∈ Set.Ioi (0 : ℝ), ENNReal.ofReal (ρ ^ (Module.finrank ℝ U - 1)) *
        (ball (0 : U) 1).indicator (fun x => blockDensity m x * ψ x) (ρ • (v : U))
      = (Set.Iio (1 : ℝ)).indicator (fun ρ => ENNReal.ofReal (ρ ^ (Module.finrank ℝ U - 1)) *
          (ENNReal.ofReal ((1 - ρ ^ 2) ^ (((m : ℝ) - 2) / 2)) * ψ (ρ • (v : U)))) ρ := by
    intro ρ hρ
    have hn : ‖ρ • (v : U)‖ = ρ := by
      rw [norm_smul, hv, mul_one, Real.norm_eq_abs, abs_of_pos hρ]
    by_cases h1 : ρ < 1
    · rw [Set.indicator_of_mem (by rwa [mem_ball_zero_iff, hn]),
        Set.indicator_of_mem (Set.mem_Iio.mpr h1), blockDensity, hn]
    · rw [Set.indicator_of_notMem (by rwa [mem_ball_zero_iff, hn]),
        Set.indicator_of_notMem (fun h' => h1 (Set.mem_Iio.mp h')), mul_zero]
  rw [setLIntegral_congr_fun measurableSet_Ioi hpt, lintegral_indicator measurableSet_Iio,
    Measure.restrict_restrict measurableSet_Iio, Set.inter_comm, Set.Ioi_inter_Iio, ← image_cos_Ioo,
    lintegral_image_eq_lintegral_abs_deriv_mul measurableSet_Ioo
      (fun θ _ => (Real.hasDerivAt_cos θ).hasDerivWithinAt)
      (Real.injOn_cos.mono (Set.Ioo_subset_Icc_self.trans
        (Set.Icc_subset_Icc le_rfl (by linarith [Real.pi_pos]))))]
  refine setLIntegral_congr_fun measurableSet_Ioo fun θ hθ => ?_
  have hs : 0 < Real.sin θ := Real.sin_pos_of_pos_of_lt_pi hθ.1 (by linarith [hθ.2, Real.pi_pos])
  have hc : 0 ≤ Real.cos θ :=
    (Real.cos_pos_of_mem_Ioo ⟨by linarith [hθ.1, Real.pi_pos], hθ.2⟩).le
  rw [abs_neg, abs_of_pos hs, angWeight, ← sin_mul_rpow m hm hθ,
    ENNReal.ofReal_mul (pow_nonneg hc _), ENNReal.ofReal_mul hs.le]
  ring

end Density

section MeasureForm
variable {U V : Type*} [NormedAddCommGroup U] [InnerProductSpace ℝ U] [MeasurableSpace U]
  [BorelSpace U] [FiniteDimensional ℝ U] [Nontrivial U]
  [NormedAddCommGroup V] [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V]
  [FiniteDimensional ℝ V] [Nontrivial V]
