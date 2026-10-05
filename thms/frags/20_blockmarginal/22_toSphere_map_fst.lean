
/-- **The block marginal of the sphere measure, unnormalised.**  The pushforward of
`volume.toSphere` on `sphere (0 : U × V) 1` along the `U`-block is `σ_V(Set.univ)` times the
density `(1 − ‖x‖²)^{(dim V − 2)/2}` on the unit ball of `U`. -/
theorem toSphere_map_fst :
    (volume.toSphere : Measure (sphere (0 : WithLp 2 (U × V)) 1)).map
        (fun u : sphere (0 : WithLp 2 (U × V)) 1 => (WithLp.ofLp (u : WithLp 2 (U × V))).1)
      = volume.toSphere (Set.univ : Set (sphere (0 : V) 1)) • blockMeasure U (Module.finrank ℝ V) := by
  ext s hs
  have hψ : Measurable (s.indicator (1 : U → ℝ≥0∞)) := measurable_const.indicator hs
  rw [Measure.map_apply measurable_sphere_fst hs, Measure.smul_apply, blockMeasure,
    withDensity_apply _ hs, Measure.restrict_restrict hs, smul_eq_mul]
  have hL : volume.toSphere ((fun u : sphere (0 : WithLp 2 (U × V)) 1 =>
        (WithLp.ofLp (u : WithLp 2 (U × V))).1) ⁻¹' s)
      = ∫⁻ u : sphere (0 : WithLp 2 (U × V)) 1,
          s.indicator 1 (WithLp.ofLp (u : WithLp 2 (U × V))).1 ∂volume.toSphere := by
    rw [← lintegral_indicator_one (measurable_sphere_fst hs)]
    refine lintegral_congr fun u => ?_
    by_cases h : (WithLp.ofLp (u : WithLp 2 (U × V))).1 ∈ s
    · rw [Set.indicator_of_mem h, Set.indicator_of_mem (show u ∈ (fun u : sphere (0 : WithLp 2 (U × V)) 1 =>
        (WithLp.ofLp (u : WithLp 2 (U × V))).1) ⁻¹' s from h)]
      rfl
    · rw [Set.indicator_of_notMem h, Set.indicator_of_notMem
        (show u ∉ (fun u : sphere (0 : WithLp 2 (U × V)) 1 =>
          (WithLp.ofLp (u : WithLp 2 (U × V))).1) ⁻¹' s from h)]
  have hR : ∫⁻ x in s ∩ ball (0 : U) 1, blockDensity (Module.finrank ℝ V) x
      = ∫⁻ x in ball (0 : U) 1, blockDensity (Module.finrank ℝ V) x * s.indicator 1 x := by
    rw [← Measure.restrict_restrict hs, ← lintegral_indicator hs]
    refine lintegral_congr fun x => ?_
    by_cases h : x ∈ s
    · rw [Set.indicator_of_mem h, Set.indicator_of_mem h]
      simp
    · rw [Set.indicator_of_notMem h, Set.indicator_of_notMem h, mul_zero]
  rw [hL, hR, lintegral_sphere_fst hψ, lintegral_ball_density hψ _ Module.finrank_pos]

end MeasureForm
