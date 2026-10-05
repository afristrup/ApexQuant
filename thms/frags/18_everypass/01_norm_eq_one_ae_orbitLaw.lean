/-- The direction law lives on the unit sphere: `G` acts by isometries and `‖x₀‖ = 1`. -/
theorem norm_eq_one_ae_orbitLaw [OpensMeasurableSpace E]
    (hι : ∀ (g : G) (x : E), ι g x = g • x) (hx₀ : ‖x₀‖ = 1) :
    ∀ᵐ v ∂(orbitLaw κ x₀), ‖v‖ = 1 := by
  unfold orbitLaw
  refine (ae_map_iff (measurable_smul_const x₀).aemeasurable (p := fun v : E => ‖v‖ = 1)
    (measurable_norm (measurableSet_singleton (1 : ℝ)))).2 ?_
  exact Filter.Eventually.of_forall fun g => by rw [← hι, LinearIsometryEquiv.norm_map, hx₀]
