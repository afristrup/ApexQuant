
/-- The ambient uniform law lives on the unit sphere. -/
theorem sphereUniformAmbient_ae_norm : ∀ᵐ x ∂(sphereUniformAmbient E), ‖x‖ = 1 := by
  rw [sphereUniformAmbient, ae_map_iff measurable_subtype_coe.aemeasurable
    (p := fun x : E => ‖x‖ = 1) (measurable_norm (measurableSet_singleton (1 : ℝ)))]
  exact Filter.Eventually.of_forall fun u => mem_sphere_zero_iff_norm.mp u.2
