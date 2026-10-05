
/-- The coordinate marginal of the ambient uniform law: the paper's Lemma 1 in the form §3b
consumes it (`hmarg : ∀ i, μ.map (fun ω => U ω i) = ν` with `ν = sphericalCoordLaw d`). -/
theorem sphereUniformAmbient_map_inner (hd : 2 ≤ Module.finrank ℝ E) {e : E} (he : ‖e‖ = 1) :
    (sphereUniformAmbient E).map (fun x : E => ⟪e, x⟫_ℝ)
      = sphericalCoordLaw (Module.finrank ℝ E) := by
  have hg : Measurable fun x : E => ⟪e, x⟫_ℝ := (continuous_const.inner continuous_id).measurable
  rw [sphereUniformAmbient, Measure.map_map hg measurable_subtype_coe]
  exact sphereUniform_map_inner hd he

end Coordinate
