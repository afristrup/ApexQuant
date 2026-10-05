
theorem toSphere_univ_eq (L : E ≃ₗᵢ[ℝ] F) :
    volume.toSphere (Set.univ : Set (sphere (0 : E) 1)) = volume.toSphere (Set.univ : Set (sphere (0 : F) 1)) := by
  rw [← toSphere_map_sphereMap L, Measure.map_apply (measurable_sphereMap L) MeasurableSet.univ,
    Set.preimage_univ]
