
/-- **The sphere measure is natural under linear isometries.** -/
theorem toSphere_map_sphereMap (L : E ≃ₗᵢ[ℝ] F) :
    (volume.toSphere : Measure (sphere (0 : E) 1)).map (sphereMap L)
      = (volume.toSphere : Measure (sphere (0 : F) 1)) := by
  ext s hs
  rw [Measure.map_apply (measurable_sphereMap L) hs,
    Measure.toSphere_apply' _ (measurable_sphereMap L hs), Measure.toSphere_apply' _ hs,
    ← preimage_cone, L.measurePreserving.measure_preimage_emb L.toHomeomorph.measurableEmbedding,
    L.toLinearEquiv.finrank_eq]
