
/-- The uniform sphere law is natural under linear isometries. -/
theorem sphereUniform_map_sphereMap (L : E ≃ₗᵢ[ℝ] F) :
    (sphereUniform E).map (sphereMap L) = sphereUniform F := by
  rw [sphereUniform, sphereUniform, Measure.map_smul, toSphere_map_sphereMap, toSphere_univ_eq L]

end Naturality

section Coordinate

open scoped InnerProductSpace
