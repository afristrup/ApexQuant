
theorem continuous_sphereMap (L : E ≃ₗᵢ[ℝ] F) : Continuous (sphereMap L) :=
  (L.continuous.comp continuous_subtype_val).subtype_mk _
