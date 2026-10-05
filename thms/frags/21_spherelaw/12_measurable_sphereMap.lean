
theorem measurable_sphereMap (L : E ≃ₗᵢ[ℝ] F) : Measurable (sphereMap L) :=
  (continuous_sphereMap L).measurable
