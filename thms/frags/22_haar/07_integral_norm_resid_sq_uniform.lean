
/-- **Proposition 2 with the uniform law**: `ρ²` is the stage distortion against the uniform
direction, the quantity Table 2 measures. -/
theorem integral_norm_resid_sq_uniform (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (sphereUniformAmbient E)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι q (R ω)) (W ω) P‖ ^ 2 ∂μ
      = stageDistortion q (sphereUniformAmbient E) ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  rw [← h.orbitLaw_eq hx₀] at hint ⊢
  exact integral_norm_resid_sq_rotationStream h hW hint P

end RotationStreamUniform

section Coordinates
