
/-- **Corollary 1 at the residual level, for every stage.**  Of two stages driven by the same
rotation stream from the same weight rows, the one with the smaller stage distortion has the
smaller expected squared residual after every number of passes. -/
theorem integral_norm_resid_sq_le_of_stageDistortion_le {q q' : E → E}
    (h : RotationStream ι q W R κ x₀ μ) (h' : RotationStream ι q' W R κ x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀))
    (hint' : Integrable (fun u : E => ‖u - q' u‖ ^ 2) (orbitLaw κ x₀))
    (hle : stageDistortion q (orbitLaw κ x₀) ≤ stageDistortion q' (orbitLaw κ x₀)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι q (R ω)) (W ω) P‖ ^ 2 ∂μ
      ≤ ∫ ω, ‖resid (rotScheme ι q' (R ω)) (W ω) P‖ ^ 2 ∂μ := by
  rw [integral_norm_resid_sq_rotationStream h hW hint P,
    integral_norm_resid_sq_rotationStream h' hW hint' P]
  exact mul_le_mul_of_nonneg_right (pow_le_pow_left₀ (stageDistortion_nonneg _ _) hle P)
    (integral_nonneg fun _ => by positivity)
