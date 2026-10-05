
/-- **Proposition 2, unconditionally**, for a re-isotropised progressive quantiser: the geometric
decay law with the constant *identified* as the stage distortion of the codebook against the
direction law, rather than assumed to exist. -/
theorem integral_norm_resid_sq_rotationStream (h : RotationStream ι q W R κ x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι q (R ω)) (W ω) P‖ ^ 2 ∂μ
      = stageDistortion q (orbitLaw κ x₀) ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  have hkey := integral_norm_resid_sq (isotropicPasses_of_rotationStream h hW hint) P
  rwa [pow_mul, Real.sq_sqrt (stageDistortion_nonneg q (orbitLaw κ x₀))] at hkey
