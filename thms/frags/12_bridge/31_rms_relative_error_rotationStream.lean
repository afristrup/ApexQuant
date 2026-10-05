
/-- **Eq. (2.5), unconditionally**, for a deterministic weight row: the RMS relative error after
`P` passes is exactly `ρ^P` with `ρ = √(stage distortion)`. -/
theorem rms_relative_error_rotationStream {w : E} (hw : w ≠ 0)
    (h : RotationStream ι q (fun _ => w) R κ x₀ μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) :
    Real.sqrt (∫ ω, ‖resid (rotScheme ι q (R ω)) w P‖ ^ 2 ∂μ) / ‖w‖
      = Real.sqrt (stageDistortion q (orbitLaw κ x₀)) ^ P :=
  rms_relative_error hw (Real.sqrt_nonneg _)
    (isotropicPasses_of_rotationStream h (integrable_const _) hint) P

end BridgeAssembly
