
/-- **Proposition 2 for a block stage**, with `ρ²` evaluated by the block quadrature: for an
`E₈` stage `m = d/8` and `ν` is the law of eight coordinates of the direction, for a trellis
stage of length `T` it is `m = d/T`, and for the scalar stage `m = d` and `ν = f_d`. -/
theorem integral_norm_resid_sq_blockStage {qB : F → F} {ν : Measure F}
    (h : RotationStream ι (blockStage qB) W R κ x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hmarg : ∀ j, (orbitLaw κ x₀).map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι (blockStage qB) (R ω)) (W ω) P‖ ^ 2 ∂μ
      = (m * ∫ x, ‖x - qB x‖ ^ 2 ∂ν) ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  rw [integral_norm_resid_sq_rotationStream h hW (integrable_blockStage_err qB hmarg hint) P,
    stageDistortion_blockStage qB hmarg hint]
