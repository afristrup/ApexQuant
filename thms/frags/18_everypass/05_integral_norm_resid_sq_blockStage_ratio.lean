
/-- The same with `ρ²` in the normalised form of eq. (2.3), `E_ν‖x − q_B x‖² / E_ν‖x‖²`; the
sphere constraint `‖u‖ = 1` is now derived from the stream rather than assumed. -/
theorem integral_norm_resid_sq_blockStage_ratio {qB : F → F} {ν : Measure F} (hm : m ≠ 0)
    (h : RotationStream ι (blockStage qB) W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hmarg : ∀ j, (orbitLaw κ x₀).map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) (hint₂ : Integrable (fun x : F => ‖x‖ ^ 2) ν)
    (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι (blockStage qB) (R ω)) (W ω) P‖ ^ 2 ∂μ
      = ((∫ x, ‖x - qB x‖ ^ 2 ∂ν) / ∫ x, ‖x‖ ^ 2 ∂ν) ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  haveI : IsProbabilityMeasure κ := by
    rw [← h.law_R 0]; exact Measure.isProbabilityMeasure_map (h.meas_R 0).aemeasurable
  haveI : IsProbabilityMeasure (orbitLaw κ x₀) :=
    Measure.isProbabilityMeasure_map (measurable_smul_const x₀).aemeasurable
  rw [integral_norm_resid_sq_rotationStream h hW (integrable_blockStage_err qB hmarg hint) P,
    stageDistortion_blockStage_eq_ratio hm qB hmarg hint hint₂
      (norm_eq_one_ae_orbitLaw h.implements hx₀)]

end BlockEveryPass
