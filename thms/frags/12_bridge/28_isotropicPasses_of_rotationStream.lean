
/-- **The bridge.**  A rotation stream satisfying `RotationStream`, hypotheses about the
*inputs* only, supplies `IsotropicPasses` with `ρ² = stageDistortion q (orbitLaw κ x₀)`.  All
four fields, including the two substantive ones, are derived. -/
theorem isotropicPasses_of_rotationStream (h : RotationStream ι q W R κ x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀)) :
    IsotropicPasses (fun ω => rotScheme ι q (R ω)) W
      (Real.sqrt (stageDistortion q (orbitLaw κ x₀))) μ where
  integrable p := integrable_normSq_resid h hW hint p
  integrable_err p := integrable_stageErr h hint p
  indep p := indepFun_scale_stageErr h p
  distortion p := by
    rw [Real.sq_sqrt (stageDistortion_nonneg q (orbitLaw κ x₀))]
    exact integral_stageErr_eq_stageDistortion h p
