
/-- **Theorem 1, abstract form, part 2 (the independence).**  `R • r` is independent of `r`.
This — not the uniformity — is what Proposition 2 consumes, via `u_p ⟂ s_p`. -/
theorem indepFun_smul_left [IsProbabilityMeasure μ] {κ : Measure G} [IsProbabilityMeasure κ]
    [κ.IsMulRightInvariant] [MeasurableMul G] [MeasurableSMul₂ G X]
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    IndepFun (fun ω => R ω • r ω) r μ :=
  aux_indepFun_smul_left hRmeas hrmeas hR hindep horb
