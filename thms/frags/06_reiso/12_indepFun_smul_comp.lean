
/-- Composing: any function of `r`, in particular `‖r‖`, is independent of `R • r`. -/
theorem indepFun_smul_comp [IsProbabilityMeasure μ] {κ : Measure G} [IsProbabilityMeasure κ]
    [κ.IsMulRightInvariant] [MeasurableMul G] [MeasurableSMul₂ G X] {β : Type*} [MeasurableSpace β] {f : X → β}
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hfmeas : Measurable f) (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    IndepFun (fun ω => R ω • r ω) (fun ω => f (r ω)) μ :=
  aux_indepFun_smul_comp hRmeas hrmeas hfmeas hR hindep horb
