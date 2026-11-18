
/-- `indepFun_smul_comp`, proved, but with the joint-measurability instance added. -/
private theorem aux_indepFun_smul_comp [MeasurableSMul₂ G X] [IsProbabilityMeasure μ]
    {κ : Measure G} [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] [MeasurableMul G]
    {β : Type*} [MeasurableSpace β] {f : X → β}
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hfmeas : Measurable f) (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    IndepFun (fun ω => R ω • r ω) (fun ω => f (r ω)) μ :=
  (aux_indepFun_smul_left hRmeas hrmeas hR hindep horb).comp measurable_id hfmeas
