
/-- `indepFun_smul_left`, proved, but with the joint-measurability instance added. -/
private theorem aux_indepFun_smul_left [MeasurableSMul₂ G X] [IsProbabilityMeasure μ]
    {κ : Measure G} [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] [MeasurableMul G]
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    IndepFun (fun ω => R ω • r ω) r μ := by
  refine (indepFun_iff_map_prod_eq_prod_map_map
    (hRmeas.smul hrmeas).aemeasurable hrmeas.aemeasurable).2 ?_
  show μ.map (fun ω => (R ω • r ω, r ω)) = (μ.map (fun ω => R ω • r ω)).prod (μ.map r)
  rw [aux_map_pair hRmeas hrmeas hR hindep horb,
    aux_law_smul_eq_orbitLaw hRmeas hrmeas hR hindep horb]
