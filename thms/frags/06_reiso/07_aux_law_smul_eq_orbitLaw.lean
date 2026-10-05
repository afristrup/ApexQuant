
/-- `law_smul_eq_orbitLaw`, proved — but with the joint-measurability instance added. -/
private theorem aux_law_smul_eq_orbitLaw [MeasurableSMul₂ G X] [IsProbabilityMeasure μ]
    {κ : Measure G} [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] [MeasurableMul G]
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    μ.map (fun ω => R ω • r ω) = orbitLaw κ x₀ := by
  haveI : IsProbabilityMeasure (μ.map r) :=
    Measure.isProbabilityMeasure_map hrmeas.aemeasurable
  have hpm : Measurable (fun ω => (R ω • r ω, r ω)) := (hRmeas.smul hrmeas).prodMk hrmeas
  have h := congrArg (Measure.map Prod.fst) (aux_map_pair hRmeas hrmeas hR hindep horb)
  rwa [Measure.map_map measurable_fst hpm, Measure.map_fst_prod, measure_univ, one_smul] at h
