
/-- **Theorem 1, abstract form, part 1 (the law).**  If `R ∼ κ` is Haar-distributed and
independent of `r`, and `r` lives in a single orbit, then `R • r` has the orbit law — *whatever
the law of `r`*.  This is the "`r` may be anything" clause that licenses reuse of one codebook
across residual passes. -/
theorem law_smul_eq_orbitLaw [IsProbabilityMeasure μ] {κ : Measure G} [IsProbabilityMeasure κ]
    [κ.IsMulRightInvariant] [MeasurableMul G] [MeasurableSMul₂ G X]
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    μ.map (fun ω => R ω • r ω) = orbitLaw κ x₀ :=
  aux_law_smul_eq_orbitLaw hRmeas hrmeas hR hindep horb
