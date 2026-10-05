
/-- **The statement Proposition 2 actually consumes.**

`indepFun_smul_comp` above is, at `G = O(d)` acting on `ℝ^d`, *vacuous* for the paper's purpose:
the hypothesis `∀ ω, r ω ∈ orbit G x₀` forces `‖r ω‖` to be a.s. constant, so these lemmas can
only be instantiated with the *normalised* residual `r̂ = r/‖r‖` — and then `f(r̂) = ‖r̂‖ ≡ 1`.
The paper's claim is `u ⟂ ‖r‖`, and `‖r‖` is not a function of `r̂`.

The correct statement carries the scale along as a separate variable `Z`: `R • r̂` is independent
of *any* auxiliary variable `Z` that is jointly independent of the fresh rotation.  Taking
`Z = ‖r‖` gives the paper's "`u` is independent of `‖r‖`", and taking `Z = (‖r‖, R_0, …, R_{p-1})`
gives the inductive step across passes. -/
theorem indepFun_smul_of_indepPair [IsProbabilityMeasure μ] {κ : Measure G}
    [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] [MeasurableMul G] [MeasurableSMul₂ G X]
    {β : Type*} [MeasurableSpace β]
    {R : Ω → G} {r : Ω → X} {Z : Ω → β}
    (hRmeas : Measurable R) (hrmeas : Measurable r) (hZmeas : Measurable Z)
    (hR : μ.map R = κ) (hindep : IndepFun R (fun ω => (r ω, Z ω)) μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    IndepFun (fun ω => R ω • r ω) Z μ := by
  haveI := aux_isProbabilityMeasure_orbitLaw κ x₀
  haveI : IsProbabilityMeasure (μ.map Z) :=
    Measure.isProbabilityMeasure_map hZmeas.aemeasurable
  have hmp := aux_map_pair_snd hRmeas hrmeas hZmeas hR hindep horb
  -- the first marginal of that joint law is the orbit law: the scale variable integrates out
  have hfst : μ.map (fun ω => R ω • r ω) = orbitLaw κ x₀ := by
    have hpm : Measurable (fun ω => (R ω • r ω, Z ω)) := (hRmeas.smul hrmeas).prodMk hZmeas
    have h := congrArg (Measure.map Prod.fst) hmp
    rwa [Measure.map_map measurable_fst hpm, Measure.map_fst_prod, measure_univ, one_smul] at h
  refine (indepFun_iff_map_prod_eq_prod_map_map
    (hRmeas.smul hrmeas).aemeasurable hZmeas.aemeasurable).2 ?_
  show μ.map (fun ω => (R ω • r ω, Z ω)) = (μ.map fun ω => R ω • r ω).prod (μ.map Z)
  rw [hmp, hfst]

end Reisotropisation
