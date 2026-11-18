
/-- The joint law of `(R • r, Z)` for an auxiliary variable `Z` carried alongside `r`.  This is
`aux_map_pair` with the action applied to the *first* component of `X × β` while `β` is dragged
through untouched: the orbit hypothesis still makes the inner `κ`-measure the constant
`orbitLaw κ x₀ s`, so Fubini factorises exactly as before.  Note that the second factor is the
law of `Z` alone, all information about `r` beyond its orbit has been washed out. -/
private theorem aux_map_pair_snd [MeasurableSMul₂ G X] [IsProbabilityMeasure μ]
    {κ : Measure G} [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] [MeasurableMul G]
    {β : Type*} [MeasurableSpace β]
    {R : Ω → G} {r : Ω → X} {Z : Ω → β}
    (hRmeas : Measurable R) (hrmeas : Measurable r) (hZmeas : Measurable Z)
    (hR : μ.map R = κ) (hindep : IndepFun R (fun ω => (r ω, Z ω)) μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    μ.map (fun ω => (R ω • r ω, Z ω)) = (orbitLaw κ x₀).prod (μ.map Z) := by
  haveI := aux_isProbabilityMeasure_orbitLaw κ x₀
  haveI : IsProbabilityMeasure (μ.map Z) :=
    Measure.isProbabilityMeasure_map hZmeas.aemeasurable
  have hYmeas : Measurable (fun ω => (r ω, Z ω)) := hrmeas.prodMk hZmeas
  haveI : IsProbabilityMeasure (μ.map (fun ω => (r ω, Z ω))) :=
    Measure.isProbabilityMeasure_map hYmeas.aemeasurable
  have hΦ : Measurable (fun p : G × (X × β) => (p.1 • p.2.1, p.2.2)) := by fun_prop
  have hpair : μ.map (fun ω => (R ω, (r ω, Z ω)))
      = κ.prod (μ.map fun ω => (r ω, Z ω)) := by
    rw [hindep.map_prod_eq_prod_map_map hRmeas.aemeasurable hYmeas.aemeasurable, hR]
  have h1 : μ.map (fun ω => (R ω • r ω, Z ω))
      = (κ.prod (μ.map fun ω => (r ω, Z ω))).map
          (fun p : G × (X × β) => (p.1 • p.2.1, p.2.2)) := by
    rw [← hpair, Measure.map_map hΦ (hRmeas.prodMk hYmeas)]; rfl
  rw [h1]
  refine (Measure.prod_eq ?_).symm
  intro s t hs ht
  have hst : MeasurableSet
      ((fun p : G × (X × β) => (p.1 • p.2.1, p.2.2)) ⁻¹' (s ×ˢ t)) := hΦ (hs.prod ht)
  rw [Measure.map_apply hΦ (hs.prod ht), Measure.prod_apply_symm hst]
  set F : X × β → ℝ≥0∞ :=
    fun y => κ ((fun x : G => (x, y)) ⁻¹'
      ((fun p : G × (X × β) => (p.1 • p.2.1, p.2.2)) ⁻¹' (s ×ˢ t))) with hF
  have hFmeas : Measurable F := measurable_measure_prodMk_right hst
  -- the orbit hypothesis turns the inner measure into the *constant* `orbitLaw κ x₀ s`,
  -- independently of `ω`; only the `Z`-slice survives
  have hstep : ∀ᵐ ω ∂μ, F (r ω, Z ω) = (Z ⁻¹' t).indicator (fun _ => orbitLaw κ x₀ s) ω := by
    filter_upwards [horb] with ω horbω
    by_cases hyt : Z ω ∈ t
    · have he : (fun x : G => (x, (r ω, Z ω))) ⁻¹'
          ((fun p : G × (X × β) => (p.1 • p.2.1, p.2.2)) ⁻¹' (s ×ˢ t))
            = {g : G | g • r ω ∈ s} := by
        ext g; simp [hyt]
      simp only [hF, he, Set.indicator_of_mem (show ω ∈ Z ⁻¹' t from hyt)]
      rw [← aux_orbitLaw_apply κ (r ω) hs, orbitLaw_eq_of_mem_orbit κ horbω]
    · have he : (fun x : G => (x, (r ω, Z ω))) ⁻¹'
          ((fun p : G × (X × β) => (p.1 • p.2.1, p.2.2)) ⁻¹' (s ×ˢ t)) = ∅ := by
        ext g; simp [hyt]
      simp only [hF, he, Set.indicator_of_notMem (show ω ∉ Z ⁻¹' t from hyt), measure_empty]
  rw [lintegral_map hFmeas hYmeas]
  calc ∫⁻ ω, F (r ω, Z ω) ∂μ
      = ∫⁻ ω, (Z ⁻¹' t).indicator (fun _ => orbitLaw κ x₀ s) ω ∂μ := lintegral_congr_ae hstep
    _ = orbitLaw κ x₀ s * (μ.map Z) t := by
        rw [lintegral_indicator (hZmeas ht), setLIntegral_const, Measure.map_apply hZmeas ht]
