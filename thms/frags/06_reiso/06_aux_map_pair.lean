
/-- The joint law of `(R • r, r)` is the product of the orbit law and the law of `r`.  This is
the single computation behind all three of `law_smul_eq_orbitLaw`, `indepFun_smul_left` and
`indepFun_smul_comp`; it is proved here in full, but only under `[MeasurableSMul₂ G X]`. -/
private theorem aux_map_pair [MeasurableSMul₂ G X] [IsProbabilityMeasure μ]
    {κ : Measure G} [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] [MeasurableMul G]
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    μ.map (fun ω => (R ω • r ω, r ω)) = (orbitLaw κ x₀).prod (μ.map r) := by
  haveI := aux_isProbabilityMeasure_orbitLaw κ x₀
  haveI : IsProbabilityMeasure (μ.map r) :=
    Measure.isProbabilityMeasure_map hrmeas.aemeasurable
  have hΦ : Measurable (fun p : G × X => (p.1 • p.2, p.2)) := by fun_prop
  have hpair : μ.map (fun ω => (R ω, r ω)) = κ.prod (μ.map r) := by
    rw [hindep.map_prod_eq_prod_map_map hRmeas.aemeasurable hrmeas.aemeasurable, hR]
  have h1 : μ.map (fun ω => (R ω • r ω, r ω))
      = (κ.prod (μ.map r)).map (fun p : G × X => (p.1 • p.2, p.2)) := by
    rw [← hpair, Measure.map_map hΦ (hRmeas.prodMk hrmeas)]; rfl
  rw [h1]
  refine (Measure.prod_eq ?_).symm
  intro s t hs ht
  have hst : MeasurableSet ((fun p : G × X => (p.1 • p.2, p.2)) ⁻¹' (s ×ˢ t)) := hΦ (hs.prod ht)
  rw [Measure.map_apply hΦ (hs.prod ht), Measure.prod_apply_symm hst]
  set F : X → ℝ≥0∞ :=
    fun y => κ ((fun x : G => (x, y)) ⁻¹' ((fun p : G × X => (p.1 • p.2, p.2)) ⁻¹' (s ×ˢ t)))
    with hF
  have hFmeas : Measurable F := measurable_measure_prodMk_right hst
  -- the orbit hypothesis turns the inner measure into the *constant* `orbitLaw κ x₀ s`
  have hstep : ∀ᵐ ω ∂μ, F (r ω) = t.indicator (fun _ => orbitLaw κ x₀ s) (r ω) := by
    filter_upwards [horb] with ω horbω
    by_cases hyt : r ω ∈ t
    · have he : (fun x : G => (x, r ω)) ⁻¹'
          ((fun p : G × X => (p.1 • p.2, p.2)) ⁻¹' (s ×ˢ t)) = {g : G | g • r ω ∈ s} := by
        ext g; simp [hyt]
      simp only [hF, he, Set.indicator_of_mem hyt]
      rw [← aux_orbitLaw_apply κ (r ω) hs, orbitLaw_eq_of_mem_orbit κ horbω]
    · have he : (fun x : G => (x, r ω)) ⁻¹'
          ((fun p : G × X => (p.1 • p.2, p.2)) ⁻¹' (s ×ˢ t)) = ∅ := by
        ext g; simp [hyt]
      simp only [hF, he, Set.indicator_of_notMem hyt, measure_empty]
  rw [lintegral_map hFmeas hrmeas]
  calc ∫⁻ ω, F (r ω) ∂μ
      = ∫⁻ ω, (r ⁻¹' t).indicator (fun _ => orbitLaw κ x₀ s) ω ∂μ := by
        refine lintegral_congr_ae ?_
        filter_upwards [hstep] with ω hstepω
        rw [hstepω]
        by_cases hyt : r ω ∈ t <;> simp [hyt, Set.indicator_of_mem, Set.indicator_of_notMem]
    _ = orbitLaw κ x₀ s * (μ.map r) t := by
        rw [lintegral_indicator (hrmeas ht), setLIntegral_const, Measure.map_apply hrmeas ht]
