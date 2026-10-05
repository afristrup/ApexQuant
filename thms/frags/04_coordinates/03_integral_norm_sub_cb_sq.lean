
/-- **Linearity of expectation over exchangeable coordinates.**  If the `d` coordinates of the
random unit vector `U` each have law `ν`, then `E‖U - q U‖² = d · E_ν[(t - cb t)²] = ρ²`,
where `q` is `cb` applied coordinate-wise.  No independence between coordinates is used. -/
theorem integral_norm_sub_cb_sq
    {U : Ω → EuclideanSpace ℝ (Fin d)} {cb : ℝ → ℝ} {ν : Measure ℝ}
    (hmeas : ∀ i, Measurable fun ω => U ω i)
    (hmarg : ∀ i, μ.map (fun ω => U ω i) = ν)
    (hint : Integrable (fun t => (t - cb t) ^ 2) ν) :
    ∫ ω, ‖U ω - WithLp.toLp 2 (fun i => cb (U ω i))‖ ^ 2 ∂μ = rhoSq d cb ν := by
  -- Each coordinate map pushes `μ` forward to `ν`, so `(t - cb t)^2` is `ν`-a.e. strongly
  -- measurable when read through any one of them.
  have hasm : ∀ i, AEStronglyMeasurable (fun t => (t - cb t) ^ 2)
      (μ.map (fun ω => U ω i)) := by
    intro i; rw [hmarg i]; exact hint.aestronglyMeasurable
  -- Integrability transported back along each coordinate.  This is the *only* place a
  -- side condition beyond the marginal law is needed, and it is per-coordinate: nothing
  -- about the joint law of the `d` coordinates is used.
  have hmapint : ∀ i, Integrable (fun ω => (U ω i - cb (U ω i)) ^ 2) μ := fun i =>
    (integrable_map_measure (hasm i) (hmeas i).aemeasurable).mp (by rw [hmarg i]; exact hint)
  -- Change of variables, one coordinate at a time.
  have hcoord : ∀ i : Fin d, ∫ ω, (U ω i - cb (U ω i)) ^ 2 ∂μ = ∫ t, (t - cb t) ^ 2 ∂ν := by
    intro i; rw [← hmarg i, integral_map (hmeas i).aemeasurable (hasm i)]
  -- `‖·‖²` on `EuclideanSpace` is the coordinate sum; `q` acts coordinate-wise.
  have hexp : ∀ ω : Ω, ‖U ω - WithLp.toLp 2 (fun i => cb (U ω i))‖ ^ 2
      = ∑ i, (U ω i - cb (U ω i)) ^ 2 := by
    intro ω; rw [EuclideanSpace.real_norm_sq_eq]; simp
  -- Linearity of expectation: swap `∫` and the finite `∑`.  No independence enters.
  simp_rw [hexp]
  rw [integral_finsetSum _ (fun i _ => hmapint i)]
  simp_rw [hcoord]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, rhoSq]
