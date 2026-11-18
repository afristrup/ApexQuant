/-! ## §25  The concentration remark, reduced to the Poincaré inequality on the sphere

Header item 5 records that the `Var = O(1/d)` estimate behind the paper's Remark (Concentration)
was not formalised, and that the paper's own route to it, Dirichlet mixed moments and the
`O(1/d³)` covariances of the spherical marginal, is a computation specific to the scalar stage.
There is a better route.  For a nearest-codeword stage the per-pass ratio is
`F(u) = dist(u, C)²`, whose pointwise Lipschitz constant is `2 dist(u, C)`; the Poincaré
inequality on `S^{d−1}`, with the spectral-gap constant `1/(d−1)`, therefore gives
`Var F ≤ 4 ρ²/(d−1)` at once, for scalar, `E₈` and trellis stages alike.

This section takes that one inequality as a hypothesis, `SpherePoincare σ c`, in a two-point
Lipschitz form that needs no differentiability, and proves everything downstream of it: the
variance bound (`variance_stageErr_le`), its transport to the rotation stream through the
uniform law of §22 (`variance_stageErrSq_le`), and the concentration statement with its
constants (`meas_not_decayWithin_le_poincare`), feeding §13.  `SpherePoincare` is taken as a
hypothesis here; §26 derives it from the Gaussian inequality and §39 proves that
(`spherePoincare_sphereUniformAmbient`). -/

section PoincareForm

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]

/-- **The Poincaré inequality on the sphere, in the form the concentration remark needs.**
For a law `σ` and a constant `c`: whenever `F` satisfies the two-point Lipschitz bound
`|F x − F y| ≤ ‖x − y‖ (g x + g y)` with `g` continuous, so that the pointwise Lipschitz constant
of `F` is at most `2g`, the variance of `F` is at most `c ∫ (2g)²`.  For the uniform law on
`S^{d−1}` this holds with `c = 1/(d−1)`, the spectral gap of the sphere; it is the one classical
inequality this development takes as a hypothesis rather than proving. -/
structure SpherePoincare (σ : Measure E) (c : ℝ) : Prop where
  bound : ∀ (F g : E → ℝ), Continuous g →
    (∀ x y, |F x - F y| ≤ ‖x - y‖ * (g x + g y)) →
    MemLp F 2 σ → Integrable (fun x => g x ^ 2) σ →
    variance F σ ≤ c * ∫ x, (2 * g x) ^ 2 ∂σ

variable {C : Set E} {q : E → E}

/-- The two-point Lipschitz bound for the squared distance to a codebook, with slope function
`g = dist(·, C) = ‖x − q x‖`. -/
theorem abs_stageErr_sub_le (hq : IsNearest C q) (x y : E) :
    |‖x - q x‖ ^ 2 - ‖y - q y‖ ^ 2| ≤ ‖x - y‖ * (‖x - q x‖ + ‖y - q y‖) := by
  rw [norm_sub_eq_infDist hq, norm_sub_eq_infDist hq, sq_sub_sq, abs_mul,
    abs_of_nonneg (add_nonneg Metric.infDist_nonneg Metric.infDist_nonneg), mul_comm,
    add_comm (Metric.infDist x C)]
  refine mul_le_mul_of_nonneg_right ?_ (add_nonneg Metric.infDist_nonneg Metric.infDist_nonneg)
  have := (Metric.lipschitz_infDist_pt C).dist_le_mul x y
  rwa [Real.dist_eq, NNReal.coe_one, one_mul, dist_eq_norm] at this

theorem continuous_stageErrSlope (hq : IsNearest C q) : Continuous fun x => ‖x - q x‖ := by
  have : (fun x => ‖x - q x‖) = fun x => Metric.infDist x C := by
    funext x; exact norm_sub_eq_infDist hq x
  rw [this]; exact Metric.continuous_infDist_pt C

/-- **The variance bound.**  Under a law satisfying Poincaré with constant `c`, every
nearest-codeword stage has `Var ‖u − q u‖² ≤ 4 c ρ²`, with `ρ² = stageDistortion q σ`.  For the
uniform law and `c = 1/(d−1)` this is the `O(1/d)` the paper asserts, for scalar, `E₈` and
trellis stages alike, and with an explicit constant. -/
theorem variance_stageErr_le {σ : Measure E} [IsProbabilityMeasure σ] {c : ℝ}
    (hP : SpherePoincare σ c) (hq : IsNearest C q)
    (hmem : MemLp (fun x => ‖x - q x‖ ^ 2) 2 σ) :
    variance (fun x => ‖x - q x‖ ^ 2) σ ≤ 4 * c * stageDistortion q σ := by
  have h := hP.bound (fun x => ‖x - q x‖ ^ 2) (fun x => ‖x - q x‖) (continuous_stageErrSlope hq)
    (abs_stageErr_sub_le hq) hmem (hmem.integrable one_le_two)
  refine h.trans (le_of_eq ?_)
  have h4 : ∫ x, (2 * ‖x - q x‖) ^ 2 ∂σ = 4 * ∫ x, ‖x - q x‖ ^ 2 ∂σ := by
    rw [← integral_const_mul]
    exact integral_congr_ae (Filter.Eventually.of_forall fun x => by ring)
  rw [h4, stageDistortion]; ring

/-- A bounded codebook gives a bounded stage error on the sphere, hence one in every `L^p`. -/
theorem memLp_stageErr_of_bounded [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]
    (hq : IsNearest C q) (hC : Bornology.IsBounded C) (hmeas : Measurable q) :
    MemLp (fun x => ‖x - q x‖ ^ 2) 2 (sphereUniformAmbient E) := by
  obtain ⟨M, hM⟩ := hC.exists_norm_le
  refine (memLp_top_of_bound ((measurable_id.sub hmeas).norm.pow_const 2).aestronglyMeasurable
    ((1 + M) ^ 2) ?_).mono_exponent le_top
  filter_upwards [sphereUniformAmbient_ae_norm (E := E)] with x hx
  rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
  have : ‖x - q x‖ ≤ 1 + M :=
    (norm_sub_le _ _).trans (by rw [hx]; linarith [hM (q x) (hq x).1])
  exact pow_le_pow_left₀ (norm_nonneg _) this 2

end PoincareForm

section StreamConcentration

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] [FiniteDimensional ℝ E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G] [MeasurableMul G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {q : E → E} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G}
  [κ.IsMulRightInvariant] {x₀ : E}

omit [FiniteDimensional ℝ E] [MeasurableMul G] [IsProbabilityMeasure μ]
  [κ.IsMulRightInvariant] in
theorem measurable_unitVec_stream (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    Measurable fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p := by
  simp only [unitVec_rotScheme_eq_smul h.implements]
  exact (h.meas_R p).smul ((measurable_residDirPast h.meas_q p).comp
    (h.meas_W.prodMk (measurable_pi_lambda _ fun i => h.meas_R i)))

/-- The pass-`p` direction is a measure-preserving map onto the uniform law. -/
theorem measurePreserving_unitVec_stream (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1)
    (p : ℕ) :
    MeasurePreserving (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p) μ
      (sphereUniformAmbient E) :=
  ⟨measurable_unitVec_stream h p, law_unitVec_eq_sphereUniform h hx₀ p⟩

/-- **The per-pass variance bound on the stream**, transported from the uniform law. -/
theorem variance_stageErrSq_le (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1) {c : ℝ}
    (hP : SpherePoincare (sphereUniformAmbient E) c) {C : Set E} (hq : IsNearest C q)
    (hC : Bornology.IsBounded C) (p : ℕ) :
    variance (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p) μ
      ≤ 4 * c * stageDistortion q (sphereUniformAmbient E) := by
  haveI : Nontrivial E := ⟨⟨x₀, 0, by rintro rfl; simp at hx₀⟩⟩
  have hmp := measurePreserving_unitVec_stream h hx₀ p
  have hF : Measurable fun x : E => ‖x - q x‖ ^ 2 := (measurable_id.sub h.meas_q).norm.pow_const 2
  have heq : (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p)
      = (fun x : E => ‖x - q x‖ ^ 2) ∘ (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p) := rfl
  rw [heq, ← variance_map (by rw [hmp.map_eq]; exact hF.aemeasurable) hmp.measurable.aemeasurable,
    hmp.map_eq]
  exact variance_stageErr_le hP hq (memLp_stageErr_of_bounded hq hC h.meas_q)

/-- **Concentration of the decay law, with explicit constants.**  Under Poincaré with constant
`c` for the uniform law, a nearest-codeword stage with a bounded codebook, and `P` passes of a
rotation stream: the `P`-pass squared error lies within the compounded band `(1 ± ε)^P ρ^{2P}‖w‖²`
except on an event of probability at most `P · 4c / (ε² ρ²)`.  With `c = 1/(d−1)` this is the
paper's Remark (Concentration), with its rate `P/(d ε² ρ²)` made explicit, and it holds for
scalar, `E₈` and trellis stages alike. -/
theorem meas_not_decayWithin_le_poincare (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1)
    {c : ℝ} (hP : SpherePoincare (sphereUniformAmbient E) c) {C : Set E} (hq : IsNearest C q)
    (hC : Bornology.IsBounded C) {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε ≤ 1)
    (hρ : 0 < stageDistortion q (sphereUniformAmbient E)) (P : ℕ) :
    μ {ω | ¬ DecayWithin (rotScheme ι q (R ω)) (W ω)
        (Real.sqrt (stageDistortion q (sphereUniformAmbient E))) ε P}
      ≤ P * ENNReal.ofReal (4 * c / (ε ^ 2 * stageDistortion q (sphereUniformAmbient E))) := by
  haveI : Nontrivial E := ⟨⟨x₀, 0, by rintro rfl; simp at hx₀⟩⟩
  set D := stageDistortion q (sphereUniformAmbient E) with hD
  have hρ2 : Real.sqrt D ^ 2 = D := Real.sq_sqrt hρ.le
  have hmem : ∀ p < P, MemLp (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p) 2 μ := fun p _ =>
    (memLp_stageErr_of_bounded hq hC h.meas_q).comp_measurePreserving
      (measurePreserving_unitVec_stream h hx₀ p)
  have hmean : ∀ p < P, ∫ ω, stageErrSq (rotScheme ι q (R ω)) (W ω) p ∂μ = Real.sqrt D ^ 2 := by
    intro p _
    rw [hρ2, hD, ← h.orbitLaw_eq hx₀]
    exact integral_stageErr_eq_stageDistortion h p
  have hvar : ∀ p < P, variance (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p) μ
      ≤ 4 * c * D := fun p _ => variance_stageErrSq_le h hx₀ hP hq hC p
  have hρne : Real.sqrt D ≠ 0 := (Real.sqrt_pos.mpr hρ).ne'
  have key := meas_not_decayWithin_le_of_variance hε0 hε1 hρne hmem hmean hvar
  have hD0 : D ≠ 0 := hρ.ne'
  have hε : ε ≠ 0 := hε0.ne'
  have hconst : 4 * c * D / (ε * Real.sqrt D ^ 2) ^ 2 = 4 * c / (ε ^ 2 * D) := by
    rw [hρ2]; field_simp
  rw [← hconst]; exact key

end StreamConcentration
