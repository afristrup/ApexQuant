/-! ## §39  The general two-point Gaussian Poincaré inequality, proved

§26 stated `GaussPoincare γ`: for every `f ∈ L²(γ)` with `|f x − f y| ≤ ‖x − y‖ (G x + G y)` off
the origin, `G` continuous off the origin and square-integrable, `Var_γ f ≤ ∫ (2G)²`.  This
section proves it, by reduction to the `C¹` inequality of §32 through three approximations.
Mollification: for `g` bounded, compactly supported and continuous off the origin, the
convolution `φ_ε ⋆ g` with a normalised bump is smooth, satisfies the same two-point bound with
the mollified slope `φ_ε ⋆ G` everywhere (`abs_moll_sub_le`; the origin is a null set for the
convolution integral and needs no cutoff), so `‖(φ_ε ⋆ g)'‖ ≤ 2 φ_ε ⋆ G` (`norm_fderiv_moll_le`)
and §32 gives `Var (φ_ε ⋆ g) ≤ 4 ∫ (φ_ε ⋆ G)²` on a ball around the support
(`variance_moll_le`); Jensen and the comparison `ρ(y + t) ≤ e^{‖y‖‖t‖} ρ(y)` of the Gaussian
density bound this by `4 e^{O(ε)} ∫ G²` (`integral_indicator_moll_sq_le`), and dominated
convergence closes the mollification (`variance_le_of_twoPoint_compact`).  A smooth cutoff at
radius `R` costs `Cc M/(2R)` in the slope and `M² d/R²` in `L²` (`variance_clampF_le`), and
truncation at level `M` costs nothing in the slope and vanishes in `L²` as `M → ∞`
(`variance_le_of_twoPoint`).  Hence `gaussPoincare_gaussianE : GaussPoincare (gaussianE E)`,
`spherePoincare_sphereUniformAmbient`, and `meas_not_decayWithin_le_general`, the concentration
remark of §26 for an arbitrary codebook with its hypothesis discharged.  Nothing in the
development is now taken as a hypothesis beyond the rotation-stream model. -/

section MollifyPrelim

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- `∫⁻ h dγ = Z⁻¹ ∫⁻ h ρ dx`. -/
theorem lintegral_gaussianE_eq {h : E → ℝ≥0∞} (hh : Measurable h) :
    ∫⁻ x, h x ∂(gaussianE E) = (gaussRaw E Set.univ)⁻¹ * ∫⁻ x, h x * gaussDensity x := by
  unfold gaussianE
  rw [lintegral_smul_measure, gaussRaw, lintegral_withDensity_eq_lintegral_mul _
    measurable_gaussDensity hh]
  congr 1
  refine lintegral_congr fun x => ?_
  simp [mul_comm]

theorem gaussDensity_ge_of_norm_le {r : ℝ} {x : E} (hx : ‖x‖ ≤ r) :
    ENNReal.ofReal (Real.exp (-(r ^ 2) / 2)) ≤ gaussDensity x := by
  unfold gaussDensity
  refine ENNReal.ofReal_le_ofReal (Real.exp_le_exp.mpr ?_)
  have := norm_nonneg x
  nlinarith

/-- A function whose square is `γ`-integrable has locally integrable square for Lebesgue
measure. -/
theorem integrableOn_sq_of_gauss {G : E → ℝ} (hG : Measurable G)
    (hint : Integrable (fun x => G x ^ 2) (gaussianE E)) {k : Set E} (hk : IsCompact k) :
    IntegrableOn (fun x => G x ^ 2) k volume := by
  obtain ⟨r, hr⟩ := hk.isBounded.subset_closedBall (0 : E)
  set c : ℝ≥0∞ := ENNReal.ofReal (Real.exp (-(r ^ 2) / 2)) with hc
  have hc0 : c ≠ 0 := by rw [hc]; exact (ENNReal.ofReal_pos.mpr (Real.exp_pos _)).ne'
  have hcT : c ≠ ⊤ := ENNReal.ofReal_ne_top
  refine ⟨(hG.pow_const 2).aestronglyMeasurable.restrict, ?_⟩
  rw [hasFiniteIntegral_iff_ofReal (Filter.Eventually.of_forall fun x => sq_nonneg _)]
  have hfin : ∫⁻ x, ENNReal.ofReal (G x ^ 2) ∂(gaussianE E) < ⊤ := by
    have := hint.hasFiniteIntegral
    rwa [hasFiniteIntegral_iff_ofReal (Filter.Eventually.of_forall fun x => sq_nonneg _)] at this
  rw [lintegral_gaussianE_eq (hG.pow_const 2).ennreal_ofReal] at hfin
  have hZ : (gaussRaw E Set.univ)⁻¹ ≠ 0 := ENNReal.inv_ne_zero.mpr gaussRaw_univ_ne_top
  have hfin2 : ∫⁻ x, ENNReal.ofReal (G x ^ 2) * gaussDensity x < ⊤ := by
    by_contra h
    rw [not_lt, top_le_iff] at h
    rw [h, ENNReal.mul_top hZ] at hfin
    exact absurd hfin (lt_irrefl _)
  calc ∫⁻ x in k, ENNReal.ofReal (G x ^ 2) ∂volume
      ≤ ∫⁻ x in k, c⁻¹ * (ENNReal.ofReal (G x ^ 2) * gaussDensity x) ∂volume := by
        refine lintegral_mono_ae ?_
        rw [ae_restrict_iff' hk.measurableSet]
        refine Filter.Eventually.of_forall fun x hx => ?_
        have hx' : ‖x‖ ≤ r := by
          have := hr hx; rwa [Metric.mem_closedBall, dist_zero_right] at this
        have h1 : c ≤ gaussDensity x := gaussDensity_ge_of_norm_le hx'
        calc ENNReal.ofReal (G x ^ 2) = c⁻¹ * (ENNReal.ofReal (G x ^ 2) * c) := by
              rw [mul_comm, mul_assoc, ENNReal.mul_inv_cancel hc0 hcT, mul_one]
          _ ≤ c⁻¹ * (ENNReal.ofReal (G x ^ 2) * gaussDensity x) :=
              mul_le_mul' le_rfl (mul_le_mul' le_rfl h1)
    _ = c⁻¹ * ∫⁻ x in k, ENNReal.ofReal (G x ^ 2) * gaussDensity x ∂volume :=
        lintegral_const_mul _ ((hG.pow_const 2).ennreal_ofReal.mul measurable_gaussDensity)
    _ ≤ c⁻¹ * ∫⁻ x, ENNReal.ofReal (G x ^ 2) * gaussDensity x ∂volume :=
        mul_le_mul' le_rfl (setLIntegral_le_lintegral _ _)
    _ < ⊤ := ENNReal.mul_lt_top (ENNReal.inv_lt_top.mpr (pos_iff_ne_zero.mpr hc0)) hfin2

theorem locallyIntegrable_sq_of_gauss {G : E → ℝ} (hG : Measurable G)
    (hint : Integrable (fun x => G x ^ 2) (gaussianE E)) :
    LocallyIntegrable (fun x => G x ^ 2) volume :=
  locallyIntegrable_iff.mpr fun k hk => integrableOn_sq_of_gauss hG hint hk

theorem locallyIntegrable_of_gauss {G : E → ℝ} (hG : Measurable G)
    (hint : Integrable (fun x => G x ^ 2) (gaussianE E)) : LocallyIntegrable G volume := by
  refine locallyIntegrable_iff.mpr fun k hk => ?_
  have h2 := integrableOn_sq_of_gauss hG hint hk
  have h1 : IntegrableOn (fun _ : E => (1 : ℝ)) k volume :=
    integrableOn_const hk.measure_lt_top.ne
  refine (h1.add h2).mono' hG.aestronglyMeasurable.restrict
    (Filter.Eventually.of_forall fun x => ?_)
  show ‖G x‖ ≤ 1 + G x ^ 2
  rw [Real.norm_eq_abs]
  nlinarith [abs_nonneg (G x), sq_abs (G x), sq_nonneg (|G x| - 1)]

end MollifyPrelim

section Mollify

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- The mollification of `g` by the normalised bump `φ`. -/
noncomputable def moll (φ : ContDiffBump (0 : E)) (g : E → ℝ) : E → ℝ :=
  convolution (φ.normed volume) g (ContinuousLinearMap.lsmul ℝ ℝ) volume

theorem moll_apply (φ : ContDiffBump (0 : E)) (g : E → ℝ) (x : E) :
    moll φ g x = ∫ t, φ.normed volume t * g (x - t) := by
  simp only [moll, convolution_def, ContinuousLinearMap.lsmul_apply, smul_eq_mul]

theorem continuous_normed (φ : ContDiffBump (0 : E)) : Continuous (φ.normed volume) :=
  (φ.contDiff_normed (n := 0)).continuous

theorem hasCompactSupport_normed' (φ : ContDiffBump (0 : E)) :
    HasCompactSupport (φ.normed volume) := φ.hasCompactSupport_normed

theorem normed_eq_zero_of_le {φ : ContDiffBump (0 : E)} {t : E} (ht : φ.rOut ≤ ‖t‖) :
    φ.normed volume t = 0 := by
  by_contra h
  have : t ∈ Function.support (φ.normed volume) := h
  rw [φ.support_normed_eq, Metric.mem_ball, dist_zero_right] at this
  linarith

/-- The convolution integrand is integrable for locally integrable `g`. -/
theorem integrable_moll_integrand (φ : ContDiffBump (0 : E)) {g : E → ℝ}
    (hg : LocallyIntegrable g volume) (x : E) :
    Integrable (fun t => φ.normed volume t * g (x - t)) volume :=
  (hasCompactSupport_normed' φ).convolutionExists_left (ContinuousLinearMap.lsmul ℝ ℝ)
    (continuous_normed φ) hg x

theorem moll_nonneg (φ : ContDiffBump (0 : E)) {G : E → ℝ} (hG0 : ∀ x, 0 ≤ G x) (x : E) :
    0 ≤ moll φ G x := by
  rw [moll_apply]
  exact integral_nonneg fun t => mul_nonneg (φ.nonneg_normed (μ := volume) t) (hG0 _)

theorem abs_moll_le (φ : ContDiffBump (0 : E)) {g : E → ℝ} (hg : LocallyIntegrable g volume)
    {M : ℝ} (hM : ∀ x, |g x| ≤ M) (x : E) : |moll φ g x| ≤ M := by
  rw [moll_apply]
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  calc |∫ t, φ.normed volume t * g (x - t)| ≤ ∫ t, |φ.normed volume t * g (x - t)| :=
        abs_integral_le_integral_abs
    _ ≤ ∫ t, φ.normed volume t * M := by
        refine integral_mono_of_nonneg (Filter.Eventually.of_forall fun t => abs_nonneg _)
          ((φ.integrable_normed).mul_const M) (Filter.Eventually.of_forall fun t => ?_)
        show |φ.normed volume t * g (x - t)| ≤ φ.normed volume t * M
        rw [abs_mul, abs_of_nonneg (φ.nonneg_normed (μ := volume) t)]
        exact mul_le_mul_of_nonneg_left (hM _) (φ.nonneg_normed (μ := volume) t)
    _ = M := by rw [integral_mul_const, φ.integral_normed, one_mul]

/-- The mollification vanishes at distance `rOut` from the support. -/
theorem moll_eq_zero_of_norm_ge (φ : ContDiffBump (0 : E)) {g : E → ℝ} {R : ℝ}
    (hg : ∀ x, R < ‖x‖ → g x = 0) {x : E} (hx : R + φ.rOut ≤ ‖x‖) : moll φ g x = 0 := by
  rw [moll_apply]
  refine integral_eq_zero_of_ae (Filter.Eventually.of_forall fun t => ?_)
  show φ.normed volume t * g (x - t) = 0
  by_cases ht : φ.rOut ≤ ‖t‖
  · rw [normed_eq_zero_of_le ht, zero_mul]
  · push_neg at ht
    rw [hg (x - t) (by
      have := norm_sub_norm_le x t
      linarith), mul_zero]

/-- **The two-point bound passes to the mollification**, with slope `moll φ G`. -/
theorem abs_moll_sub_le (φ : ContDiffBump (0 : E)) {g G : E → ℝ}
    (hg : LocallyIntegrable g volume) (hG : LocallyIntegrable G volume)
    (h : ∀ x y, x ≠ 0 → y ≠ 0 → |g x - g y| ≤ ‖x - y‖ * (G x + G y)) (x y : E) :
    |moll φ g x - moll φ g y| ≤ ‖x - y‖ * (moll φ G x + moll φ G y) := by
  have hix := integrable_moll_integrand φ hg x
  have hiy := integrable_moll_integrand φ hg y
  have hGx := integrable_moll_integrand φ hG x
  have hGy := integrable_moll_integrand φ hG y
  simp only [moll_apply]
  rw [← integral_sub hix hiy, ← integral_add hGx hGy, ← integral_const_mul]
  have hae : ∀ᵐ t ∂(volume : Measure E), x - t ≠ 0 ∧ y - t ≠ 0 := by
    have h1 : ∀ᵐ t ∂(volume : Measure E), t ≠ x := by
      rw [ae_iff]; simp only [not_not]
      exact measure_singleton x
    have h2 : ∀ᵐ t ∂(volume : Measure E), t ≠ y := by
      rw [ae_iff]; simp only [not_not]
      exact measure_singleton y
    filter_upwards [h1, h2] with t ht1 ht2
    exact ⟨sub_ne_zero.mpr (Ne.symm ht1), sub_ne_zero.mpr (Ne.symm ht2)⟩
  refine abs_integral_le_integral_abs.trans (integral_mono_of_nonneg
    (Filter.Eventually.of_forall fun t => abs_nonneg _) ((hGx.add hGy).const_mul _) ?_)
  filter_upwards [hae] with t ⟨ht1, ht2⟩
  have hφ := φ.nonneg_normed (μ := volume) t
  show |φ.normed volume t * g (x - t) - φ.normed volume t * g (y - t)|
    ≤ ‖x - y‖ * (φ.normed volume t * G (x - t) + φ.normed volume t * G (y - t))
  rw [← mul_sub, abs_mul, abs_of_nonneg hφ]
  have hxy : ‖x - t - (y - t)‖ = ‖x - y‖ := by rw [sub_sub_sub_cancel_right]
  calc φ.normed volume t * |g (x - t) - g (y - t)|
      ≤ φ.normed volume t * (‖x - t - (y - t)‖ * (G (x - t) + G (y - t))) :=
        mul_le_mul_of_nonneg_left (h _ _ ht1 ht2) hφ
    _ = ‖x - y‖ * (φ.normed volume t * G (x - t) + φ.normed volume t * G (y - t)) := by
        rw [hxy]; ring

theorem contDiff_moll (φ : ContDiffBump (0 : E)) {g : E → ℝ} (hg : LocallyIntegrable g volume) :
    ContDiff ℝ 1 (moll φ g) :=
  (hasCompactSupport_normed' φ).contDiff_convolution_left _ (φ.contDiff_normed) hg

theorem continuous_moll (φ : ContDiffBump (0 : E)) {g : E → ℝ} (hg : LocallyIntegrable g volume) :
    Continuous (moll φ g) :=
  (hasCompactSupport_normed' φ).continuous_convolution_left _ (continuous_normed φ) hg

/-- **The gradient of the mollification is bounded by twice the mollified slope.** -/
theorem norm_fderiv_moll_le (φ : ContDiffBump (0 : E)) {g G : E → ℝ}
    (hg : LocallyIntegrable g volume) (hG : LocallyIntegrable G volume) (hG0 : ∀ x, 0 ≤ G x)
    (h : ∀ x y, x ≠ 0 → y ≠ 0 → |g x - g y| ≤ ‖x - y‖ * (G x + G y)) {x : E} (hx : x ≠ 0) :
    ‖fderiv ℝ (moll φ g) x‖ ≤ 2 * moll φ G x :=
  norm_fderiv_le_of_twoPoint hx (continuous_moll φ hG).continuousAt (moll_nonneg φ hG0 x)
    (fun z w hz hw => abs_moll_sub_le φ hg hG h z w)
    (((contDiff_moll φ hg).differentiable one_ne_zero) x).hasFDerivAt


theorem hasCompactSupport_of_vanish {g : E → ℝ} {R : ℝ} (hR : ∀ x, R < ‖x‖ → g x = 0) :
    HasCompactSupport g :=
  HasCompactSupport.of_support_subset_isCompact (isCompact_closedBall (0 : E) R) fun x hx => by
    rw [Metric.mem_closedBall, dist_zero_right]
    by_contra h; push_neg at h
    exact hx (hR x h)

/-- **Step L4: the variance of the mollification** is controlled by the mollified slope on a
ball around the support. -/
theorem variance_moll_le (hd : 2 ≤ Module.finrank ℝ E) (φ : ContDiffBump (0 : E)) {g G : E → ℝ}
    (hg : LocallyIntegrable g volume) {M : ℝ} (hM : ∀ x, |g x| ≤ M) {R : ℝ}
    (hR : ∀ x, R < ‖x‖ → g x = 0) (hG : LocallyIntegrable G volume) (hG0 : ∀ x, 0 ≤ G x)
    (h : ∀ x y, x ≠ 0 → y ≠ 0 → |g x - g y| ≤ ‖x - y‖ * (G x + G y)) :
    variance (moll φ g) (gaussianE E)
      ≤ 4 * ∫ x, (Metric.ball (0 : E) (R + φ.rOut + 1)).indicator (fun x => moll φ G x ^ 2) x
          ∂(gaussianE E) := by
  have hdiff : Differentiable ℝ (moll φ g) := (contDiff_moll φ hg).differentiable one_ne_zero
  have hbM : ∀ z, |moll φ g z| ≤ M := abs_moll_le φ hg hM
  have hsupp : HasCompactSupport (moll φ g) :=
    (hasCompactSupport_normed' φ).convolution _ (hasCompactSupport_of_vanish hR)
  obtain ⟨K, hK⟩ := ((contDiff_moll φ hg).continuous_fderiv one_ne_zero).bounded_above_of_compact_support
    (hsupp.fderiv ℝ)
  refine (variance_le_integral_norm_fderiv_sq hd hdiff hbM hK).trans ?_
  have hcont : Continuous (moll φ G) := continuous_moll φ hG
  have hint : Integrable ((Metric.ball (0 : E) (R + φ.rOut + 1)).indicator
      (fun x => moll φ G x ^ 2)) (gaussianE E) := by
    rw [integrable_indicator_iff measurableSet_ball]
    exact ((hcont.pow 2).continuousOn.integrableOn_compact (isCompact_closedBall _ _)).mono_set
      Metric.ball_subset_closedBall
  rw [← integral_const_mul]
  refine integral_mono_ae (integrable_norm_fderiv_sq hK measurable_id) (hint.const_mul _) ?_
  filter_upwards [gaussianE_ae_ne_zero (E := E)] with x hx
  by_cases hxb : x ∈ Metric.ball (0 : E) (R + φ.rOut + 1)
  · rw [Set.indicator_of_mem hxb]
    have := norm_fderiv_moll_le φ hg hG hG0 h hx
    have h0 := moll_nonneg φ hG0 x
    nlinarith [norm_nonneg (fderiv ℝ (moll φ g) x)]
  · rw [Set.indicator_of_notMem hxb, mul_zero]
    rw [Metric.mem_ball, dist_zero_right, not_lt] at hxb
    have hev : moll φ g =ᶠ[𝓝 x] fun _ => 0 := by
      have hopen : {z : E | R + φ.rOut < ‖z‖} ∈ 𝓝 x :=
        (isOpen_lt continuous_const continuous_norm).mem_nhds (by
          show R + φ.rOut < ‖x‖; linarith)
      filter_upwards [hopen] with z hz
      exact moll_eq_zero_of_norm_ge φ hR (le_of_lt hz)
    rw [hev.fderiv_eq, fderiv_const_apply]
    simp

/-- The Gaussian density moves by at most `e^{‖y‖‖t‖}` under a shift by `t`. -/
theorem gaussDensity_add_le (y t : E) :
    gaussDensity (y + t) ≤ ENNReal.ofReal (Real.exp (‖y‖ * ‖t‖)) * gaussDensity y := by
  unfold gaussDensity
  rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
  refine ENNReal.ofReal_le_ofReal (Real.exp_le_exp.mpr ?_)
  have h1 : |‖y‖ - ‖t‖| ≤ ‖y + t‖ := by
    have := abs_norm_sub_norm_le y (-t)
    rwa [norm_neg, sub_neg_eq_add] at this
  have h5 : (‖y‖ - ‖t‖) ^ 2 ≤ ‖y + t‖ ^ 2 := by
    rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) h1 2
  have h3 := norm_nonneg t
  nlinarith [sq_nonneg ‖t‖]

/-- Jensen: `(moll φ G x)² ≤ moll φ (G²) x`. -/
theorem sq_moll_le_moll_sq (φ : ContDiffBump (0 : E)) {G : E → ℝ} (hGm : Measurable G)
    (hG2 : LocallyIntegrable (fun x => G x ^ 2) volume) (x : E) :
    moll φ G x ^ 2 ≤ moll φ (fun t => G t ^ 2) x := by
  set ν : Measure E := volume.withDensity fun t => ENNReal.ofReal (φ.normed volume t) with hν
  have hφm : Measurable (φ.normed volume) := (continuous_normed φ).measurable
  haveI : IsProbabilityMeasure ν := by
    constructor
    rw [hν, withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ,
      ← ofReal_integral_eq_lintegral_ofReal φ.integrable_normed
        (Filter.Eventually.of_forall (φ.nonneg_normed (μ := volume))), φ.integral_normed,
      ENNReal.ofReal_one]
  have hfin : ∀ᵐ t ∂(volume : Measure E), ENNReal.ofReal (φ.normed volume t) < ⊤ :=
    Filter.Eventually.of_forall fun _ => ENNReal.ofReal_lt_top
  have hconv : ∀ k : E → ℝ, ∫ t, k t ∂ν = ∫ t, φ.normed volume t * k t := by
    intro k
    rw [hν, integral_withDensity_eq_integral_toReal_smul hφm.ennreal_ofReal hfin]
    refine integral_congr_ae (Filter.Eventually.of_forall fun t => ?_)
    show (ENNReal.ofReal (φ.normed volume t)).toReal • k t = φ.normed volume t * k t
    rw [ENNReal.toReal_ofReal (φ.nonneg_normed (μ := volume) t), smul_eq_mul]
  have hint2 : Integrable (fun t => (G (x - t)) ^ 2) ν := by
    rw [hν, integrable_withDensity_iff hφm.ennreal_ofReal hfin]
    have := integrable_moll_integrand φ hG2 x
    refine this.congr (Filter.Eventually.of_forall fun t => ?_)
    simp only
    rw [ENNReal.toReal_ofReal (φ.nonneg_normed (μ := volume) t), mul_comm]
  have hmem : MemLp (fun t => G (x - t)) 2 ν := by
    refine (memLp_two_iff_integrable_sq ?_).mpr hint2
    exact (hGm.comp (measurable_const.sub measurable_id)).aestronglyMeasurable
  have hvar := variance_nonneg (fun t => G (x - t)) ν
  rw [variance_eq_sub hmem] at hvar
  have e1 : ∫ t, ((fun t => G (x - t)) ^ 2) t ∂ν = moll φ (fun t => G t ^ 2) x := by
    rw [moll_apply, ← hconv]; rfl
  have e2 : ∫ t, G (x - t) ∂ν = moll φ G x := by rw [moll_apply, ← hconv]
  rw [e1, e2] at hvar
  linarith

/-- **Step L5: the mollified slope, squared and cut to a ball, against `γ`.** -/
theorem integral_indicator_moll_sq_le (φ : ContDiffBump (0 : E)) {G : E → ℝ} (hGm : Measurable G)
    (hG0 : ∀ x, 0 ≤ G x) (hint : Integrable (fun x => G x ^ 2) (gaussianE E)) {S : ℝ}
    (hS : 0 ≤ S) :
    ∫ x, (Metric.ball (0 : E) S).indicator (fun x => moll φ G x ^ 2) x ∂(gaussianE E)
      ≤ Real.exp ((S + φ.rOut) * φ.rOut) * ∫ x, G x ^ 2 ∂(gaussianE E) := by
  have hG2 : LocallyIntegrable (fun x => G x ^ 2) volume := locallyIntegrable_sq_of_gauss hGm hint
  have hG : LocallyIntegrable G volume := locallyIntegrable_of_gauss hGm hint
  have hφm : Measurable (φ.normed volume) := (continuous_normed φ).measurable
  have hcont : Continuous (moll φ G) := continuous_moll φ hG
  set c : ℝ := (S + φ.rOut) * φ.rOut with hc
  have hc0 : 0 ≤ c := mul_nonneg (add_nonneg hS φ.rOut_pos.le) φ.rOut_pos.le
  -- the target in `lintegral` form
  have hR : ∫ x, G x ^ 2 ∂(gaussianE E)
      = (∫⁻ x, ENNReal.ofReal (G x ^ 2) ∂(gaussianE E)).toReal :=
    integral_eq_lintegral_of_nonneg_ae (Filter.Eventually.of_forall fun x => sq_nonneg _)
      (hGm.pow_const 2).aestronglyMeasurable
  have hRfin : ∫⁻ x, ENNReal.ofReal (G x ^ 2) ∂(gaussianE E) ≠ ⊤ := by
    have := hint.hasFiniteIntegral
    rw [hasFiniteIntegral_iff_ofReal (Filter.Eventually.of_forall fun x => sq_nonneg _)] at this
    exact this.ne
  have hL : ∫ x, (Metric.ball (0 : E) S).indicator (fun x => moll φ G x ^ 2) x ∂(gaussianE E)
      = (∫⁻ x, ENNReal.ofReal ((Metric.ball (0 : E) S).indicator (fun x => moll φ G x ^ 2) x)
          ∂(gaussianE E)).toReal :=
    integral_eq_lintegral_of_nonneg_ae
      (Filter.Eventually.of_forall fun x => Set.indicator_nonneg (fun _ _ => sq_nonneg _) _)
      (((hcont.pow 2).measurable).indicator measurableSet_ball).aestronglyMeasurable
  rw [hL, hR, ← ENNReal.toReal_ofReal (Real.exp_pos c).le, ← ENNReal.toReal_mul]
  refine ENNReal.toReal_mono (ENNReal.mul_ne_top ENNReal.ofReal_ne_top hRfin) ?_
  -- Jensen pointwise, then the density comparison
  have hJ : ∀ x, ENNReal.ofReal ((Metric.ball (0 : E) S).indicator (fun x => moll φ G x ^ 2) x)
      ≤ (Metric.ball (0 : E) S).indicator (fun _ => (1 : ℝ≥0∞)) x *
          ∫⁻ t, ENNReal.ofReal (φ.normed volume t) * ENNReal.ofReal (G (x - t) ^ 2) := by
    intro x
    by_cases hx : x ∈ Metric.ball (0 : E) S
    · rw [Set.indicator_of_mem hx, Set.indicator_of_mem hx, one_mul]
      refine (ENNReal.ofReal_le_ofReal (sq_moll_le_moll_sq φ hGm hG2 x)).trans (le_of_eq ?_)
      rw [moll_apply, ofReal_integral_eq_lintegral_ofReal (integrable_moll_integrand φ hG2 x)
        (Filter.Eventually.of_forall fun t =>
          mul_nonneg (φ.nonneg_normed (μ := volume) t) (sq_nonneg _))]
      refine lintegral_congr fun t => ?_
      rw [ENNReal.ofReal_mul (φ.nonneg_normed (μ := volume) t)]
    · rw [Set.indicator_of_notMem hx, Set.indicator_of_notMem hx, zero_mul, ENNReal.ofReal_zero]
  refine (lintegral_mono hJ).trans ?_
  have hmeasG : Measurable fun p : E × E => ENNReal.ofReal (G (p.1 - p.2) ^ 2) :=
    ((hGm.comp (measurable_fst.sub measurable_snd)).pow_const 2).ennreal_ofReal
  -- write the left side with the Gaussian density and swap the integrals
  have hmeas2 : Measurable fun x : E =>
      ∫⁻ t, ENNReal.ofReal (φ.normed volume t) * ENNReal.ofReal (G (x - t) ^ 2) :=
    Measurable.lintegral_prod_right' (f := fun p : E × E =>
      ENNReal.ofReal (φ.normed volume p.2) * ENNReal.ofReal (G (p.1 - p.2) ^ 2))
      ((hφm.ennreal_ofReal.comp measurable_snd).mul hmeasG)
  have hmeas1 : Measurable fun x : E => (Metric.ball (0 : E) S).indicator (fun _ => (1 : ℝ≥0∞)) x *
      ∫⁻ t, ENNReal.ofReal (φ.normed volume t) * ENNReal.ofReal (G (x - t) ^ 2) :=
    (measurable_const.indicator measurableSet_ball).mul hmeas2
  rw [lintegral_gaussianE_eq hmeas1]
  have hmA : ∀ x : E, Measurable fun t : E =>
      ENNReal.ofReal (φ.normed volume t) * ENNReal.ofReal (G (x - t) ^ 2) := fun x =>
    (hφm.ennreal_ofReal).mul ((hGm.comp (measurable_const.sub measurable_id)).pow_const 2).ennreal_ofReal
  have hmB : ∀ t : E, Measurable fun x : E =>
      (Metric.ball (0 : E) S).indicator (fun _ => (1 : ℝ≥0∞)) x * gaussDensity x *
        ENNReal.ofReal (G (x - t) ^ 2) := fun t =>
    ((measurable_const.indicator measurableSet_ball).mul measurable_gaussDensity).mul
      ((hGm.comp (measurable_id.sub measurable_const)).pow_const 2).ennreal_ofReal
  have hswap : ∫⁻ x, ((Metric.ball (0 : E) S).indicator (fun _ => (1 : ℝ≥0∞)) x *
        ∫⁻ t, ENNReal.ofReal (φ.normed volume t) * ENNReal.ofReal (G (x - t) ^ 2)) * gaussDensity x
      = ∫⁻ t, ENNReal.ofReal (φ.normed volume t) *
          ∫⁻ x, (Metric.ball (0 : E) S).indicator (fun _ => (1 : ℝ≥0∞)) x * gaussDensity x *
            ENNReal.ofReal (G (x - t) ^ 2) := by
    have e : ∀ x, ((Metric.ball (0 : E) S).indicator (fun _ => (1 : ℝ≥0∞)) x *
        ∫⁻ t, ENNReal.ofReal (φ.normed volume t) * ENNReal.ofReal (G (x - t) ^ 2)) * gaussDensity x
        = ∫⁻ t, ENNReal.ofReal (φ.normed volume t) *
            ((Metric.ball (0 : E) S).indicator (fun _ => (1 : ℝ≥0∞)) x * gaussDensity x *
              ENNReal.ofReal (G (x - t) ^ 2)) := by
      intro x
      rw [mul_comm _ (gaussDensity x), ← mul_assoc, ← lintegral_const_mul _ (hmA x)]
      refine lintegral_congr fun t => ?_
      ring
    simp_rw [e]
    rw [lintegral_lintegral_swap]
    · refine lintegral_congr fun t => ?_
      rw [lintegral_const_mul _ (hmB t)]
    · exact ((hφm.ennreal_ofReal.comp measurable_snd).mul
        (((measurable_const.indicator measurableSet_ball).comp measurable_fst |>.mul
          (measurable_gaussDensity.comp measurable_fst)).mul hmeasG)).aemeasurable
  rw [hswap]
  -- shift `x = y + t`, then compare densities
  have hshift : ∀ t : E, ∫⁻ x, (Metric.ball (0 : E) S).indicator (fun _ => (1 : ℝ≥0∞)) x *
        gaussDensity x * ENNReal.ofReal (G (x - t) ^ 2)
      = ∫⁻ y, (Metric.ball (0 : E) S).indicator (fun _ => (1 : ℝ≥0∞)) (y + t) *
          gaussDensity (y + t) * ENNReal.ofReal (G y ^ 2) := by
    intro t
    rw [← lintegral_add_right_eq_self (fun x => (Metric.ball (0 : E) S).indicator
      (fun _ => (1 : ℝ≥0∞)) x * gaussDensity x * ENNReal.ofReal (G (x - t) ^ 2)) t]
    simp only [add_sub_cancel_right]
  simp_rw [hshift]
  have hbound : ∀ t y : E, ENNReal.ofReal (φ.normed volume t) *
        ((Metric.ball (0 : E) S).indicator (fun _ => (1 : ℝ≥0∞)) (y + t) * gaussDensity (y + t) *
          ENNReal.ofReal (G y ^ 2))
      ≤ ENNReal.ofReal (φ.normed volume t) *
          (ENNReal.ofReal (Real.exp c) * gaussDensity y * ENNReal.ofReal (G y ^ 2)) := by
    intro t y
    by_cases ht : φ.rOut ≤ ‖t‖
    · rw [normed_eq_zero_of_le ht, ENNReal.ofReal_zero, zero_mul, zero_mul]
    · push_neg at ht
      refine mul_le_mul' le_rfl (mul_le_mul' ?_ le_rfl)
      by_cases hy : y + t ∈ Metric.ball (0 : E) S
      · rw [Set.indicator_of_mem hy, one_mul]
        rw [Metric.mem_ball, dist_zero_right] at hy
        have hy' : ‖y‖ < S + φ.rOut := by
          have := norm_sub_le (y + t) t; rw [add_sub_cancel_right] at this; linarith
        refine (gaussDensity_add_le y t).trans (mul_le_mul' (ENNReal.ofReal_le_ofReal
          (Real.exp_le_exp.mpr ?_)) le_rfl)
        rw [hc]
        have := norm_nonneg y
        have := norm_nonneg t
        nlinarith
      · rw [Set.indicator_of_notMem hy, zero_mul]; exact (zero_le : (0 : ℝ≥0∞) ≤ _)
  calc ((gaussRaw E) Set.univ)⁻¹ * ∫⁻ t, ENNReal.ofReal (φ.normed volume t) *
        ∫⁻ y, (Metric.ball (0 : E) S).indicator (fun _ => (1 : ℝ≥0∞)) (y + t) *
          gaussDensity (y + t) * ENNReal.ofReal (G y ^ 2)
      ≤ ((gaussRaw E) Set.univ)⁻¹ * ∫⁻ t, ENNReal.ofReal (φ.normed volume t) *
          ∫⁻ y, ENNReal.ofReal (Real.exp c) * gaussDensity y * ENNReal.ofReal (G y ^ 2) := by
        refine mul_le_mul' le_rfl (lintegral_mono fun t => ?_)
        have hmC : Measurable fun y : E =>
            (Metric.ball (0 : E) S).indicator (fun _ => (1 : ℝ≥0∞)) (y + t) * gaussDensity (y + t) *
              ENNReal.ofReal (G y ^ 2) :=
          (((measurable_const.indicator measurableSet_ball).comp (measurable_id.add_const t)).mul
            (measurable_gaussDensity.comp (measurable_id.add_const t))).mul
            (hGm.pow_const 2).ennreal_ofReal
        have hmD : Measurable fun y : E =>
            ENNReal.ofReal (Real.exp c) * gaussDensity y * ENNReal.ofReal (G y ^ 2) :=
          (measurable_const.mul measurable_gaussDensity).mul (hGm.pow_const 2).ennreal_ofReal
        rw [← lintegral_const_mul _ hmC, ← lintegral_const_mul _ hmD]
        exact lintegral_mono fun y => hbound t y
    _ = ENNReal.ofReal (Real.exp c) * ∫⁻ x, ENNReal.ofReal (G x ^ 2) ∂(gaussianE E) := by
        have hmE : Measurable fun y : E => gaussDensity y * ENNReal.ofReal (G y ^ 2) :=
          measurable_gaussDensity.mul (hGm.pow_const 2).ennreal_ofReal
        have e1 : ∫⁻ y, ENNReal.ofReal (Real.exp c) * gaussDensity y * ENNReal.ofReal (G y ^ 2)
            = ENNReal.ofReal (Real.exp c) * ∫⁻ y, gaussDensity y * ENNReal.ofReal (G y ^ 2) := by
          rw [← lintegral_const_mul _ hmE]
          refine lintegral_congr fun y => ?_; ring
        rw [e1, lintegral_mul_const _ hφm.ennreal_ofReal, ← ofReal_integral_eq_lintegral_ofReal
            φ.integrable_normed (Filter.Eventually.of_forall (φ.nonneg_normed (μ := volume))),
          φ.integral_normed, ENNReal.ofReal_one, one_mul,
          lintegral_gaussianE_eq (hGm.pow_const 2).ennreal_ofReal]
        have : ∀ x, gaussDensity x * ENNReal.ofReal (G x ^ 2)
            = ENNReal.ofReal (G x ^ 2) * gaussDensity x := fun x => mul_comm _ _
        simp_rw [this]
        ring

/-- The bump of outer radius `ε`. -/
noncomputable def bump (ε : ℝ) (hε : 0 < ε) : ContDiffBump (0 : E) :=
  ⟨ε / 2, ε, by positivity, by linarith⟩

theorem bump_rOut {ε : ℝ} (hε : 0 < ε) : (bump (E := E) ε hε).rOut = ε := rfl

theorem locallyIntegrable_of_bounded {g : E → ℝ} (hgm : Measurable g) {M : ℝ}
    (hM : ∀ x, |g x| ≤ M) : LocallyIntegrable g volume :=
  locallyIntegrable_iff.mpr fun k hk => Measure.integrableOn_of_bounded hk.measure_lt_top.ne
    hgm.aestronglyMeasurable (Filter.Eventually.of_forall fun x => by
      rw [Real.norm_eq_abs]; exact hM x)

/-- **Stage 1: Poincaré for a bounded, compactly supported function with a two-point bound off
the origin**, `Var g ≤ 4 ∫ G²`, by mollification. -/
theorem variance_le_of_twoPoint_compact (hd : 2 ≤ Module.finrank ℝ E) {g G : E → ℝ}
    (hgm : Measurable g) (hgc : ContinuousOn g {x | x ≠ 0}) {M : ℝ} (hM : ∀ x, |g x| ≤ M)
    {R : ℝ} (hR0 : 0 ≤ R) (hR : ∀ x, R < ‖x‖ → g x = 0) (hGm : Measurable G) (hG0 : ∀ x, 0 ≤ G x)
    (hint : Integrable (fun x => G x ^ 2) (gaussianE E))
    (h : ∀ x y, x ≠ 0 → y ≠ 0 → |g x - g y| ≤ ‖x - y‖ * (G x + G y)) :
    variance g (gaussianE E) ≤ 4 * ∫ x, G x ^ 2 ∂(gaussianE E) := by
  have hg : LocallyIntegrable g volume := locallyIntegrable_of_bounded hgm hM
  have hG : LocallyIntegrable G volume := locallyIntegrable_of_gauss hGm hint
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  set I := ∫ x, G x ^ 2 ∂(gaussianE E) with hI
  have hI0 : 0 ≤ I := integral_nonneg fun x => sq_nonneg _
  -- the mollified variance bound, for `ε ∈ (0, 1]`
  have hvar : ∀ ε : ℝ, ∀ hε : 0 < ε, ε ≤ 1 →
      variance (moll (bump ε hε) g) (gaussianE E)
        ≤ 4 * (Real.exp ((R + 2 + ε) * ε) * I) := by
    intro ε hε hε1
    refine (variance_moll_le hd (bump ε hε) hg hM hR hG hG0 h).trans ?_
    refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
    have hcont : Continuous (moll (bump ε hε) G) := continuous_moll _ hG
    have hmono : ∫ x, (Metric.ball (0 : E) (R + (bump (E := E) ε hε).rOut + 1)).indicator
          (fun x => moll (bump ε hε) G x ^ 2) x ∂(gaussianE E)
        ≤ ∫ x, (Metric.ball (0 : E) (R + 2)).indicator
          (fun x => moll (bump ε hε) G x ^ 2) x ∂(gaussianE E) := by
      have hint2 : Integrable ((Metric.ball (0 : E) (R + 2)).indicator
          (fun x => moll (bump ε hε) G x ^ 2)) (gaussianE E) := by
        rw [integrable_indicator_iff measurableSet_ball]
        exact ((hcont.pow 2).continuousOn.integrableOn_compact (isCompact_closedBall _ _)).mono_set
          Metric.ball_subset_closedBall
      refine integral_mono_of_nonneg (Filter.Eventually.of_forall fun x =>
        Set.indicator_nonneg (fun _ _ => sq_nonneg _) _) hint2
        (Filter.Eventually.of_forall fun x => ?_)
      have hsub : Metric.ball (0 : E) (R + (bump ε hε).rOut + 1) ⊆ Metric.ball (0 : E) (R + 2) :=
        Metric.ball_subset_ball (by rw [bump_rOut (E := E)]; linarith)
      by_cases hx : x ∈ Metric.ball (0 : E) (R + (bump (E := E) ε hε).rOut + 1)
      · rw [Set.indicator_of_mem hx, Set.indicator_of_mem (hsub hx)]
      · rw [Set.indicator_of_notMem hx]
        exact Set.indicator_nonneg (fun _ _ => sq_nonneg _) _
    refine hmono.trans ?_
    have := integral_indicator_moll_sq_le (bump ε hε) hGm hG0 hint (S := R + 2) (by linarith)
    rw [bump_rOut (E := E)] at this
    exact this
  -- closeness in `L²`: `∫ (g − moll g)² → 0`
  have hclose : Tendsto (fun ε : ℝ => if hε : 0 < ε then
      ∫ x, (g x - moll (bump ε hε) g x) ^ 2 ∂(gaussianE E) else 0) (𝓝[>] 0) (𝓝 0) := by
    have key : Tendsto (fun ε : ℝ => ∫ x, (if hε : 0 < ε then
        (g x - moll (bump ε hε) g x) ^ 2 else 0) ∂(gaussianE E)) (𝓝[>] 0) (𝓝 (∫ _, (0 : ℝ) ∂(gaussianE E))) := by
      refine tendsto_integral_filter_of_dominated_convergence (fun _ => (2 * M) ^ 2) ?_ ?_
        (integrable_const _) ?_
      · refine Filter.Eventually.of_forall fun ε => ?_
        by_cases hε : 0 < ε
        · simp only [hε, dite_true]
          exact Measurable.aestronglyMeasurable
            ((hgm.sub (continuous_moll (bump ε hε) hg).measurable).pow_const 2)
        · simp only [hε, dite_false]; exact aestronglyMeasurable_const
      · refine Filter.Eventually.of_forall fun ε => Filter.Eventually.of_forall fun x => ?_
        by_cases hε : 0 < ε
        · simp only [hε, dite_true]
          rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
          have h1 := hM x
          have h2 := abs_moll_le (bump ε hε) hg hM x
          have : |g x - moll (bump ε hε) g x| ≤ 2 * M := by
            calc |g x - moll (bump ε hε) g x| ≤ |g x| + |moll (bump ε hε) g x| := abs_sub _ _
              _ ≤ 2 * M := by linarith
          calc (g x - moll (bump ε hε) g x) ^ 2 = |g x - moll (bump ε hε) g x| ^ 2 := (sq_abs _).symm
            _ ≤ (2 * M) ^ 2 := pow_le_pow_left₀ (abs_nonneg _) this 2
        · simp only [hε, dite_false]; rw [norm_zero]; positivity
      · filter_upwards [gaussianE_ae_ne_zero (E := E)] with x hx
        have hten : Tendsto (fun ε : ℝ => if hε : 0 < ε then moll (bump ε hε) g x else g x)
            (𝓝[>] 0) (𝓝 (g x)) := by
          -- on `𝓝[>] 0` the branch is the mollification, which converges by continuity at `x`
          have hcv := ContDiffBump.convolution_tendsto_right (μ := (volume : Measure E))
            (φ := fun ε : ℝ => if hε : 0 < ε then bump ε hε else bump 1 one_pos)
            (g := fun _ => g) (k := fun _ => x) (x₀ := x) (z₀ := g x) (l := 𝓝[>] 0) ?_ ?_ ?_ tendsto_const_nhds
          · refine hcv.congr' ?_
            filter_upwards [self_mem_nhdsWithin] with ε (hε : 0 < ε)
            simp only [hε, dite_true]; rfl
          · have : Tendsto (fun ε : ℝ => ε) (𝓝[>] 0) (𝓝 0) :=
              tendsto_nhdsWithin_of_tendsto_nhds tendsto_id
            refine this.congr' ?_
            filter_upwards [self_mem_nhdsWithin] with ε (hε : 0 < ε)
            simp only [hε, dite_true]; rfl
          · exact Filter.Eventually.of_forall fun _ => hgm.aestronglyMeasurable
          · have hc : ContinuousAt g x := hgc.continuousAt (isOpen_ne.mem_nhds hx)
            exact hc.tendsto.comp tendsto_snd
        have := (tendsto_const_nhds (x := g x)).sub hten
        rw [sub_self] at this
        have := this.pow 2
        rw [zero_pow two_ne_zero] at this
        refine this.congr' ?_
        filter_upwards [self_mem_nhdsWithin] with ε (hε : 0 < ε)
        simp only [hε, dite_true]
    rw [integral_zero] at key
    refine key.congr' ?_
    filter_upwards [self_mem_nhdsWithin] with ε (hε : 0 < ε)
    simp only [hε, dite_true]
  -- the comparison, then `ε → 0`, then `η → 0`
  have hlim : ∀ η : ℝ, 0 < η → variance g (gaussianE E) ≤ (1 + η) * (4 * I) := by
    intro η hη
    have hF : Tendsto (fun ε : ℝ => (1 + η) * (4 * (Real.exp ((R + 2 + ε) * ε) * I))
        + (1 + 1 / η) * (if hε : 0 < ε then
            ∫ x, (g x - moll (bump ε hε) g x) ^ 2 ∂(gaussianE E) else 0)) (𝓝[>] 0)
        (𝓝 ((1 + η) * (4 * (Real.exp ((R + 2 + 0) * 0) * I)) + (1 + 1 / η) * 0)) := by
      refine Tendsto.add (tendsto_const_nhds.mul ((Continuous.tendsto (by fun_prop) 0 |>.mono_left
        nhdsWithin_le_nhds).const_mul 4)) (hclose.const_mul _)
    simp only [mul_zero, Real.exp_zero, one_mul, add_zero] at hF
    refine ge_of_tendsto hF ?_
    filter_upwards [Ioc_mem_nhdsGT (zero_lt_one' ℝ)] with ε hε
    simp only [hε.1, dite_true]
    have h3 := variance_le_of_close (μ := gaussianE E) hgm (continuous_moll (bump ε hε.1) hg).measurable
      hM (abs_moll_le (bump ε hε.1) hg hM) hη
    have h4 := hvar ε hε.1 hε.2
    calc variance g (gaussianE E) ≤ _ := h3
      _ ≤ _ := add_le_add (mul_le_mul_of_nonneg_left h4 (by linarith)) le_rfl
  have hcont2 : Continuous fun η : ℝ => (1 + η) * (4 * I) := by fun_prop
  have htend2 := hcont2.tendsto 0
  simp only [add_zero, one_mul] at htend2
  refine ge_of_tendsto (tendsto_nhdsWithin_of_tendsto_nhds (s := Set.Ioi 0) htend2) ?_
  filter_upwards [self_mem_nhdsWithin] with η hη
  exact hlim η hη

end Mollify

section VarianceCompareLp

variable {X : Type*} [MeasurableSpace X] {μ : Measure X} [IsProbabilityMeasure μ]

/-- `Var f ≤ (1 + η) Var g + (1 + 1/η) E(f − g)²` for `f, g ∈ L²`. -/
theorem variance_le_of_close' {f g : X → ℝ} (hfmem : MemLp f 2 μ) (hgmem : MemLp g 2 μ) {η : ℝ}
    (hη : 0 < η) :
    variance f μ ≤ (1 + η) * variance g μ + (1 + 1 / η) * ∫ x, (f x - g x) ^ 2 ∂μ := by
  have hbound : ∀ x, (f x - ∫ y, g y ∂μ) ^ 2
      ≤ (1 + η) * (g x - ∫ y, g y ∂μ) ^ 2 + (1 + 1 / η) * (f x - g x) ^ 2 := by
    intro x
    have key : ∀ a b : ℝ, (a + b) ^ 2 ≤ (1 + η) * a ^ 2 + (1 + 1 / η) * b ^ 2 := by
      intro a b
      have h0 : 0 ≤ (η * a - b) ^ 2 / η := div_nonneg (sq_nonneg _) hη.le
      have e : (η * a - b) ^ 2 / η = η * a ^ 2 - 2 * a * b + b ^ 2 / η := by
        field_simp; ring
      rw [e] at h0
      have e2 : (1 + 1 / η) * b ^ 2 = b ^ 2 + b ^ 2 / η := by ring
      rw [e2]
      nlinarith [h0]
    have := key (g x - ∫ y, g y ∂μ) (f x - g x)
    rwa [show g x - ∫ y, g y ∂μ + (f x - g x) = f x - ∫ y, g y ∂μ by ring] at this
  -- `Var f ≤ E(f − a)²` for every `a`
  have hvar : variance f μ ≤ ∫ x, (f x - ∫ y, g y ∂μ) ^ 2 ∂μ := by
    set a := ∫ y, g y ∂μ
    have hfi : Integrable f μ := hfmem.integrable one_le_two
    have hf2 : Integrable (fun x => f x ^ 2) μ := hfmem.integrable_sq
    rw [variance_eq_sub hfmem]
    have h1 : ∫ x, (f x - a) ^ 2 ∂μ = ∫ x, f x ^ 2 ∂μ - 2 * a * ∫ x, f x ∂μ + a ^ 2 := by
      have : (fun x => (f x - a) ^ 2) = fun x => f x ^ 2 - 2 * a * f x + a ^ 2 := by
        funext x; ring
      have hB : Integrable (fun x => 2 * a * f x) μ := hfi.const_mul _
      have hA : Integrable (fun x => f x ^ 2 - 2 * a * f x) μ := hf2.sub hB
      rw [this, integral_add hA (integrable_const _), integral_sub hf2 hB, integral_const_mul,
        integral_const, measureReal_def, measure_univ, ENNReal.toReal_one, one_smul]
    rw [h1]
    have : ∫ x, (f ^ 2) x ∂μ = ∫ x, f x ^ 2 ∂μ := rfl
    rw [this]
    nlinarith [sq_nonneg ((∫ x, f x ∂μ) - a)]
  refine hvar.trans ?_
  have hgi : Integrable (fun x => (g x - ∫ y, g y ∂μ) ^ 2) μ :=
    (hgmem.sub (memLp_const _)).integrable_sq
  have hfgi : Integrable (fun x => (f x - g x) ^ 2) μ := (hfmem.sub hgmem).integrable_sq
  calc ∫ x, (f x - ∫ y, g y ∂μ) ^ 2 ∂μ
      ≤ ∫ x, ((1 + η) * (g x - ∫ y, g y ∂μ) ^ 2 + (1 + 1 / η) * (f x - g x) ^ 2) ∂μ :=
        integral_mono ((hfmem.sub (memLp_const _)).integrable_sq)
          ((hgi.const_mul _).add (hfgi.const_mul _)) hbound
    _ = (1 + η) * variance g μ + (1 + 1 / η) * ∫ x, (f x - g x) ^ 2 ∂μ := by
        rw [integral_add (hgi.const_mul _) (hfgi.const_mul _), integral_const_mul,
          integral_const_mul, variance_eq_integral hgmem.aestronglyMeasurable.aemeasurable]

end VarianceCompareLp

section GaussPoincareGeneral

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- A two-point bound with a slope continuous off the origin makes `f` continuous off the
origin. -/
theorem continuousOn_of_twoPoint {f G : E → ℝ} (hG : ContinuousOn G {x | x ≠ 0})
    (h : ∀ x y, x ≠ 0 → y ≠ 0 → |f x - f y| ≤ ‖x - y‖ * (G x + G y)) :
    ContinuousOn f {x | x ≠ 0} := by
  intro x hx
  have hx' : x ≠ 0 := hx
  have hGx : ContinuousAt G x := hG.continuousAt (isOpen_ne.mem_nhds hx')
  have hev : ∀ᶠ y in 𝓝 x, G y < G x + 1 := hGx (Iio_mem_nhds (by linarith))
  obtain ⟨r, hr, hball⟩ := Metric.eventually_nhds_iff.mp (hev.and (isOpen_ne.mem_nhds hx'))
  refine (continuousAt_of_locally_lipschitz (K := 2 * G x + 1) hr fun y hy => ?_).continuousWithinAt
  obtain ⟨h1, h2⟩ := hball hy
  rw [Real.dist_eq, dist_eq_norm]
  calc |f y - f x| ≤ ‖y - x‖ * (G y + G x) := h y x h2 hx'
    _ ≤ (2 * G x + 1) * ‖y - x‖ := by
        rw [mul_comm]; exact mul_le_mul_of_nonneg_right (by linarith) (norm_nonneg _)

theorem measurable_of_twoPoint {f G : E → ℝ} (hG : ContinuousOn G {x | x ≠ 0})
    (h : ∀ x y, x ≠ 0 → y ≠ 0 → |f x - f y| ≤ ‖x - y‖ * (G x + G y)) : Measurable f :=
  measurable_of_continuousOn_compl_singleton 0 (continuousOn_of_twoPoint hG h)

/-- Truncation at level `M`. -/
noncomputable def clampF (M : ℝ) (f : E → ℝ) (x : E) : ℝ := max (-M) (min (f x) M)

theorem abs_clampF_le {M : ℝ} (hM : 0 ≤ M) (f : E → ℝ) (x : E) : |clampF M f x| ≤ M := by
  unfold clampF
  rw [abs_le]
  constructor
  · exact le_max_left _ _
  · exact max_le (by linarith) (min_le_right _ _)

theorem abs_clampF_le_abs {M : ℝ} (hM : 0 ≤ M) (f : E → ℝ) (x : E) :
    |clampF M f x| ≤ |f x| := by
  unfold clampF
  rcases le_or_gt (f x) (-M) with h1 | h1
  · rw [min_eq_left (by linarith), max_eq_left (by linarith)]
    rw [abs_of_nonpos (by linarith), abs_of_nonpos (by linarith)]; linarith
  · rcases le_or_gt M (f x) with h2 | h2
    · rw [min_eq_right h2, max_eq_right (by linarith), abs_of_nonneg hM, abs_of_nonneg (by linarith)]
      exact h2
    · rw [min_eq_left h2.le, max_eq_right h1.le]

theorem clampF_eq_of_abs_le {M : ℝ} (f : E → ℝ) (x : E) (h : |f x| ≤ M) : clampF M f x = f x := by
  unfold clampF
  rw [abs_le] at h
  rw [min_eq_left h.2, max_eq_right h.1]

theorem abs_clampF_sub_le (M : ℝ) (f : E → ℝ) (x y : E) :
    |clampF M f x - clampF M f y| ≤ |f x - f y| := by
  have hlip : LipschitzWith 1 (fun s : ℝ => max (-M) (min s M)) :=
    (LipschitzWith.id.min_const M).const_max (-M)
  have := hlip.dist_le_mul (f x) (f y)
  simpa [Real.dist_eq, clampF] using this

theorem continuousOn_clampF {M : ℝ} {f : E → ℝ} {s : Set E} (hf : ContinuousOn f s) :
    ContinuousOn (clampF M f) s :=
  ContinuousOn.sup continuousOn_const (hf.inf continuousOn_const)

theorem measurable_clampF {M : ℝ} {f : E → ℝ} (hf : Measurable f) : Measurable (clampF M f) :=
  measurable_const.max (hf.min measurable_const)

/-- The truncated function cut off at infinity: `(1 − χ_R) · f_M`. -/
noncomputable def cutF (R M : ℝ) (f : E → ℝ) (x : E) : ℝ := (1 - cutoff R x) * clampF M f x

theorem abs_cutF_le {R M : ℝ} (hM : 0 ≤ M) (f : E → ℝ) (x : E) : |cutF R M f x| ≤ M := by
  unfold cutF
  rw [abs_mul, abs_of_nonneg (by linarith [cutoff_le_one R x])]
  calc (1 - cutoff R x) * |clampF M f x| ≤ 1 * M :=
        mul_le_mul (by linarith [cutoff_nonneg R x]) (abs_clampF_le hM f x) (abs_nonneg _) zero_le_one
    _ = M := one_mul _

theorem cutF_eq_zero {R M : ℝ} (hR : 0 < R) (f : E → ℝ) {x : E} (hx : Real.sqrt 2 * R < ‖x‖) :
    cutF R M f x = 0 := by
  unfold cutF
  rw [cutoff_eq_one hR (by
    have h2 : (Real.sqrt 2 * R) ^ 2 = 2 * R ^ 2 := by
      rw [mul_pow, Real.sq_sqrt zero_le_two]
    have := pow_le_pow_left₀ (mul_nonneg (Real.sqrt_nonneg 2) hR.le) hx.le 2
    linarith), sub_self, zero_mul]

theorem measurable_cutF {R M : ℝ} (hR : 0 < R) {f : E → ℝ} (hf : Measurable f) :
    Measurable (cutF R M f) :=
  (measurable_const.sub (contDiff_cutoff hR).continuous.measurable).mul (measurable_clampF hf)

theorem continuousOn_cutF {R M : ℝ} (hR : 0 < R) {f : E → ℝ} (hf : ContinuousOn f {x | x ≠ 0}) :
    ContinuousOn (cutF R M f) {x | x ≠ 0} :=
  (continuousOn_const.sub (contDiff_cutoff hR).continuous.continuousOn).mul (continuousOn_clampF hf)

/-- The two-point bound for the truncated, cut-off function: the slope grows by `Cc M/(2R)`. -/
theorem abs_cutF_sub_le {R M : ℝ} (hR : 0 < R) (hM : 0 ≤ M) {Cc : ℝ}
    (hCc : ∀ δ : ℝ, 0 < δ → ∀ z : E, ‖fderiv ℝ (cutoff δ) z‖ ≤ Cc / δ) {f G : E → ℝ}
    (h : ∀ x y, x ≠ 0 → y ≠ 0 → |f x - f y| ≤ ‖x - y‖ * (G x + G y)) {x y : E} (hx : x ≠ 0)
    (hy : y ≠ 0) :
    |cutF R M f x - cutF R M f y|
      ≤ ‖x - y‖ * ((G x + Cc * M / (2 * R)) + (G y + Cc * M / (2 * R))) := by
  have hCc0 : 0 ≤ Cc := by
    have := (norm_nonneg _).trans (hCc 1 one_pos 0); simpa using this
  have hlip : LipschitzWith (Real.toNNReal (Cc / R)) (cutoff (E := E) R) :=
    lipschitzWith_of_nnnorm_fderiv_le (fun z => (contDiff_cutoff hR).differentiable one_ne_zero z)
      fun z => by
        rw [← NNReal.coe_le_coe, coe_nnnorm, Real.coe_toNNReal _ (div_nonneg hCc0 hR.le)]
        exact hCc R hR z
  have hdist := hlip.dist_le_mul x y
  rw [Real.dist_eq, dist_eq_norm, Real.coe_toNNReal _ (div_nonneg hCc0 hR.le)] at hdist
  have hc0 := cutoff_nonneg R x
  have hc1 := cutoff_le_one R x
  have hb := abs_clampF_le hM f y
  have hab := (abs_clampF_sub_le M f x y).trans (h x y hx hy)
  have hsplit : cutF R M f x - cutF R M f y
      = (1 - cutoff R x) * (clampF M f x - clampF M f y)
        + (cutoff R y - cutoff R x) * clampF M f y := by
    unfold cutF; ring
  rw [hsplit]
  calc |(1 - cutoff R x) * (clampF M f x - clampF M f y)
        + (cutoff R y - cutoff R x) * clampF M f y|
      ≤ |(1 - cutoff R x) * (clampF M f x - clampF M f y)|
        + |(cutoff R y - cutoff R x) * clampF M f y| := abs_add_le _ _
    _ ≤ 1 * (‖x - y‖ * (G x + G y)) + (Cc / R * ‖x - y‖) * M := by
        rw [abs_mul, abs_mul, abs_of_nonneg (by linarith), abs_sub_comm (cutoff R y)]
        exact add_le_add (mul_le_mul (by linarith) hab (abs_nonneg _) zero_le_one)
          (mul_le_mul hdist hb (abs_nonneg _) (mul_nonneg (div_nonneg hCc0 hR.le) (norm_nonneg _)))
    _ = ‖x - y‖ * ((G x + Cc * M / (2 * R)) + (G y + Cc * M / (2 * R))) := by
        field_simp; ring

theorem integral_normSq_gaussianE :
    ∫ x, ‖x‖ ^ 2 ∂(gaussianE E) = (Module.finrank ℝ E : ℝ) := by
  rw [integral_eq_lintegral_of_nonneg_ae (Filter.Eventually.of_forall fun x => sq_nonneg _)
    (measurable_norm.pow_const 2).aestronglyMeasurable, lintegral_normSq_gaussianE,
    ENNReal.toReal_natCast]

theorem integrable_normSq_gaussianE : Integrable (fun x : E => ‖x‖ ^ 2) (gaussianE E) := by
  refine ⟨(measurable_norm.pow_const 2).aestronglyMeasurable, ?_⟩
  rw [hasFiniteIntegral_iff_ofReal (Filter.Eventually.of_forall fun x => sq_nonneg _),
    lintegral_normSq_gaussianE]
  exact ENNReal.natCast_lt_top _

theorem cutoff_sq_le {R : ℝ} (hR : 0 < R) (x : E) : cutoff R x ^ 2 ≤ ‖x‖ ^ 2 / R ^ 2 := by
  rcases le_or_gt ‖x‖ R with hx | hx
  · rw [cutoff_eq_zero hR hx, zero_pow two_ne_zero]; positivity
  · have h1 := cutoff_le_one R x
    have h0 := cutoff_nonneg R x
    calc cutoff R x ^ 2 ≤ 1 := by nlinarith
      _ ≤ ‖x‖ ^ 2 / R ^ 2 := by
          rw [le_div_iff₀ (by positivity)]; nlinarith

/-- **Stage 2: Poincaré for the truncation of a function with a two-point bound off the
origin.**  The cutoff at infinity costs `Cc M/(2R)` in the slope and `M² d/R²` in `L²`, both
vanishing as `R → ∞`. -/
theorem variance_clampF_le (hd : 2 ≤ Module.finrank ℝ E) {f G : E → ℝ}
    (hG : ContinuousOn G {x | x ≠ 0}) (hG0 : ∀ x, 0 ≤ G x)
    (hint : Integrable (fun x => G x ^ 2) (gaussianE E))
    (h : ∀ x y, x ≠ 0 → y ≠ 0 → |f x - f y| ≤ ‖x - y‖ * (G x + G y)) {M : ℝ} (hM : 0 ≤ M) :
    variance (clampF M f) (gaussianE E) ≤ 4 * ∫ x, G x ^ 2 ∂(gaussianE E) := by
  obtain ⟨Cc, hCc0, hCc⟩ := exists_bound_fderiv_cutoff (E := E)
  have hfc := continuousOn_of_twoPoint hG h
  have hfm := measurable_of_twoPoint hG h
  have hGm : Measurable G := measurable_of_continuousOn_compl_singleton 0 hG
  set I := ∫ x, G x ^ 2 ∂(gaussianE E) with hI
  have hI0 : 0 ≤ I := integral_nonneg fun x => sq_nonneg _
  have hGint : Integrable G (gaussianE E) :=
    ((memLp_two_iff_integrable_sq hGm.aestronglyMeasurable).mpr hint).integrable one_le_two
  set J := ∫ x, G x ∂(gaussianE E) with hJ
  set d : ℝ := (Module.finrank ℝ E : ℝ) with hd_def
  have hdint := integral_normSq_gaussianE (E := E)
  have hbound : ∀ η : ℝ, 0 < η → ∀ R : ℝ, 1 ≤ R →
      variance (clampF M f) (gaussianE E)
        ≤ (1 + η) * (4 * (I + 2 * (Cc * M / (2 * R)) * J + (Cc * M / (2 * R)) ^ 2))
          + (1 + 1 / η) * (M ^ 2 * d / R ^ 2) := by
    intro η hη R hR1
    have hR : 0 < R := by linarith
    set c := Cc * M / (2 * R) with hc
    have hc0 : 0 ≤ c := by positivity
    have hGc : Integrable (fun x => (G x + c) ^ 2) (gaussianE E) := by
      have : (fun x => (G x + c) ^ 2) = fun x => G x ^ 2 + (2 * c) * G x + c ^ 2 := by
        funext x; ring
      rw [this]
      exact (hint.add (hGint.const_mul _)).add (integrable_const _)
    have hGcI : ∫ x, (G x + c) ^ 2 ∂(gaussianE E) = I + 2 * c * J + c ^ 2 := by
      have : (fun x => (G x + c) ^ 2) = fun x => G x ^ 2 + (2 * c) * G x + c ^ 2 := by
        funext x; ring
      have hA : Integrable (fun x => G x ^ 2 + 2 * c * G x) (gaussianE E) :=
        hint.add (hGint.const_mul _)
      have hB : Integrable (fun x => 2 * c * G x) (gaussianE E) := hGint.const_mul _
      rw [this, integral_add hA (integrable_const _), integral_add hint hB, integral_const_mul,
        integral_const, measureReal_def, measure_univ, ENNReal.toReal_one, one_smul]
    have h1 := variance_le_of_twoPoint_compact hd (g := cutF R M f) (G := fun x => G x + c)
      (measurable_cutF hR hfm) (continuousOn_cutF hR hfc) (abs_cutF_le hM f)
      (R := Real.sqrt 2 * R) (mul_nonneg (Real.sqrt_nonneg 2) hR.le)
      (fun x hx => cutF_eq_zero hR f hx)
      (hGm.add_const c) (fun x => add_nonneg (hG0 x) hc0) hGc
      (fun x y hx hy => abs_cutF_sub_le hR hM hCc h hx hy)
    rw [hGcI] at h1
    have h2 : ∫ x, (clampF M f x - cutF R M f x) ^ 2 ∂(gaussianE E) ≤ M ^ 2 * d / R ^ 2 := by
      have hpt : ∀ x, (clampF M f x - cutF R M f x) ^ 2 ≤ M ^ 2 / R ^ 2 * ‖x‖ ^ 2 := by
        intro x
        have e : clampF M f x - cutF R M f x = cutoff R x * clampF M f x := by
          unfold cutF; ring
        rw [e, mul_pow]
        have hb : clampF M f x ^ 2 ≤ M ^ 2 := by
          rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) (abs_clampF_le hM f x) 2
        calc cutoff R x ^ 2 * clampF M f x ^ 2 ≤ (‖x‖ ^ 2 / R ^ 2) * M ^ 2 :=
              mul_le_mul (cutoff_sq_le hR x) hb (sq_nonneg _) (by positivity)
          _ = M ^ 2 / R ^ 2 * ‖x‖ ^ 2 := by ring
      calc ∫ x, (clampF M f x - cutF R M f x) ^ 2 ∂(gaussianE E)
          ≤ ∫ x, M ^ 2 / R ^ 2 * ‖x‖ ^ 2 ∂(gaussianE E) :=
            integral_mono_of_nonneg (Filter.Eventually.of_forall fun x => sq_nonneg _)
              (integrable_normSq_gaussianE.const_mul _) (Filter.Eventually.of_forall hpt)
        _ = M ^ 2 * d / R ^ 2 := by rw [integral_const_mul, hdint]; ring
    have h3 := variance_le_of_close (μ := gaussianE E) (measurable_clampF hfm)
      (measurable_cutF hR hfm) (abs_clampF_le hM f) (abs_cutF_le hM f) hη
    calc variance (clampF M f) (gaussianE E) ≤ _ := h3
      _ ≤ _ := add_le_add (mul_le_mul_of_nonneg_left h1 (by linarith))
          (mul_le_mul_of_nonneg_left h2 (by positivity))
  have hlim : ∀ η : ℝ, 0 < η → variance (clampF M f) (gaussianE E) ≤ (1 + η) * (4 * I) := by
    intro η hη
    have hc : Tendsto (fun R : ℝ => Cc * M / (2 * R)) atTop (𝓝 0) := by
      have : (fun R : ℝ => Cc * M / (2 * R)) = fun R => (Cc * M / 2) / R := by
        funext R; ring
      rw [this]
      exact tendsto_const_nhds.div_atTop tendsto_id
    have hr : Tendsto (fun R : ℝ => M ^ 2 * d / R ^ 2) atTop (𝓝 0) :=
      tendsto_const_nhds.div_atTop (tendsto_pow_atTop two_ne_zero)
    have hF : Tendsto (fun R : ℝ => (1 + η) * (4 * (I + 2 * (Cc * M / (2 * R)) * J
        + (Cc * M / (2 * R)) ^ 2)) + (1 + 1 / η) * (M ^ 2 * d / R ^ 2)) atTop
        (𝓝 ((1 + η) * (4 * (I + 2 * 0 * J + 0 ^ 2)) + (1 + 1 / η) * 0)) :=
      (tendsto_const_nhds.mul ((tendsto_const_nhds.add ((tendsto_const_nhds.mul hc).mul
        tendsto_const_nhds)).add (hc.pow 2) |>.const_mul 4)).add (hr.const_mul _)
    simp only [mul_zero, zero_mul, add_zero, zero_pow two_ne_zero] at hF
    refine ge_of_tendsto hF ?_
    filter_upwards [Filter.eventually_ge_atTop (1 : ℝ)] with R hR
    exact hbound η hη R hR
  have hcont2 : Continuous fun η : ℝ => (1 + η) * (4 * I) := by fun_prop
  have htend2 := hcont2.tendsto 0
  simp only [add_zero, one_mul] at htend2
  refine ge_of_tendsto (tendsto_nhdsWithin_of_tendsto_nhds (s := Set.Ioi 0) htend2) ?_
  filter_upwards [self_mem_nhdsWithin] with η hη
  exact hlim η hη

/-- **The general two-point Gaussian Poincaré inequality**: `Var f ≤ 4 ∫ G²` for every `f ∈ L²`
with `|f x − f y| ≤ ‖x − y‖ (G x + G y)` off the origin, `G` continuous off the origin and
square-integrable. -/
theorem variance_le_of_twoPoint (hd : 2 ≤ Module.finrank ℝ E) {f G : E → ℝ}
    (hG : ContinuousOn G {x | x ≠ 0})
    (h : ∀ x y, x ≠ 0 → y ≠ 0 → |f x - f y| ≤ ‖x - y‖ * (G x + G y))
    (hf : MemLp f 2 (gaussianE E)) (hint : Integrable (fun x => G x ^ 2) (gaussianE E)) :
    variance f (gaussianE E) ≤ 4 * ∫ x, G x ^ 2 ∂(gaussianE E) := by
  -- replace `G` by `|G|`
  have hG' : ContinuousOn (fun x => |G x|) {x | x ≠ 0} := hG.abs
  have h' : ∀ x y, x ≠ 0 → y ≠ 0 → |f x - f y| ≤ ‖x - y‖ * (|G x| + |G y|) := fun x y hx hy =>
    (h x y hx hy).trans (mul_le_mul_of_nonneg_left (add_le_add (le_abs_self _) (le_abs_self _))
      (norm_nonneg _))
  have hint' : Integrable (fun x => |G x| ^ 2) (gaussianE E) := by simpa [sq_abs] using hint
  have hI : ∫ x, |G x| ^ 2 ∂(gaussianE E) = ∫ x, G x ^ 2 ∂(gaussianE E) := by simp [sq_abs]
  rw [← hI]
  set I := ∫ x, |G x| ^ 2 ∂(gaussianE E) with hIdef
  have hfm := measurable_of_twoPoint hG h
  have hf2 : Integrable (fun x => f x ^ 2) (gaussianE E) := hf.integrable_sq
  -- `L²` closeness of the truncations
  have hclose : Tendsto (fun n : ℕ => ∫ x, (f x - clampF n f x) ^ 2 ∂(gaussianE E)) atTop
      (𝓝 0) := by
    have := tendsto_integral_of_dominated_convergence (μ := gaussianE E)
      (F := fun n : ℕ => fun x => (f x - clampF n f x) ^ 2) (f := fun _ => (0 : ℝ))
      (bound := fun x => 4 * f x ^ 2)
      (fun n => ((hfm.sub (measurable_clampF hfm)).pow_const 2).aestronglyMeasurable)
      (hf2.const_mul 4) (fun n => Filter.Eventually.of_forall fun x => by
        rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
        have hb := abs_clampF_le_abs (Nat.cast_nonneg n) f x
        have : |f x - clampF n f x| ≤ 2 * |f x| := by
          calc |f x - clampF n f x| ≤ |f x| + |clampF n f x| := abs_sub _ _
            _ ≤ 2 * |f x| := by linarith
        calc (f x - clampF n f x) ^ 2 = |f x - clampF n f x| ^ 2 := (sq_abs _).symm
          _ ≤ (2 * |f x|) ^ 2 := pow_le_pow_left₀ (abs_nonneg _) this 2
          _ = 4 * f x ^ 2 := by rw [mul_pow, sq_abs]; ring)
      (Filter.Eventually.of_forall fun x => by
        refine tendsto_const_nhds.congr' ?_
        filter_upwards [Filter.eventually_ge_atTop ⌈|f x|⌉₊] with n hn
        rw [clampF_eq_of_abs_le _ _ ((Nat.le_ceil _).trans (by exact_mod_cast hn)), sub_self,
          zero_pow two_ne_zero])
    simpa using this
  have hlim : ∀ η : ℝ, 0 < η → variance f (gaussianE E) ≤ (1 + η) * (4 * I) := by
    intro η hη
    have hF : Tendsto (fun n : ℕ => (1 + η) * (4 * I)
        + (1 + 1 / η) * ∫ x, (f x - clampF n f x) ^ 2 ∂(gaussianE E)) atTop
        (𝓝 ((1 + η) * (4 * I) + (1 + 1 / η) * 0)) :=
      tendsto_const_nhds.add (hclose.const_mul _)
    rw [mul_zero, add_zero] at hF
    refine ge_of_tendsto hF (Filter.Eventually.of_forall fun n => ?_)
    have hmem : MemLp (clampF n f) 2 (gaussianE E) :=
      MemLp.of_bound (measurable_clampF hfm).aestronglyMeasurable n
        (Filter.Eventually.of_forall fun x => by
          rw [Real.norm_eq_abs]; exact abs_clampF_le (Nat.cast_nonneg n) f x)
    have h3 := variance_le_of_close' hf hmem hη
    have h4 := variance_clampF_le hd hG' (fun x => abs_nonneg _) hint' h' (M := n)
      (Nat.cast_nonneg n)
    calc variance f (gaussianE E) ≤ _ := h3
      _ ≤ _ := add_le_add (mul_le_mul_of_nonneg_left h4 (by linarith)) le_rfl
  have hcont2 : Continuous fun η : ℝ => (1 + η) * (4 * I) := by fun_prop
  have htend2 := hcont2.tendsto 0
  simp only [add_zero, one_mul] at htend2
  refine ge_of_tendsto (tendsto_nhdsWithin_of_tendsto_nhds (s := Set.Ioi 0) htend2) ?_
  filter_upwards [self_mem_nhdsWithin] with η hη
  exact hlim η hη

/-- **The Gaussian Poincaré inequality in the two-point form of §26, proved.** -/
theorem gaussPoincare_gaussianE : GaussPoincare (gaussianE E) := by
  refine ⟨fun hd f G hG h hf hint => ?_⟩
  refine (variance_le_of_twoPoint hd hG h hf hint).trans (le_of_eq ?_)
  rw [← integral_const_mul]
  refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
  ring

/-- The sphere Poincaré inequality of §25, now a theorem: constant `4/(d − 2)`. -/
theorem spherePoincare_sphereUniformAmbient (hd : 3 ≤ Module.finrank ℝ E) :
    SpherePoincare (sphereUniformAmbient E) (4 / ((Module.finrank ℝ E : ℝ) - 2)) :=
  spherePoincare_of_gaussPoincare hd gaussPoincare_gaussianE

end GaussPoincareGeneral

section StreamGeneral

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] [FiniteDimensional ℝ E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G] [MeasurableMul G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {q : E → E} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G}
  [κ.IsMulRightInvariant] {x₀ : E}

/-- **The concentration remark for an arbitrary bounded codebook, with no hypothesis**: §26's
`meas_not_decayWithin_le_gauss` with `GaussPoincare` discharged by §39. -/
theorem meas_not_decayWithin_le_general (hd : 3 ≤ Module.finrank ℝ E)
    (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1) {C : Set E} (hq : IsNearest C q)
    (hC : Bornology.IsBounded C) {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε ≤ 1)
    (hρ : 0 < stageDistortion q (sphereUniformAmbient E)) (P : ℕ) :
    μ {ω | ¬ DecayWithin (rotScheme ι q (R ω)) (W ω)
        (Real.sqrt (stageDistortion q (sphereUniformAmbient E))) ε P}
      ≤ P * ENNReal.ofReal (16 / (((Module.finrank ℝ E : ℝ) - 2) * ε ^ 2 *
          stageDistortion q (sphereUniformAmbient E))) := by
  haveI : Nontrivial E := ⟨⟨x₀, 0, by rintro rfl; simp at hx₀⟩⟩
  exact meas_not_decayWithin_le_gauss hd gaussPoincare_gaussianE h hx₀ hq hC hε0 hε1 hρ P

end StreamGeneral
