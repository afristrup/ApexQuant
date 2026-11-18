/-! ## §32  The Gaussian Poincaré inequality with constant one

For a bounded `C¹` function `g` with bounded derivative, `Var_γ g ≤ E‖g'‖²`
(`variance_le_integral_norm_fderiv_sq`), with the sharp constant `1`.  The proof is the
covariance identity of the Ornstein–Uhlenbeck semigroup, done pathwise:
`−Var g = ∫₀^{π/2} E[g(X) g'(X_θ)[Y_θ]] dθ` for independent Gaussian `X, Y` and the rotation
`X_θ = cos θ·X + sin θ·Y` (Steps A and B), and for each `θ` an integration by parts in the
Gaussian variable turns `g(X) g'(X_θ)[Y_θ]` into `−sin θ · ⟨g'(X), g'(X_θ)⟩` (Step C).  The
integration by parts itself is proved by rotating an auxiliary Gaussian `W` against `Y_θ`
and differentiating along the rotation: no Gaussian integration by parts is taken from
Mathlib, only the invariance of `γ ⊗ γ` under rotations (§27).  Cauchy–Schwarz in the form
`ab ≤ (a² + b²)/2` and `∫₀^{π/2} sin θ dθ = 1` finish (Step D).  All integrability is by
polynomial bounds in the Gaussian variables. -/

section ProdHelpers

variable {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]

/-- Integrability on a product from a uniform bound on the slice `lintegral`s. -/
theorem integrable_prod_of_slice {μ : Measure α} {ν : Measure β} [SFinite μ] [IsFiniteMeasure ν]
    {F : α × β → ℝ} (hF : AEStronglyMeasurable F (μ.prod ν)) {B : ℝ≥0∞} (hB : B ≠ ⊤)
    (h : ∀ y, ∫⁻ x, ‖F (x, y)‖ₑ ∂μ ≤ B) : Integrable F (μ.prod ν) := by
  refine ⟨hF, ?_⟩
  rw [hasFiniteIntegral_iff_enorm, lintegral_prod_symm _ hF.enorm]
  calc ∫⁻ y, ∫⁻ x, ‖F (x, y)‖ₑ ∂μ ∂ν ≤ ∫⁻ _, B ∂ν := lintegral_mono fun y => h y
    _ = B * ν Set.univ := lintegral_const _
    _ < ⊤ := ENNReal.mul_lt_top hB.lt_top (measure_lt_top _ _)

theorem lintegral_fst_prod {μ : Measure α} {ν : Measure β} [IsProbabilityMeasure ν] [SFinite μ]
    {F : α → ℝ≥0∞} (hF : Measurable F) : ∫⁻ p, F p.1 ∂(μ.prod ν) = ∫⁻ x, F x ∂μ := by
  rw [← lintegral_map hF measurable_fst, Measure.map_fst_prod, measure_univ, one_smul]

theorem lintegral_snd_prod {μ : Measure α} {ν : Measure β} [IsProbabilityMeasure μ] [SFinite ν]
    {F : β → ℝ≥0∞} (hF : Measurable F) : ∫⁻ p, F p.2 ∂(μ.prod ν) = ∫⁻ y, F y ∂ν := by
  rw [← lintegral_map hF measurable_snd, Measure.map_snd_prod, measure_univ, one_smul]

theorem integral_fst_prod {μ : Measure α} {ν : Measure β} [IsProbabilityMeasure ν] [SFinite μ]
    {F : α → ℝ} (hF : AEStronglyMeasurable F μ) : ∫ p, F p.1 ∂(μ.prod ν) = ∫ x, F x ∂μ := by
  have := integral_map (μ := μ.prod ν) (φ := Prod.fst) measurable_fst.aemeasurable (f := F)
    (by rw [Measure.map_fst_prod, measure_univ, one_smul]; exact hF)
  rw [Measure.map_fst_prod, measure_univ, one_smul] at this
  exact this.symm

theorem integral_snd_prod {μ : Measure α} {ν : Measure β} [IsProbabilityMeasure μ] [SFinite ν]
    {F : β → ℝ} (hF : AEStronglyMeasurable F ν) : ∫ p, F p.2 ∂(μ.prod ν) = ∫ y, F y ∂ν := by
  have := integral_map (μ := μ.prod ν) (φ := Prod.snd) measurable_snd.aemeasurable (f := F)
    (by rw [Measure.map_snd_prod, measure_univ, one_smul]; exact hF)
  rw [Measure.map_snd_prod, measure_univ, one_smul] at this
  exact this.symm

end ProdHelpers

section PathMore

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

theorem norm_pathX_le (x y : E) (θ : ℝ) : ‖pathX x y θ‖ ≤ ‖x‖ + ‖y‖ := by
  unfold pathX
  calc ‖Real.cos θ • x + Real.sin θ • y‖ ≤ ‖Real.cos θ • x‖ + ‖Real.sin θ • y‖ := norm_add_le _ _
    _ ≤ ‖x‖ + ‖y‖ := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs]
        gcongr
        · exact mul_le_of_le_one_left (norm_nonneg _) (Real.abs_cos_le_one θ)
        · exact mul_le_of_le_one_left (norm_nonneg _) (Real.abs_sin_le_one θ)

theorem norm_pathY_le (x y : E) (θ : ℝ) : ‖pathY x y θ‖ ≤ ‖x‖ + ‖y‖ := by
  unfold pathY
  calc ‖-Real.sin θ • x + Real.cos θ • y‖ ≤ ‖-Real.sin θ • x‖ + ‖Real.cos θ • y‖ := norm_add_le _ _
    _ ≤ ‖x‖ + ‖y‖ := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, abs_neg]
        gcongr
        · exact mul_le_of_le_one_left (norm_nonneg _) (Real.abs_sin_le_one θ)
        · exact mul_le_of_le_one_left (norm_nonneg _) (Real.abs_cos_le_one θ)

theorem hasDerivAt_pathY (x y : E) (θ : ℝ) : HasDerivAt (pathY x y) (-pathX x y θ) θ := by
  have h : HasDerivAt (fun t => -(Real.sin t • x) + Real.cos t • y)
      (-(Real.cos θ • x) + -Real.sin θ • y) θ :=
    ((Real.hasDerivAt_sin θ).smul_const x).neg.add ((Real.hasDerivAt_cos θ).smul_const y)
  have e1 : (fun t => -(Real.sin t • x) + Real.cos t • y) = pathY x y := by
    funext t; rw [pathY, neg_smul]
  have e2 : -(Real.cos θ • x) + -Real.sin θ • y = -pathX x y θ := by
    rw [pathX, neg_smul, neg_add]
  rw [e1, e2] at h; exact h

theorem pathY_zero (x y : E) : pathY x y 0 = y := by simp [pathY]

theorem pathY_pi_div_two (x y : E) : pathY x y (Real.pi / 2) = -x := by simp [pathY]

theorem contDiff_pathY (x y : E) : ContDiff ℝ 1 (pathY x y) := by
  unfold pathY
  exact ((Real.contDiff_sin.neg).smul contDiff_const).add (Real.contDiff_cos.smul contDiff_const)

/-- Rotating back by `−θ` recovers the first point. -/
theorem pathX_rot_neg (x y : E) (θ : ℝ) : pathX (pathX x y θ) (pathY x y θ) (-θ) = x := by
  unfold pathX pathY
  simp only [Real.cos_neg, Real.sin_neg, smul_add, smul_smul, neg_smul, smul_neg, neg_neg]
  linear_combination (norm := module) (Real.cos_sq_add_sin_sq θ) • x

end PathMore

section C1Poincare

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E] [SecondCountableTopology E]

theorem measurable_fderiv_apply₂ {g : E → ℝ} {X : Type*} [MeasurableSpace X] {a b : X → E}
    (ha : Measurable a) (hb : Measurable b) : Measurable fun t => fderiv ℝ g (a t) (b t) :=
  Continuous.measurable2 (c := fun (L : E →L[ℝ] ℝ) (v : E) => L v)
    isBoundedBilinearMap_apply.continuous ((measurable_fderiv ℝ g).comp ha) hb

theorem measurable_pathX_apply {X : Type*} [MeasurableSpace X] {a b : X → E} (ha : Measurable a)
    (hb : Measurable b) (θ : ℝ) : Measurable fun t => pathX (a t) (b t) θ := by
  unfold pathX; fun_prop

theorem measurable_pathY_apply {X : Type*} [MeasurableSpace X] {a b : X → E} (ha : Measurable a)
    (hb : Measurable b) (θ : ℝ) : Measurable fun t => pathY (a t) (b t) θ := by
  unfold pathY; fun_prop

variable {g : E → ℝ} {M K : ℝ}

theorem abs_fderiv_path_le (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) (x y : E) (θ : ℝ) :
    |fderiv ℝ g (pathX x y θ) (pathY x y θ)| ≤ K * (‖x‖ + ‖y‖) := by
  rw [← Real.norm_eq_abs]
  calc ‖fderiv ℝ g (pathX x y θ) (pathY x y θ)‖
      ≤ ‖fderiv ℝ g (pathX x y θ)‖ * ‖pathY x y θ‖ := ContinuousLinearMap.le_opNorm _ _
    _ ≤ K * (‖x‖ + ‖y‖) := mul_le_mul (hK _) (norm_pathY_le x y θ) (norm_nonneg _)
        ((norm_nonneg _).trans (hK 0))

/-- **Step A: the path formula** `g(y) − g(x) = ∫₀^{π/2} g'(X_θ)[Y_θ] dθ`. -/
theorem sub_eq_integral_path (hg : Differentiable ℝ g) (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) (x y : E) :
    g y - g x = ∫ θ in (0 : ℝ)..(Real.pi / 2), fderiv ℝ g (pathX x y θ) (pathY x y θ) := by
  have hpi : (0 : ℝ) ≤ Real.pi / 2 := by positivity
  have hder : ∀ θ ∈ Set.uIcc (0 : ℝ) (Real.pi / 2), HasDerivAt (fun θ => g (pathX x y θ))
      (fderiv ℝ g (pathX x y θ) (pathY x y θ)) θ := fun θ _ =>
    (hg (pathX x y θ)).hasFDerivAt.comp_hasDerivAt θ (hasDerivAt_pathX x y θ)
  have hint : IntervalIntegrable (fun θ => fderiv ℝ g (pathX x y θ) (pathY x y θ)) volume 0
      (Real.pi / 2) := by
    rw [intervalIntegrable_iff_integrableOn_Ioc_of_le hpi]
    refine Measure.integrableOn_of_bounded (M := K * (‖x‖ + ‖y‖)) measure_Ioc_lt_top.ne
      (measurable_fderiv_apply₂ (contDiff_pathX x y).continuous.measurable
        (contDiff_pathY x y).continuous.measurable).aestronglyMeasurable
      (Filter.Eventually.of_forall fun θ => ?_)
    rw [Real.norm_eq_abs]; exact abs_fderiv_path_le hK x y θ
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hder hint, pathX_pi_div_two, pathX_zero]


theorem integrable_of_bound_gauss {X : Type*} [MeasurableSpace X]
    {μ : Measure X} [IsFiniteMeasure μ] {F : X → ℝ} (hF : AEStronglyMeasurable F μ) (C : ℝ)
    (h : ∀ x, |F x| ≤ C) : Integrable F μ :=
  Integrable.of_bound hF C (Filter.Eventually.of_forall fun x => by rw [Real.norm_eq_abs]; exact h x)

/-- `−Var g = E[g(X)(g(Y) − g(X))]` for independent copies. -/
theorem neg_variance_eq_integral_prod (hg : Differentiable ℝ g) (hM : ∀ z, |g z| ≤ M) :
    -variance g (gaussianE E)
      = ∫ p, g p.1 * (g p.2 - g p.1) ∂((gaussianE E).prod (gaussianE E)) := by
  have hgm : Measurable g := hg.continuous.measurable
  have hmem : MemLp g 2 (gaussianE E) :=
    MemLp.of_bound hgm.aestronglyMeasurable M (Filter.Eventually.of_forall fun z => by
      rw [Real.norm_eq_abs]; exact hM z)
  rw [variance_eq_half_integral_prod hmem]
  have hb : ∀ p : E × E, (g p.1 - g p.2) ^ 2
      = g p.1 * (g p.1 - g p.2) + g p.2 * (g p.2 - g p.1) := fun p => by ring
  simp_rw [hb]
  have h1 : Integrable (fun p : E × E => g p.1 * (g p.1 - g p.2)) ((gaussianE E).prod (gaussianE E)) :=
    integrable_of_bound_gauss (((hgm.comp measurable_fst).mul
      ((hgm.comp measurable_fst).sub (hgm.comp measurable_snd))).aestronglyMeasurable) (M * (2 * M))
      fun p => by
        rw [abs_mul]
        exact mul_le_mul (hM _) ((abs_sub _ _).trans (by linarith [hM p.1, hM p.2])) (abs_nonneg _)
          ((abs_nonneg _).trans (hM 0))
  have h2 : Integrable (fun p : E × E => g p.2 * (g p.2 - g p.1)) ((gaussianE E).prod (gaussianE E)) :=
    integrable_of_bound_gauss (((hgm.comp measurable_snd).mul
      ((hgm.comp measurable_snd).sub (hgm.comp measurable_fst))).aestronglyMeasurable) (M * (2 * M))
      fun p => by
        rw [abs_mul]
        exact mul_le_mul (hM _) ((abs_sub _ _).trans (by linarith [hM p.1, hM p.2])) (abs_nonneg _)
          ((abs_nonneg _).trans (hM 0))
  rw [integral_add h1 h2]
  have hswap : ∫ p, g p.2 * (g p.2 - g p.1) ∂((gaussianE E).prod (gaussianE E))
      = ∫ p, g p.1 * (g p.1 - g p.2) ∂((gaussianE E).prod (gaussianE E)) := by
    rw [← integral_prod_swap]; rfl
  rw [hswap]
  have : ∫ p, g p.1 * (g p.2 - g p.1) ∂((gaussianE E).prod (gaussianE E))
      = -∫ p, g p.1 * (g p.1 - g p.2) ∂((gaussianE E).prod (gaussianE E)) := by
    rw [← integral_neg]; congr 1; funext p; ring
  rw [this]; ring

theorem lintegral_norm_gaussianE_le :
    ∫⁻ x, ENNReal.ofReal ‖x‖ ∂(gaussianE E) ≤ (Module.finrank ℝ E : ℝ≥0∞) + 1 := by
  calc ∫⁻ x, ENNReal.ofReal ‖x‖ ∂(gaussianE E)
      ≤ ∫⁻ x, ENNReal.ofReal (‖x‖ ^ 2) + 1 ∂(gaussianE E) := by
        refine lintegral_mono fun x => ?_
        rw [← ENNReal.ofReal_one, ← ENNReal.ofReal_add (sq_nonneg _) zero_le_one]
        exact ENNReal.ofReal_le_ofReal (by nlinarith [sq_nonneg (‖x‖ - 1)])
    _ = (Module.finrank ℝ E : ℝ≥0∞) + 1 := by
        rw [lintegral_add_right _ measurable_const, lintegral_normSq_gaussianE, lintegral_const,
          measure_univ, mul_one]

/-- The path integrand is integrable on `γ ⊗ γ ⊗ dθ`. -/
theorem integrable_path_integrand (hg : Differentiable ℝ g) (hM : ∀ z, |g z| ≤ M)
    (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) :
    Integrable (fun q : (E × E) × ℝ => g q.1.1 * fderiv ℝ g (pathX q.1.1 q.1.2 q.2) (pathY q.1.1 q.1.2 q.2))
      (((gaussianE E).prod (gaussianE E)).prod (volume.restrict (Set.Ioc (0 : ℝ) (Real.pi / 2)))) := by
  have hgm : Measurable g := hg.continuous.measurable
  have hmeas : Measurable fun q : (E × E) × ℝ =>
      g q.1.1 * fderiv ℝ g (pathX q.1.1 q.1.2 q.2) (pathY q.1.1 q.1.2 q.2) :=
    (hgm.comp (measurable_fst.comp measurable_fst)).mul (measurable_fderiv_apply₂
      (by unfold pathX; fun_prop) (by unfold pathY; fun_prop))
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  have hK0 : 0 ≤ K := (norm_nonneg _).trans (hK 0)
  refine integrable_prod_of_slice hmeas.aestronglyMeasurable
    (B := ENNReal.ofReal (M * K) * (2 * ((Module.finrank ℝ E : ℝ≥0∞) + 1)))
    (ENNReal.mul_ne_top ENNReal.ofReal_ne_top (ENNReal.mul_ne_top (by simp)
      (ENNReal.add_ne_top.mpr ⟨ENNReal.natCast_ne_top _, ENNReal.one_ne_top⟩))) fun θ => ?_
  have hnorm := lintegral_norm_gaussianE_le (E := E)
  calc ∫⁻ p, ‖g p.1 * fderiv ℝ g (pathX p.1 p.2 θ) (pathY p.1 p.2 θ)‖ₑ ∂((gaussianE E).prod (gaussianE E))
      ≤ ∫⁻ p, ENNReal.ofReal (M * K) * (ENNReal.ofReal ‖p.1‖ + ENNReal.ofReal ‖p.2‖)
          ∂((gaussianE E).prod (gaussianE E)) := by
        refine lintegral_mono fun p => ?_
        rw [← ofReal_norm, ← ENNReal.ofReal_add (norm_nonneg _) (norm_nonneg _),
          ← ENNReal.ofReal_mul (by positivity)]
        refine ENNReal.ofReal_le_ofReal ?_
        rw [Real.norm_eq_abs, abs_mul]
        calc |g p.1| * |fderiv ℝ g (pathX p.1 p.2 θ) (pathY p.1 p.2 θ)|
            ≤ M * (K * (‖p.1‖ + ‖p.2‖)) :=
              mul_le_mul (hM _) (abs_fderiv_path_le hK _ _ _) (abs_nonneg _) hM0
          _ = M * K * (‖p.1‖ + ‖p.2‖) := by ring
    _ = ENNReal.ofReal (M * K) * (∫⁻ x, ENNReal.ofReal ‖x‖ ∂(gaussianE E) +
          ∫⁻ x, ENNReal.ofReal ‖x‖ ∂(gaussianE E)) := by
        have hm1 : Measurable fun p : E × E => ENNReal.ofReal ‖p.1‖ + ENNReal.ofReal ‖p.2‖ := by
          fun_prop
        have hm2 : Measurable fun p : E × E => ENNReal.ofReal ‖p.1‖ := by fun_prop
        rw [lintegral_const_mul _ hm1, lintegral_add_left hm2,
          lintegral_fst_prod measurable_norm.ennreal_ofReal,
          lintegral_snd_prod measurable_norm.ennreal_ofReal]
    _ ≤ ENNReal.ofReal (M * K) * (2 * ((Module.finrank ℝ E : ℝ≥0∞) + 1)) := by
        rw [two_mul]; exact mul_le_mul' le_rfl (add_le_add hnorm hnorm)

/-- **Step A': the variance as a path integral.** -/
theorem neg_variance_eq_integral_path (hg : Differentiable ℝ g) (hM : ∀ z, |g z| ≤ M)
    (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) :
    -variance g (gaussianE E) = ∫ θ in (0 : ℝ)..(Real.pi / 2), ∫ p,
      g p.1 * fderiv ℝ g (pathX p.1 p.2 θ) (pathY p.1 p.2 θ) ∂((gaussianE E).prod (gaussianE E)) := by
  have hpi : (0 : ℝ) ≤ Real.pi / 2 := by positivity
  rw [neg_variance_eq_integral_prod hg hM]
  have h1 : ∀ p : E × E, g p.1 * (g p.2 - g p.1)
      = ∫ θ in Set.Ioc (0 : ℝ) (Real.pi / 2), g p.1 * fderiv ℝ g (pathX p.1 p.2 θ) (pathY p.1 p.2 θ) := by
    intro p
    rw [sub_eq_integral_path hg hK, intervalIntegral.integral_of_le hpi, ← integral_const_mul]
  simp_rw [h1]
  rw [intervalIntegral.integral_of_le hpi]
  exact integral_integral_swap (integrable_path_integrand hg hM hK)

/-- **Step B: rotation.**  For each `θ`, `E[g(X) g'(X_θ)[Y_θ]] = E[g(X_{−θ}(U,V)) g'(U)[V]]`. -/
theorem integral_path_eq_rot (hg : Differentiable ℝ g) (θ : ℝ) :
    ∫ p, g p.1 * fderiv ℝ g (pathX p.1 p.2 θ) (pathY p.1 p.2 θ) ∂((gaussianE E).prod (gaussianE E))
      = ∫ q, g (pathX q.1 q.2 (-θ)) * fderiv ℝ g q.1 q.2 ∂((gaussianE E).prod (gaussianE E)) := by
  have hgm : Measurable g := hg.continuous.measurable
  have hF : AEStronglyMeasurable (fun q : E × E => g (pathX q.1 q.2 (-θ)) * fderiv ℝ g q.1 q.2)
      ((gaussianE E).prod (gaussianE E)) :=
    ((hgm.comp (by unfold pathX; fun_prop)).mul (measurable_fderiv_apply₂ measurable_fst measurable_snd))
      |>.aestronglyMeasurable
  have hmp := measurePreserving_rotProd (E := E) θ
  have := integral_map hmp.measurable.aemeasurable (f := fun q : E × E =>
    g (pathX q.1 q.2 (-θ)) * fderiv ℝ g q.1 q.2) (by rw [hmp.map_eq]; exact hF)
  rw [hmp.map_eq] at this
  rw [this]
  refine integral_congr_ae (Filter.Eventually.of_forall fun p => ?_)
  simp only [pathX_rot_neg]


/-! ### Step C: integration by parts by rotating an auxiliary Gaussian -/

theorem measurePreserving_neg_gaussianE :
    MeasurePreserving (fun x : E => -x) (gaussianE E) (gaussianE E) := by
  have h := measurePreserving_isometry_gaussianE (E := E) (LinearIsometryEquiv.neg ℝ)
  convert h using 1
  ext x; simp

/-- The rotation of the last two coordinates of a triple. -/
noncomputable def rot3 (φ : ℝ) (t : E × (E × E)) : E × (E × E) :=
  (t.1, (pathX t.2.1 t.2.2 φ, pathY t.2.1 t.2.2 φ))

theorem measurable_rot3 (φ : ℝ) : Measurable (rot3 (E := E) φ) := by
  unfold rot3 pathX pathY; fun_prop

theorem measurable_rot3_uncurry :
    Measurable fun q : (E × (E × E)) × ℝ => rot3 q.2 q.1 := by
  unfold rot3 pathX pathY; fun_prop

theorem measurePreserving_rot3 (φ : ℝ) :
    MeasurePreserving (rot3 (E := E) φ) ((gaussianE E).prod ((gaussianE E).prod (gaussianE E)))
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) :=
  (MeasurePreserving.id _).prod (measurePreserving_rotProd φ)

/-- The derivative in `φ` of `g(cos θ·u − sin θ·X_φ) · g'(u)[Y_φ]` at `φ = 0`, as a function of
the triple `(u, v, w)`. -/
noncomputable def ibpInt (g : E → ℝ) (θ : ℝ) (t : E × (E × E)) : ℝ :=
  fderiv ℝ g (pathX t.1 t.2.1 (-θ)) (Real.sin (-θ) • t.2.2) * fderiv ℝ g t.1 t.2.2
    + g (pathX t.1 t.2.1 (-θ)) * fderiv ℝ g t.1 (-t.2.1)

theorem measurable_ibpInt (hg : Differentiable ℝ g) (θ : ℝ) : Measurable (ibpInt g θ) := by
  have hgm : Measurable g := hg.continuous.measurable
  unfold ibpInt
  refine ((measurable_fderiv_apply₂ (by unfold pathX; fun_prop) (by fun_prop)).mul
    (measurable_fderiv_apply₂ measurable_fst (measurable_snd.comp measurable_snd))).add
    ((hgm.comp (by unfold pathX; fun_prop)).mul
      (measurable_fderiv_apply₂ measurable_fst (measurable_fst.comp measurable_snd).neg))

theorem abs_ibpInt_le (hM : ∀ z, |g z| ≤ M) (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) (θ : ℝ)
    (t : E × (E × E)) : |ibpInt g θ t| ≤ K ^ 2 * ‖t.2.2‖ ^ 2 + M * K * ‖t.2.1‖ := by
  have hK0 : 0 ≤ K := (norm_nonneg _).trans (hK 0)
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  unfold ibpInt
  refine (abs_add_le _ _).trans (add_le_add ?_ ?_)
  · rw [abs_mul, ← Real.norm_eq_abs, ← Real.norm_eq_abs]
    calc ‖fderiv ℝ g (pathX t.1 t.2.1 (-θ)) (Real.sin (-θ) • t.2.2)‖ * ‖fderiv ℝ g t.1 t.2.2‖
        ≤ (K * ‖Real.sin (-θ) • t.2.2‖) * (K * ‖t.2.2‖) :=
          mul_le_mul ((ContinuousLinearMap.le_opNorm _ _).trans
              (mul_le_mul_of_nonneg_right (hK _) (norm_nonneg _)))
            ((ContinuousLinearMap.le_opNorm _ _).trans
              (mul_le_mul_of_nonneg_right (hK _) (norm_nonneg _))) (norm_nonneg _) (by positivity)
      _ ≤ (K * ‖t.2.2‖) * (K * ‖t.2.2‖) := by
          gcongr
          rw [norm_smul, Real.norm_eq_abs]
          exact mul_le_of_le_one_left (norm_nonneg _) (Real.abs_sin_le_one _)
      _ = K ^ 2 * ‖t.2.2‖ ^ 2 := by ring
  · rw [abs_mul, ← Real.norm_eq_abs (fderiv ℝ g t.1 (-t.2.1))]
    calc |g (pathX t.1 t.2.1 (-θ))| * ‖fderiv ℝ g t.1 (-t.2.1)‖
        ≤ M * (K * ‖-t.2.1‖) := mul_le_mul (hM _) ((ContinuousLinearMap.le_opNorm _ _).trans
            (mul_le_mul_of_nonneg_right (hK _) (norm_nonneg _))) (norm_nonneg _) hM0
      _ = M * K * ‖t.2.1‖ := by rw [norm_neg]; ring

theorem hasDerivAt_pathX_pathX (u v w : E) (θ φ : ℝ) :
    HasDerivAt (fun φ => pathX u (pathX v w φ) (-θ)) (Real.sin (-θ) • pathY v w φ) φ := by
  have h : HasDerivAt (fun φ => Real.cos (-θ) • u + Real.sin (-θ) • pathX v w φ)
      (0 + Real.sin (-θ) • pathY v w φ) φ :=
    (hasDerivAt_const φ _).add ((hasDerivAt_pathX v w φ).const_smul (Real.sin (-θ)))
  rw [zero_add] at h
  exact h

theorem hasDerivAt_ibp_path (hg : Differentiable ℝ g) (t : E × (E × E)) (θ φ : ℝ) :
    HasDerivAt (fun φ => g (pathX t.1 (pathX t.2.1 t.2.2 φ) (-θ)) *
      fderiv ℝ g t.1 (pathY t.2.1 t.2.2 φ)) (ibpInt g θ (rot3 φ t)) φ := by
  have h1 : HasDerivAt (fun φ => g (pathX t.1 (pathX t.2.1 t.2.2 φ) (-θ)))
      (fderiv ℝ g (pathX t.1 (pathX t.2.1 t.2.2 φ) (-θ)) (Real.sin (-θ) • pathY t.2.1 t.2.2 φ)) φ :=
    (hg _).hasFDerivAt.comp_hasDerivAt φ (hasDerivAt_pathX_pathX t.1 t.2.1 t.2.2 θ φ)
  have h2 : HasDerivAt (fun φ => fderiv ℝ g t.1 (pathY t.2.1 t.2.2 φ))
      (fderiv ℝ g t.1 (-pathX t.2.1 t.2.2 φ)) φ :=
    (fderiv ℝ g t.1).hasFDerivAt.comp_hasDerivAt φ (hasDerivAt_pathY t.2.1 t.2.2 φ)
  exact h1.mul h2

theorem abs_ibpInt_rot3_le (hM : ∀ z, |g z| ≤ M) (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) (θ φ : ℝ)
    (t : E × (E × E)) :
    |ibpInt g θ (rot3 φ t)| ≤ K ^ 2 * (‖t.2.1‖ + ‖t.2.2‖) ^ 2 + M * K * (‖t.2.1‖ + ‖t.2.2‖) := by
  have hK0 : 0 ≤ K := (norm_nonneg _).trans (hK 0)
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  have h1 : ‖(rot3 φ t).2.2‖ ≤ ‖t.2.1‖ + ‖t.2.2‖ := norm_pathY_le _ _ _
  have h2 : ‖(rot3 φ t).2.1‖ ≤ ‖t.2.1‖ + ‖t.2.2‖ := norm_pathX_le _ _ _
  refine (abs_ibpInt_le hM hK θ _).trans (add_le_add ?_ ?_)
  · exact mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (norm_nonneg _) h1 2) (by positivity)
  · exact mul_le_mul_of_nonneg_left h2 (by positivity)

/-- **The pathwise identity** behind the integration by parts. -/
theorem ibp_path_sub_eq_integral (hg : Differentiable ℝ g) (hM : ∀ z, |g z| ≤ M)
    (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) (θ : ℝ) (t : E × (E × E)) :
    g (pathX t.1 t.2.2 (-θ)) * fderiv ℝ g t.1 (-t.2.1)
        - g (pathX t.1 t.2.1 (-θ)) * fderiv ℝ g t.1 t.2.2
      = ∫ φ in (0 : ℝ)..(Real.pi / 2), ibpInt g θ (rot3 φ t) := by
  have hpi : (0 : ℝ) ≤ Real.pi / 2 := by positivity
  have hint : IntervalIntegrable (fun φ => ibpInt g θ (rot3 φ t)) volume 0 (Real.pi / 2) := by
    rw [intervalIntegrable_iff_integrableOn_Ioc_of_le hpi]
    refine Measure.integrableOn_of_bounded
      (M := K ^ 2 * (‖t.2.1‖ + ‖t.2.2‖) ^ 2 + M * K * (‖t.2.1‖ + ‖t.2.2‖)) measure_Ioc_lt_top.ne
      ((measurable_ibpInt hg θ).comp (by unfold rot3 pathX pathY; fun_prop)).aestronglyMeasurable
      (Filter.Eventually.of_forall fun φ => ?_)
    rw [Real.norm_eq_abs]; exact abs_ibpInt_rot3_le hM hK θ φ t
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun φ _ => hasDerivAt_ibp_path hg t θ φ) hint,
    pathX_pi_div_two, pathY_pi_div_two, pathX_zero, pathY_zero]

theorem integrable_of_enorm_le {X : Type*} [MeasurableSpace X] {μ : Measure X} {F : X → ℝ}
    {G : X → ℝ≥0∞} (hF : AEStronglyMeasurable F μ) (hG : ∫⁻ x, G x ∂μ < ⊤)
    (h : ∀ x, ‖F x‖ₑ ≤ G x) : Integrable F μ :=
  ⟨hF, by rw [hasFiniteIntegral_iff_enorm]; exact (lintegral_mono h).trans_lt hG⟩

/-- `∫⁻ ‖w‖² ` and `∫⁻ ‖v‖` over the triple. -/
theorem lintegral_triple_normSq_snd_snd :
    ∫⁻ t : E × (E × E), ENNReal.ofReal (‖t.2.2‖ ^ 2)
        ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E)))
      = (Module.finrank ℝ E : ℝ≥0∞) := by
  rw [lintegral_snd_prod (F := fun p : E × E => ENNReal.ofReal (‖p.2‖ ^ 2)) (by fun_prop),
    lintegral_snd_prod (F := fun x : E => ENNReal.ofReal (‖x‖ ^ 2)) (by fun_prop),
    lintegral_normSq_gaussianE]

theorem lintegral_triple_norm_snd_fst_le :
    ∫⁻ t : E × (E × E), ENNReal.ofReal ‖t.2.1‖
        ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E)))
      ≤ (Module.finrank ℝ E : ℝ≥0∞) + 1 := by
  rw [lintegral_snd_prod (F := fun p : E × E => ENNReal.ofReal ‖p.1‖) (by fun_prop),
    lintegral_fst_prod (F := fun x : E => ENNReal.ofReal ‖x‖) (by fun_prop)]
  exact lintegral_norm_gaussianE_le

theorem lintegral_triple_norm_snd_snd_le :
    ∫⁻ t : E × (E × E), ENNReal.ofReal ‖t.2.2‖
        ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E)))
      ≤ (Module.finrank ℝ E : ℝ≥0∞) + 1 := by
  rw [lintegral_snd_prod (F := fun p : E × E => ENNReal.ofReal ‖p.2‖) (by fun_prop),
    lintegral_snd_prod (F := fun x : E => ENNReal.ofReal ‖x‖) (by fun_prop)]
  exact lintegral_norm_gaussianE_le

/-- The bound `K²‖w‖² + MK‖v‖ + MK‖w‖` has finite integral over the triple. -/
theorem lintegral_ibp_bound_lt_top (hM0 : 0 ≤ M) (hK0 : 0 ≤ K) :
    ∫⁻ t : E × (E × E), ENNReal.ofReal (K ^ 2 * ‖t.2.2‖ ^ 2 + M * K * ‖t.2.1‖ + M * K * ‖t.2.2‖)
        ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) < ⊤ := by
  have hd1 : (Module.finrank ℝ E : ℝ≥0∞) + 1 < ⊤ :=
    ENNReal.add_lt_top.mpr ⟨ENNReal.natCast_lt_top _, ENNReal.one_lt_top⟩
  calc ∫⁻ t : E × (E × E), ENNReal.ofReal (K ^ 2 * ‖t.2.2‖ ^ 2 + M * K * ‖t.2.1‖ + M * K * ‖t.2.2‖)
        ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E)))
      = ∫⁻ t : E × (E × E), ENNReal.ofReal (K ^ 2) * ENNReal.ofReal (‖t.2.2‖ ^ 2)
          + ENNReal.ofReal M * ENNReal.ofReal K * ENNReal.ofReal ‖t.2.1‖
          + ENNReal.ofReal M * ENNReal.ofReal K * ENNReal.ofReal ‖t.2.2‖
          ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
        refine lintegral_congr fun t => ?_
        rw [ENNReal.ofReal_add (by positivity) (by positivity),
          ENNReal.ofReal_add (by positivity) (by positivity)]
        simp only [ENNReal.ofReal_mul (sq_nonneg K), ENNReal.ofReal_mul (mul_nonneg hM0 hK0),
          ENNReal.ofReal_mul hM0]
    _ = ENNReal.ofReal (K ^ 2) * (Module.finrank ℝ E : ℝ≥0∞)
          + ENNReal.ofReal M * ENNReal.ofReal K * ∫⁻ t : E × (E × E), ENNReal.ofReal ‖t.2.1‖
            ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E)))
          + ENNReal.ofReal M * ENNReal.ofReal K * ∫⁻ t : E × (E × E), ENNReal.ofReal ‖t.2.2‖
            ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
        rw [lintegral_add_left (by fun_prop), lintegral_add_left (by fun_prop),
          lintegral_const_mul _ (by fun_prop), lintegral_const_mul _ (by fun_prop),
          lintegral_const_mul _ (by fun_prop), lintegral_triple_normSq_snd_snd]
    _ < ⊤ := by
        have hMK : ENNReal.ofReal M * ENNReal.ofReal K < ⊤ :=
          ENNReal.mul_lt_top ENNReal.ofReal_lt_top ENNReal.ofReal_lt_top
        refine ENNReal.add_lt_top.mpr ⟨ENNReal.add_lt_top.mpr
          ⟨ENNReal.mul_lt_top ENNReal.ofReal_lt_top (by simp),
           ENNReal.mul_lt_top hMK (lintegral_triple_norm_snd_fst_le.trans_lt hd1)⟩,
          ENNReal.mul_lt_top hMK (lintegral_triple_norm_snd_snd_le.trans_lt hd1)⟩

theorem integrable_ibpInt (hg : Differentiable ℝ g) (hM : ∀ z, |g z| ≤ M)
    (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) (θ : ℝ) :
    Integrable (ibpInt g θ) ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) :=
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  have hK0 : 0 ≤ K := (norm_nonneg _).trans (hK 0)
  integrable_of_enorm_le (measurable_ibpInt hg θ).aestronglyMeasurable
    (lintegral_ibp_bound_lt_top hM0 hK0)
    fun t => by
      rw [Real.enorm_eq_ofReal_abs]
      exact ENNReal.ofReal_le_ofReal ((abs_ibpInt_le hM hK θ t).trans (by
        nlinarith [mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.2))]))

theorem integral_ibpInt_rot3 (hg : Differentiable ℝ g) (θ φ : ℝ) :
    ∫ t, ibpInt g θ (rot3 φ t) ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E)))
      = ∫ t, ibpInt g θ t ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
  have hmp := measurePreserving_rot3 (E := E) φ
  have := integral_map hmp.measurable.aemeasurable (f := ibpInt g θ)
    (by rw [hmp.map_eq]; exact (measurable_ibpInt hg θ).aestronglyMeasurable)
  rw [hmp.map_eq] at this
  exact this.symm

theorem integrable_ibpInt_rot3_prod (hg : Differentiable ℝ g) (hM : ∀ z, |g z| ≤ M)
    (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) (θ : ℝ) :
    Integrable (fun q : (E × (E × E)) × ℝ => ibpInt g θ (rot3 q.2 q.1))
      (((gaussianE E).prod ((gaussianE E).prod (gaussianE E))).prod
        (volume.restrict (Set.Ioc (0 : ℝ) (Real.pi / 2)))) := by
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  have hK0 : 0 ≤ K := (norm_nonneg _).trans (hK 0)
  have hFm : Measurable fun q : (E × (E × E)) × ℝ => ibpInt g θ (rot3 q.2 q.1) :=
    (measurable_ibpInt hg θ).comp measurable_rot3_uncurry
  refine integrable_prod_of_slice hFm.aestronglyMeasurable (B := ∫⁻ t : E × (E × E),
      ENNReal.ofReal (K ^ 2 * ‖t.2.2‖ ^ 2 + M * K * ‖t.2.1‖ + M * K * ‖t.2.2‖)
        ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E))))
    (lintegral_ibp_bound_lt_top hM0 hK0).ne fun φ => ?_
  have hmp := measurePreserving_rot3 (E := E) φ
  rw [hmp.lintegral_comp (f := fun t => ‖ibpInt g θ t‖ₑ) (measurable_ibpInt hg θ).enorm]
  refine lintegral_mono fun t => ?_
  rw [Real.enorm_eq_ofReal_abs]
  exact ENNReal.ofReal_le_ofReal ((abs_ibpInt_le hM hK θ t).trans (by
    nlinarith [mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.2))]))

/-- The two boundary terms vanish by the symmetry `w ↦ −w`, resp. `v ↦ −v`. -/
theorem integral_ibp_boundary_zero₀ (hg : Differentiable ℝ g) (θ : ℝ) :
    ∫ t, g (pathX t.1 t.2.1 (-θ)) * fderiv ℝ g t.1 t.2.2
      ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) = 0 := by
  have hgm : Measurable g := hg.continuous.measurable
  have hmp : MeasurePreserving (fun t : E × (E × E) => (t.1, (t.2.1, -t.2.2)))
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E)))
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) :=
    (MeasurePreserving.id _).prod ((MeasurePreserving.id _).prod measurePreserving_neg_gaussianE)
  have hFm : Measurable fun t : E × (E × E) => g (pathX t.1 t.2.1 (-θ)) * fderiv ℝ g t.1 t.2.2 :=
    (hgm.comp (measurable_pathX_apply measurable_fst (measurable_fst.comp measurable_snd) _)).mul
      (measurable_fderiv_apply₂ measurable_fst (measurable_snd.comp measurable_snd))
  have h := integral_map hmp.measurable.aemeasurable
    (f := fun t : E × (E × E) => g (pathX t.1 t.2.1 (-θ)) * fderiv ℝ g t.1 t.2.2)
    (by rw [hmp.map_eq]; exact hFm.aestronglyMeasurable)
  rw [hmp.map_eq] at h
  simp only [map_neg, mul_neg, integral_neg] at h
  linarith

theorem integral_ibp_boundary_zero₁ (hg : Differentiable ℝ g) (θ : ℝ) :
    ∫ t, g (pathX t.1 t.2.2 (-θ)) * fderiv ℝ g t.1 (-t.2.1)
      ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) = 0 := by
  have hgm : Measurable g := hg.continuous.measurable
  have hmp : MeasurePreserving (fun t : E × (E × E) => (t.1, (-t.2.1, t.2.2)))
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E)))
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) :=
    (MeasurePreserving.id _).prod (measurePreserving_neg_gaussianE.prod (MeasurePreserving.id _))
  have hFm : Measurable fun t : E × (E × E) => g (pathX t.1 t.2.2 (-θ)) * fderiv ℝ g t.1 (-t.2.1) :=
    (hgm.comp (measurable_pathX_apply measurable_fst (measurable_snd.comp measurable_snd) _)).mul
      (measurable_fderiv_apply₂ measurable_fst (measurable_fst.comp measurable_snd).neg)
  have h := integral_map hmp.measurable.aemeasurable
    (f := fun t : E × (E × E) => g (pathX t.1 t.2.2 (-θ)) * fderiv ℝ g t.1 (-t.2.1))
    (by rw [hmp.map_eq]; exact hFm.aestronglyMeasurable)
  rw [hmp.map_eq] at h
  simp only [neg_neg, map_neg, mul_neg, integral_neg] at h ⊢
  linarith

theorem abs_mul_fderiv_le (hM : ∀ z, |g z| ≤ M) (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) (a z v : E) :
    |g a * fderiv ℝ g z v| ≤ M * K * ‖v‖ := by
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  rw [abs_mul, ← Real.norm_eq_abs (fderiv ℝ g z v)]
  calc |g a| * ‖fderiv ℝ g z v‖ ≤ M * (K * ‖v‖) :=
        mul_le_mul (hM _) ((ContinuousLinearMap.le_opNorm _ _).trans
          (mul_le_mul_of_nonneg_right (hK _) (norm_nonneg _))) (norm_nonneg _) hM0
    _ = M * K * ‖v‖ := by ring

theorem abs_fderiv_mul_fderiv_le (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) (z z' v : E) :
    |fderiv ℝ g z v * fderiv ℝ g z' v| ≤ K ^ 2 * ‖v‖ ^ 2 := by
  have hK0 : 0 ≤ K := (norm_nonneg _).trans (hK 0)
  rw [abs_mul, ← Real.norm_eq_abs, ← Real.norm_eq_abs]
  calc ‖fderiv ℝ g z v‖ * ‖fderiv ℝ g z' v‖ ≤ (K * ‖v‖) * (K * ‖v‖) :=
        mul_le_mul ((ContinuousLinearMap.le_opNorm _ _).trans
            (mul_le_mul_of_nonneg_right (hK _) (norm_nonneg _)))
          ((ContinuousLinearMap.le_opNorm _ _).trans
            (mul_le_mul_of_nonneg_right (hK _) (norm_nonneg _))) (norm_nonneg _) (by positivity)
    _ = K ^ 2 * ‖v‖ ^ 2 := by ring

/-- Integrability over the triple of anything bounded by `K²‖w‖² + MK‖v‖ + MK‖w‖`. -/
theorem integrable_triple_of_bound (hM0 : 0 ≤ M) (hK0 : 0 ≤ K) {F : E × (E × E) → ℝ}
    (hF : Measurable F)
    (h : ∀ t, |F t| ≤ K ^ 2 * ‖t.2.2‖ ^ 2 + M * K * ‖t.2.1‖ + M * K * ‖t.2.2‖) :
    Integrable F ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) :=
  integrable_of_enorm_le hF.aestronglyMeasurable (lintegral_ibp_bound_lt_top hM0 hK0) fun t => by
    rw [Real.enorm_eq_ofReal_abs]; exact ENNReal.ofReal_le_ofReal (h t)

/-- **The integrated derivative vanishes**: `∫ ibpInt = 0`. -/
theorem integral_ibpInt_eq_zero (hg : Differentiable ℝ g) (hM : ∀ z, |g z| ≤ M)
    (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) (θ : ℝ) :
    ∫ t, ibpInt g θ t ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) = 0 := by
  have hpi : (0 : ℝ) ≤ Real.pi / 2 := by positivity
  have hpi' : (0 : ℝ) < Real.pi / 2 := by positivity
  have hgm : Measurable g := hg.continuous.measurable
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  have hK0 : 0 ≤ K := (norm_nonneg _).trans (hK 0)
  have hpath := ibp_path_sub_eq_integral hg hM hK θ
  have hsum : ∫ t, (g (pathX t.1 t.2.2 (-θ)) * fderiv ℝ g t.1 (-t.2.1)
        - g (pathX t.1 t.2.1 (-θ)) * fderiv ℝ g t.1 t.2.2)
        ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) = 0 := by
    have h1 : Integrable (fun t : E × (E × E) => g (pathX t.1 t.2.2 (-θ)) * fderiv ℝ g t.1 (-t.2.1))
        ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
      refine integrable_triple_of_bound hM0 hK0
        ((hgm.comp (measurable_pathX_apply measurable_fst (measurable_snd.comp measurable_snd) _)).mul
          (measurable_fderiv_apply₂ measurable_fst (measurable_fst.comp measurable_snd).neg))
        fun t => ?_
      have := abs_mul_fderiv_le hM hK (pathX t.1 t.2.2 (-θ)) t.1 (-t.2.1)
      rw [norm_neg] at this
      nlinarith [sq_nonneg (K * ‖t.2.2‖), mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.2))]
    have h2 : Integrable (fun t : E × (E × E) => g (pathX t.1 t.2.1 (-θ)) * fderiv ℝ g t.1 t.2.2)
        ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
      refine integrable_triple_of_bound hM0 hK0
        ((hgm.comp (measurable_pathX_apply measurable_fst (measurable_fst.comp measurable_snd) _)).mul
          (measurable_fderiv_apply₂ measurable_fst (measurable_snd.comp measurable_snd)))
        fun t => ?_
      have := abs_mul_fderiv_le hM hK (pathX t.1 t.2.1 (-θ)) t.1 t.2.2
      nlinarith [sq_nonneg (K * ‖t.2.2‖), mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.1))]
    rw [integral_sub h1 h2, integral_ibp_boundary_zero₀ hg θ, integral_ibp_boundary_zero₁ hg θ,
      sub_zero]
  simp_rw [hpath, intervalIntegral.integral_of_le hpi] at hsum
  rw [integral_integral_swap (integrable_ibpInt_rot3_prod hg hM hK θ)] at hsum
  simp_rw [integral_ibpInt_rot3 hg θ] at hsum
  simp only [setIntegral_const, smul_eq_mul, measureReal_def, Real.volume_Ioc, sub_zero,
    ENNReal.toReal_ofReal hpi] at hsum
  exact (mul_eq_zero.mp hsum).resolve_left hpi'.ne'

/-- **Step C: integration by parts.** -/
theorem integral_rot_ibp (hg : Differentiable ℝ g) (hM : ∀ z, |g z| ≤ M)
    (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) (θ : ℝ) :
    ∫ q, g (pathX q.1 q.2 (-θ)) * fderiv ℝ g q.1 q.2 ∂((gaussianE E).prod (gaussianE E))
      = -Real.sin θ * ∫ t, fderiv ℝ g (pathX t.1 t.2.1 (-θ)) t.2.2 * fderiv ℝ g t.1 t.2.2
          ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
  have hgm : Measurable g := hg.continuous.measurable
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  have hK0 : 0 ≤ K := (norm_nonneg _).trans (hK 0)
  have h0 := integral_ibpInt_eq_zero hg hM hK θ
  have hA : Integrable (fun t : E × (E × E) =>
      fderiv ℝ g (pathX t.1 t.2.1 (-θ)) t.2.2 * fderiv ℝ g t.1 t.2.2)
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
    refine integrable_triple_of_bound hM0 hK0
      ((measurable_fderiv_apply₂ (measurable_pathX_apply measurable_fst (measurable_fst.comp measurable_snd) _)
        (measurable_snd.comp measurable_snd)).mul
        (measurable_fderiv_apply₂ measurable_fst (measurable_snd.comp measurable_snd))) fun t => ?_
    have := abs_fderiv_mul_fderiv_le hK (pathX t.1 t.2.1 (-θ)) t.1 t.2.2
    nlinarith [mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.1)),
      mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.2))]
  have hB : Integrable (fun t : E × (E × E) => g (pathX t.1 t.2.1 (-θ)) * fderiv ℝ g t.1 t.2.1)
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
    refine integrable_triple_of_bound hM0 hK0
      ((hgm.comp (measurable_pathX_apply measurable_fst (measurable_fst.comp measurable_snd) _)).mul
        (measurable_fderiv_apply₂ measurable_fst (measurable_fst.comp measurable_snd))) fun t => ?_
    have := abs_mul_fderiv_le hM hK (pathX t.1 t.2.1 (-θ)) t.1 t.2.1
    nlinarith [sq_nonneg (K * ‖t.2.2‖), mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.2))]
  have hsplit : ibpInt g θ = fun t => Real.sin (-θ) *
      (fderiv ℝ g (pathX t.1 t.2.1 (-θ)) t.2.2 * fderiv ℝ g t.1 t.2.2)
      - g (pathX t.1 t.2.1 (-θ)) * fderiv ℝ g t.1 t.2.1 := by
    funext t; unfold ibpInt; rw [map_smul, map_neg, smul_eq_mul]; ring
  rw [hsplit, integral_sub (hA.const_mul _) hB, integral_const_mul, sub_eq_zero] at h0
  have hmp : MeasurePreserving (fun t : E × (E × E) => (t.1, t.2.1))
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) ((gaussianE E).prod (gaussianE E)) :=
    (MeasurePreserving.id _).prod measurePreserving_fst
  have hFm : Measurable fun q : E × E => g (pathX q.1 q.2 (-θ)) * fderiv ℝ g q.1 q.2 :=
    (hgm.comp (measurable_pathX_apply measurable_fst measurable_snd _)).mul
      (measurable_fderiv_apply₂ measurable_fst measurable_snd)
  have hmarg := integral_map hmp.measurable.aemeasurable
    (f := fun q : E × E => g (pathX q.1 q.2 (-θ)) * fderiv ℝ g q.1 q.2)
    (by rw [hmp.map_eq]; exact hFm.aestronglyMeasurable)
  rw [hmp.map_eq] at hmarg
  rw [hmarg, ← h0, Real.sin_neg]


/-! ### Step D: Cauchy–Schwarz and the conclusion -/

theorem integral_clm_sq_gaussianE (hd : 2 ≤ Module.finrank ℝ E) (ℓ : E →L[ℝ] ℝ) :
    ∫ w, ℓ w ^ 2 ∂(gaussianE E) = ‖ℓ‖ ^ 2 := by
  rw [integral_eq_lintegral_of_nonneg_ae (Filter.Eventually.of_forall fun w => sq_nonneg _)
    (ℓ.continuous.measurable.pow_const 2).aestronglyMeasurable, lintegral_clm_sq_gaussianE hd,
    ENNReal.toReal_ofReal (sq_nonneg _)]

theorem integrable_norm_fderiv_sq (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) {X : Type*} [MeasurableSpace X]
    {μ : Measure X} [IsFiniteMeasure μ] {a : X → E} (ha : Measurable a) :
    Integrable (fun x => ‖fderiv ℝ g (a x)‖ ^ 2) μ :=
  integrable_of_bound_gauss (((measurable_fderiv ℝ g).comp ha).norm.pow_const 2).aestronglyMeasurable
    (K ^ 2) fun x => by
      rw [abs_of_nonneg (by positivity)]
      exact pow_le_pow_left₀ (norm_nonneg _) (hK _) 2

theorem integral_triple_fderiv_sq_u (hd : 2 ≤ Module.finrank ℝ E) (hM : ∀ z, |g z| ≤ M)
    (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) :
    ∫ t, fderiv ℝ g t.1 t.2.2 ^ 2 ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E)))
      = ∫ z, ‖fderiv ℝ g z‖ ^ 2 ∂(gaussianE E) := by
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  have hK0 : 0 ≤ K := (norm_nonneg _).trans (hK 0)
  have hint : Integrable (fun t : E × (E × E) => fderiv ℝ g t.1 t.2.2 ^ 2)
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
    refine integrable_triple_of_bound hM0 hK0
      ((measurable_fderiv_apply₂ measurable_fst (measurable_snd.comp measurable_snd)).pow_const 2)
      fun t => ?_
    have := abs_fderiv_mul_fderiv_le hK t.1 t.1 t.2.2
    rw [← sq] at this
    nlinarith [mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.1)),
      mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.2))]
  rw [integral_prod _ hint]
  refine integral_congr_ae (Filter.Eventually.of_forall fun u => ?_)
  simp only
  rw [integral_snd_prod (F := fun w => fderiv ℝ g u w ^ 2)
    ((fderiv ℝ g u).continuous.measurable.pow_const 2).aestronglyMeasurable,
    integral_clm_sq_gaussianE hd]

theorem integral_triple_fderiv_sq_x (hd : 2 ≤ Module.finrank ℝ E) (hM : ∀ z, |g z| ≤ M)
    (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) (θ : ℝ) :
    ∫ t, fderiv ℝ g (pathX t.1 t.2.1 (-θ)) t.2.2 ^ 2
        ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E)))
      = ∫ z, ‖fderiv ℝ g z‖ ^ 2 ∂(gaussianE E) := by
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  have hK0 : 0 ≤ K := (norm_nonneg _).trans (hK 0)
  have hint : Integrable (fun t : E × (E × E) => fderiv ℝ g (pathX t.1 t.2.1 (-θ)) t.2.2 ^ 2)
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
    refine integrable_triple_of_bound hM0 hK0
      ((measurable_fderiv_apply₂ (measurable_pathX_apply measurable_fst (measurable_fst.comp measurable_snd) _)
        (measurable_snd.comp measurable_snd)).pow_const 2) fun t => ?_
    have := abs_fderiv_mul_fderiv_le hK (pathX t.1 t.2.1 (-θ)) (pathX t.1 t.2.1 (-θ)) t.2.2
    rw [← sq] at this
    nlinarith [mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.1)),
      mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.2))]
  rw [integral_prod _ hint]
  have hinner : ∀ u : E, ∫ p : E × E, fderiv ℝ g (pathX u p.1 (-θ)) p.2 ^ 2
      ∂((gaussianE E).prod (gaussianE E)) = ∫ v, ‖fderiv ℝ g (pathX u v (-θ))‖ ^ 2 ∂(gaussianE E) := by
    intro u
    have hint2 : Integrable (fun p : E × E => fderiv ℝ g (pathX u p.1 (-θ)) p.2 ^ 2)
        ((gaussianE E).prod (gaussianE E)) := by
      have hm : Measurable fun p : E × E => fderiv ℝ g (pathX u p.1 (-θ)) p.2 ^ 2 :=
        (measurable_fderiv_apply₂ (measurable_pathX_apply measurable_const measurable_fst _)
          measurable_snd).pow_const 2
      refine integrable_of_enorm_le hm.aestronglyMeasurable
        (G := fun p => ENNReal.ofReal (K ^ 2 * ‖p.2‖ ^ 2)) ?_ fun p => ?_
      · rw [lintegral_snd_prod (F := fun w : E => ENNReal.ofReal (K ^ 2 * ‖w‖ ^ 2)) (by fun_prop)]
        simp_rw [ENNReal.ofReal_mul (sq_nonneg K)]
        rw [lintegral_const_mul _ (by fun_prop), lintegral_normSq_gaussianE]
        exact ENNReal.mul_lt_top ENNReal.ofReal_lt_top (by simp)
      · rw [Real.enorm_eq_ofReal_abs]
        refine ENNReal.ofReal_le_ofReal ?_
        have := abs_fderiv_mul_fderiv_le hK (pathX u p.1 (-θ)) (pathX u p.1 (-θ)) p.2
        rwa [← sq] at this
    rw [integral_prod _ hint2]
    refine integral_congr_ae (Filter.Eventually.of_forall fun v => ?_)
    simp only
    exact integral_clm_sq_gaussianE hd _
  simp_rw [hinner]
  have hint3 : Integrable (fun q : E × E => ‖fderiv ℝ g (pathX q.1 q.2 (-θ))‖ ^ 2)
      ((gaussianE E).prod (gaussianE E)) :=
    integrable_norm_fderiv_sq hK (measurable_pathX_apply measurable_fst measurable_snd _)
  rw [← integral_prod _ hint3]
  have hmp := measurePreserving_rotProd (E := E) (-θ)
  have := integral_map hmp.measurable.aemeasurable (f := fun q : E × E => ‖fderiv ℝ g q.1‖ ^ 2)
    (by rw [hmp.map_eq]
        exact (((measurable_fderiv ℝ g).comp measurable_fst).norm.pow_const 2).aestronglyMeasurable)
  rw [hmp.map_eq] at this
  simp only at this
  rw [← this, integral_fst_prod (F := fun z => ‖fderiv ℝ g z‖ ^ 2)
    ((measurable_fderiv ℝ g).norm.pow_const 2).aestronglyMeasurable]

/-- **Step D.**  For `θ ∈ [0, π/2]`, `E[g(X) g'(X_θ)[Y_θ]] ≥ −sin θ · E‖g'‖²`. -/
theorem integral_rot_ibp_ge (hd : 2 ≤ Module.finrank ℝ E) (hg : Differentiable ℝ g)
    (hM : ∀ z, |g z| ≤ M) (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) {θ : ℝ} (hθ : 0 ≤ Real.sin θ) :
    -Real.sin θ * ∫ z, ‖fderiv ℝ g z‖ ^ 2 ∂(gaussianE E)
      ≤ ∫ q, g (pathX q.1 q.2 (-θ)) * fderiv ℝ g q.1 q.2 ∂((gaussianE E).prod (gaussianE E)) := by
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  have hK0 : 0 ≤ K := (norm_nonneg _).trans (hK 0)
  rw [integral_rot_ibp hg hM hK θ]
  refine mul_le_mul_of_nonpos_left ?_ (by linarith)
  have hA : Integrable (fun t : E × (E × E) =>
      fderiv ℝ g (pathX t.1 t.2.1 (-θ)) t.2.2 * fderiv ℝ g t.1 t.2.2)
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
    refine integrable_triple_of_bound hM0 hK0
      ((measurable_fderiv_apply₂ (measurable_pathX_apply measurable_fst (measurable_fst.comp measurable_snd) _)
        (measurable_snd.comp measurable_snd)).mul
        (measurable_fderiv_apply₂ measurable_fst (measurable_snd.comp measurable_snd))) fun t => ?_
    have := abs_fderiv_mul_fderiv_le hK (pathX t.1 t.2.1 (-θ)) t.1 t.2.2
    nlinarith [mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.1)),
      mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.2))]
  have hx : Integrable (fun t : E × (E × E) => fderiv ℝ g (pathX t.1 t.2.1 (-θ)) t.2.2 ^ 2)
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
    refine integrable_triple_of_bound hM0 hK0
      ((measurable_fderiv_apply₂ (measurable_pathX_apply measurable_fst (measurable_fst.comp measurable_snd) _)
        (measurable_snd.comp measurable_snd)).pow_const 2) fun t => ?_
    have := abs_fderiv_mul_fderiv_le hK (pathX t.1 t.2.1 (-θ)) (pathX t.1 t.2.1 (-θ)) t.2.2
    rw [← sq] at this
    nlinarith [mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.1)),
      mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.2))]
  have hu : Integrable (fun t : E × (E × E) => fderiv ℝ g t.1 t.2.2 ^ 2)
      ((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
    refine integrable_triple_of_bound hM0 hK0
      ((measurable_fderiv_apply₂ measurable_fst (measurable_snd.comp measurable_snd)).pow_const 2)
      fun t => ?_
    have := abs_fderiv_mul_fderiv_le hK t.1 t.1 t.2.2
    rw [← sq] at this
    nlinarith [mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.1)),
      mul_nonneg hM0 (mul_nonneg hK0 (norm_nonneg t.2.2))]
  calc ∫ t, fderiv ℝ g (pathX t.1 t.2.1 (-θ)) t.2.2 * fderiv ℝ g t.1 t.2.2
        ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E)))
      ≤ ∫ t, (1 / 2 : ℝ) * (fderiv ℝ g (pathX t.1 t.2.1 (-θ)) t.2.2 ^ 2 + fderiv ℝ g t.1 t.2.2 ^ 2)
          ∂((gaussianE E).prod ((gaussianE E).prod (gaussianE E))) := by
        refine integral_mono hA ((hx.add hu).const_mul _) fun t => ?_
        simp only
        nlinarith [sq_nonneg (fderiv ℝ g (pathX t.1 t.2.1 (-θ)) t.2.2 - fderiv ℝ g t.1 t.2.2)]
    _ = ∫ z, ‖fderiv ℝ g z‖ ^ 2 ∂(gaussianE E) := by
        rw [integral_const_mul, integral_add hx hu, integral_triple_fderiv_sq_x hd hM hK,
          integral_triple_fderiv_sq_u hd hM hK]
        ring

/-- **The Gaussian Poincaré inequality for bounded `C¹` functions with bounded derivative**,
with the sharp constant `1`: `Var_γ g ≤ E‖g'‖²`. -/
theorem variance_le_integral_norm_fderiv_sq (hd : 2 ≤ Module.finrank ℝ E)
    (hg : Differentiable ℝ g) (hM : ∀ z, |g z| ≤ M) (hK : ∀ z, ‖fderiv ℝ g z‖ ≤ K) :
    variance g (gaussianE E) ≤ ∫ z, ‖fderiv ℝ g z‖ ^ 2 ∂(gaussianE E) := by
  have hpi : (0 : ℝ) ≤ Real.pi / 2 := by positivity
  have h := neg_variance_eq_integral_path hg hM hK
  have hΦint : IntervalIntegrable (fun θ => ∫ p, g p.1 * fderiv ℝ g (pathX p.1 p.2 θ)
      (pathY p.1 p.2 θ) ∂((gaussianE E).prod (gaussianE E))) volume 0 (Real.pi / 2) := by
    rw [intervalIntegrable_iff_integrableOn_Ioc_of_le hpi]
    exact (integrable_path_integrand hg hM hK).integral_prod_right
  have hsin : IntervalIntegrable (fun θ => -Real.sin θ * ∫ z, ‖fderiv ℝ g z‖ ^ 2 ∂(gaussianE E))
      volume 0 (Real.pi / 2) := (Continuous.intervalIntegrable (by fun_prop) _ _)
  have hmono : ∀ θ ∈ Set.Icc (0 : ℝ) (Real.pi / 2),
      -Real.sin θ * ∫ z, ‖fderiv ℝ g z‖ ^ 2 ∂(gaussianE E)
        ≤ ∫ p, g p.1 * fderiv ℝ g (pathX p.1 p.2 θ) (pathY p.1 p.2 θ)
          ∂((gaussianE E).prod (gaussianE E)) := fun θ hθ => by
    rw [integral_path_eq_rot hg θ]
    exact integral_rot_ibp_ge hd hg hM hK
      (Real.sin_nonneg_of_nonneg_of_le_pi hθ.1 (by linarith [hθ.2, Real.pi_pos]))
  have hle := intervalIntegral.integral_mono_on hpi hsin hΦint hmono
  rw [← h, intervalIntegral.integral_mul_const, intervalIntegral.integral_neg, integral_sin,
    Real.cos_zero, Real.cos_pi_div_two, sub_zero] at hle
  linarith

end C1Poincare
