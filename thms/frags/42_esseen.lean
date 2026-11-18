/-!
# §42  Esseen's smoothing inequality, second half: from test functions to half-lines

The first half (§41) bounds `|∫ g dμ − ∫ g dν|` for a test function `g` by the
`|𝓕g|`-weighted characteristic-function gap.  This section closes the argument.

* `esseen_lemma`: for probability measures `μ, ν, K` on `ℝ` with `F_ν` `m`-Lipschitz and `K`
  putting mass at most `ε` outside `[−a, a]`, the distribution-function gap at `x₀` is
  controlled by the `K`-smoothed gap `D(x) = ∫ (F_μ − F_ν)(x − y) dK(y)` at `x₀ ± a`:
  `|F_μ(x₀) − F_ν(x₀)| (1 − ε) ≤ |D(x₀ + a)| + |D(x₀ − a)| + 2ma + ε`.
* `smoothStep K x L u = K((x − L − u, x − u])` is the `K`-smoothed indicator of `(x − L, x]`; it
  is continuous when `K` is atomless, integrable with mass `L`, and its Fourier transform is
  `e^{−2πiξx}(e^{2πiξL} − 1)/(2πiξ) · φ_K(2πξ)`, of norm at most `|φ_K(2πξ)|/(π|ξ|)`.
* `smoothDiff_sub_smoothDiff`: `D(x) − D(x − L)` is the gap of expectations of the smooth step,
  so §41 bounds it by `∫ |φ_K(2πξ)|/(π|ξ|) · |φ_μ − φ_ν|(2πξ) dξ`, uniformly in `L`; letting
  `L → ∞` (`tendsto_smoothDiff_atTop`) bounds `|D(x)|` itself (`abs_smoothDiff_le`).
* `esseen_inequality`: the assembled kernel-form inequality
  `|F_μ(x₀) − F_ν(x₀)| (1 − ε) ≤ 2 ∫ |φ_K(2πξ)|/(π|ξ|) · |φ_μ(2πξ) − φ_ν(2πξ)| dξ + 2ma + ε`.
* `esseen_inequality_gaussian`: the same with the Gaussian kernel `N(0, v)`, whose hypotheses
  (atomless, Chebyshev tail `v/a²`, integrable weight) are all discharged.

What this does **not** yet give is a Kolmogorov–Smirnov rate for the coordinate law of
`H D_ε v`: the coordinate law is a lattice law, its characteristic function is periodic, and the
§40 bound only covers `|t| ≤ √d`, so a decaying rate needs a kernel whose characteristic function
vanishes beyond `|t| ~ √d` (Fejér), whose Fourier transform is not computed here.  With the
Gaussian kernel the region `|t| > √d` contributes a term that does not decay in `d`.
-/

open Filter
open scoped Topology FourierTransform


section Esseen

/-- The distribution function of a measure on `ℝ`, as a real number. -/
noncomputable def cdfR (μ : Measure ℝ) (x : ℝ) : ℝ := μ.real (Set.Iic x)

theorem cdfR_mono (μ : Measure ℝ) [IsFiniteMeasure μ] : Monotone (cdfR μ) := fun x y hxy =>
  measureReal_mono (Set.Iic_subset_Iic.mpr hxy)

theorem cdfR_nonneg (μ : Measure ℝ) (x : ℝ) : 0 ≤ cdfR μ x := ENNReal.toReal_nonneg

theorem cdfR_le_one (μ : Measure ℝ) [IsProbabilityMeasure μ] (x : ℝ) : cdfR μ x ≤ 1 := by
  unfold cdfR
  calc μ.real (Set.Iic x) ≤ μ.real Set.univ := measureReal_mono (Set.subset_univ _)
    _ = 1 := by simp

theorem measurable_cdfR (μ : Measure ℝ) [IsFiniteMeasure μ] : Measurable (cdfR μ) :=
  (cdfR_mono μ).measurable

/-- The kernel-smoothed difference of two distribution functions,
`∫ (F_μ − F_ν)(x − y) dK(y)`. -/
noncomputable def smoothDiff (μ ν K : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y, (cdfR μ (x - y) - cdfR ν (x - y)) ∂K

/-- **Esseen's lemma**: a distribution function is controlled by its kernel-smoothed version.
If `F_ν` is `m`-Lipschitz (`ν` has density at most `m`) and the kernel puts mass at most `ε`
outside `[−a, a]`, then `|F_μ(x₀) − F_ν(x₀)| (1 − ε) ≤ |D(x₀ + a)| + |D(x₀ − a)| + 2ma + ε`
with `D` the smoothed difference. -/
theorem esseen_lemma (μ ν K : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    [IsProbabilityMeasure K] {m : ℝ} (hm : 0 ≤ m)
    (hG : ∀ x y, x ≤ y → cdfR ν y - cdfR ν x ≤ m * (y - x)) {a ε : ℝ} (ha : 0 ≤ a)
    (hε : K.real {y | a < ‖y‖} ≤ ε) (x₀ : ℝ) :
    |cdfR μ x₀ - cdfR ν x₀| * (1 - ε)
      ≤ |smoothDiff μ ν K (x₀ + a)| + |smoothDiff μ ν K (x₀ - a)| + 2 * m * a + ε := by
  set Δ : ℝ → ℝ := fun z => cdfR μ z - cdfR ν z with hΔ
  have hΔm : Measurable Δ := (measurable_cdfR μ).sub (measurable_cdfR ν)
  have hΔb : ∀ z, |Δ z| ≤ 1 := fun z => by
    rw [hΔ]; simp only
    rw [abs_le]
    have := cdfR_nonneg μ z; have := cdfR_le_one μ z
    have := cdfR_nonneg ν z; have := cdfR_le_one ν z
    constructor <;> linarith
  have hε0 : 0 ≤ K.real {y | a < ‖y‖} := measureReal_nonneg
  have hset : MeasurableSet {y : ℝ | a < ‖y‖} := measurableSet_lt measurable_const measurable_norm
  set ε' := K.real {y | a < ‖y‖} with hε'
  have hcompl : K.real {y : ℝ | ‖y‖ ≤ a} = 1 - ε' := by
    have : {y : ℝ | ‖y‖ ≤ a} = {y : ℝ | a < ‖y‖}ᶜ := by
      ext y; simp [not_lt]
    rw [this, measureReal_compl hset, measureReal_def, measure_univ, ENNReal.toReal_one]
  have hε1 : ε' ≤ 1 := by
    have := measureReal_nonneg (μ := K) (s := {y : ℝ | ‖y‖ ≤ a}); linarith
  -- the two directional lower/upper bounds on `Δ` near `x₀`
  have hup : ∀ s, 0 ≤ s → s ≤ 2 * a → Δ x₀ - 2 * m * a ≤ Δ (x₀ + s) := by
    intro s hs hs2
    have h1 : cdfR μ x₀ ≤ cdfR μ (x₀ + s) := cdfR_mono μ (by linarith)
    have h2 := hG x₀ (x₀ + s) (by linarith)
    rw [hΔ]; simp only
    have : m * (x₀ + s - x₀) ≤ 2 * m * a := by
      rw [show x₀ + s - x₀ = s by ring]; nlinarith
    linarith
  have hdown : ∀ s, 0 ≤ s → s ≤ 2 * a → Δ (x₀ - s) ≤ Δ x₀ + 2 * m * a := by
    intro s hs hs2
    have h1 : cdfR μ (x₀ - s) ≤ cdfR μ x₀ := cdfR_mono μ (by linarith)
    have h2 := hG (x₀ - s) x₀ (by linarith)
    rw [hΔ]; simp only
    have : m * (x₀ - (x₀ - s)) ≤ 2 * m * a := by
      rw [show x₀ - (x₀ - s) = s by ring]; nlinarith
    linarith
  -- integrability of the shifted differences and of the two-valued comparison functions
  have hint : ∀ x, Integrable (fun y => Δ (x - y)) K := fun x =>
    Integrable.of_bound (hΔm.comp (measurable_const.sub measurable_id)).aestronglyMeasurable 1
      (Filter.Eventually.of_forall fun y => by rw [Real.norm_eq_abs]; exact hΔb _)
  have hstep : ∀ c : ℝ, Integrable (fun y : ℝ => {y : ℝ | ‖y‖ ≤ a}.indicator (fun _ => c) y
      + {y : ℝ | a < ‖y‖}.indicator (fun _ => (-1 : ℝ)) y) K :=
    fun c => ((integrable_const c).indicator (measurableSet_le measurable_norm measurable_const)).add
      ((integrable_const _).indicator hset)
  have hstep_int : ∀ c : ℝ, ∫ y, ({y : ℝ | ‖y‖ ≤ a}.indicator (fun _ => c) y
      + {y : ℝ | a < ‖y‖}.indicator (fun _ => (-1 : ℝ)) y) ∂K = c * (1 - ε') - ε' := by
    intro c
    rw [integral_add ((integrable_const c).indicator (measurableSet_le measurable_norm measurable_const))
      ((integrable_const _).indicator hset), integral_indicator_const _
      (measurableSet_le measurable_norm measurable_const), integral_indicator_const _ hset, hcompl]
    simp only [smul_eq_mul]
    ring
  rcases le_or_gt 1 ε with hε1' | hε1'
  · -- trivial: the left side is nonpositive
    have : |Δ x₀| * (1 - ε) ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (abs_nonneg _) (by linarith)
    have h0 : 0 ≤ |smoothDiff μ ν K (x₀ + a)| + |smoothDiff μ ν K (x₀ - a)| + 2 * m * a + ε := by
      positivity
    linarith
  rcases le_or_gt 0 (Δ x₀) with hD | hD
  · -- `Δ x₀ ≥ 0`: look to the right
    have hpt : ∀ y, {y : ℝ | ‖y‖ ≤ a}.indicator (fun _ => Δ x₀ - 2 * m * a) y
        + {y : ℝ | a < ‖y‖}.indicator (fun _ => (-1 : ℝ)) y ≤ Δ (x₀ + a - y) := by
      intro y
      by_cases hy : ‖y‖ ≤ a
      · rw [Set.indicator_of_mem (by exact hy), Set.indicator_of_notMem (by
          show ¬ a < ‖y‖; exact not_lt.mpr hy), add_zero]
        rw [Real.norm_eq_abs, abs_le] at hy
        rw [show x₀ + a - y = x₀ + (a - y) by ring]
        exact hup (a - y) (by linarith) (by linarith)
      · rw [Set.indicator_of_notMem (by exact hy), Set.indicator_of_mem (by
          show a < ‖y‖; exact not_le.mp hy), zero_add]
        exact (abs_le.mp (hΔb _)).1
    have hle : Δ x₀ * (1 - ε') - 2 * m * a * (1 - ε') - ε' ≤ smoothDiff μ ν K (x₀ + a) := by
      have := integral_mono (hstep _) (hint (x₀ + a)) hpt
      rw [hstep_int] at this
      unfold smoothDiff
      linarith
    have habs : smoothDiff μ ν K (x₀ + a) ≤ |smoothDiff μ ν K (x₀ + a)| := le_abs_self _
    have h1 : |Δ x₀| = Δ x₀ := abs_of_nonneg hD
    rw [h1]
    have h2 : Δ x₀ * (1 - ε) ≤ Δ x₀ * (1 - ε') := mul_le_mul_of_nonneg_left (by linarith) hD
    have h3 : 0 ≤ |smoothDiff μ ν K (x₀ - a)| := abs_nonneg _
    nlinarith [mul_nonneg hm ha, mul_nonneg (mul_nonneg hm ha) hε0]
  · -- `Δ x₀ < 0`: look to the left
    have hpt : ∀ y, Δ (x₀ - a - y) ≤ {y : ℝ | ‖y‖ ≤ a}.indicator (fun _ => Δ x₀ + 2 * m * a) y
        + {y : ℝ | a < ‖y‖}.indicator (fun _ => (1 : ℝ)) y := by
      intro y
      by_cases hy : ‖y‖ ≤ a
      · rw [Set.indicator_of_mem (by exact hy), Set.indicator_of_notMem (by
          show ¬ a < ‖y‖; exact not_lt.mpr hy), add_zero]
        rw [Real.norm_eq_abs, abs_le] at hy
        rw [show x₀ - a - y = x₀ - (a + y) by ring]
        exact hdown (a + y) (by linarith) (by linarith)
      · rw [Set.indicator_of_notMem (by exact hy), Set.indicator_of_mem (by
          show a < ‖y‖; exact not_le.mp hy), zero_add]
        exact (abs_le.mp (hΔb _)).2
    have hstep' : Integrable (fun y : ℝ => {y : ℝ | ‖y‖ ≤ a}.indicator (fun _ => Δ x₀ + 2 * m * a) y
        + {y : ℝ | a < ‖y‖}.indicator (fun _ => (1 : ℝ)) y) K :=
      ((integrable_const _).indicator (measurableSet_le measurable_norm measurable_const)).add
        ((integrable_const _).indicator hset)
    have hle : smoothDiff μ ν K (x₀ - a) ≤ (Δ x₀ + 2 * m * a) * (1 - ε') + ε' := by
      have := integral_mono (hint (x₀ - a)) hstep' hpt
      rw [integral_add ((integrable_const _).indicator (measurableSet_le measurable_norm measurable_const))
        ((integrable_const _).indicator hset), integral_indicator_const _
        (measurableSet_le measurable_norm measurable_const), integral_indicator_const _ hset,
        hcompl] at this
      simp only [smul_eq_mul, mul_one] at this
      unfold smoothDiff
      linarith
    have habs : -|smoothDiff μ ν K (x₀ - a)| ≤ smoothDiff μ ν K (x₀ - a) := neg_abs_le _
    have h1 : |Δ x₀| = -Δ x₀ := abs_of_neg hD
    rw [h1]
    have h2 : -Δ x₀ * (1 - ε) ≤ -Δ x₀ * (1 - ε') :=
      mul_le_mul_of_nonneg_left (by linarith) (by linarith)
    have h3 : 0 ≤ |smoothDiff μ ν K (x₀ + a)| := abs_nonneg _
    nlinarith [mul_nonneg hm ha, mul_nonneg (mul_nonneg hm ha) hε0]

end Esseen


section EsseenFourier

/-- The indicator `1_{(x − L − p₁, x − p₁]}(p₂)` on the plane. -/
noncomputable def stepInd (x L : ℝ) (p : ℝ × ℝ) : ℝ :=
  (Set.Ioc (x - L - p.1) (x - p.1)).indicator (fun _ => (1 : ℝ)) p.2

theorem stepInd_nonneg (x L : ℝ) (p : ℝ × ℝ) : 0 ≤ stepInd x L p :=
  Set.indicator_nonneg (fun _ _ => zero_le_one) _

theorem stepInd_le_one (x L : ℝ) (p : ℝ × ℝ) : stepInd x L p ≤ 1 :=
  Set.indicator_le_self' (fun _ _ => zero_le_one) _

theorem norm_stepInd_le_one (x L : ℝ) (p : ℝ × ℝ) : ‖stepInd x L p‖ ≤ 1 := by
  rw [Real.norm_of_nonneg (stepInd_nonneg x L p)]
  exact stepInd_le_one x L p

/-- The membership condition in the step indicator is symmetric in the two coordinates. -/
theorem mem_Ioc_swap {x L u y : ℝ} :
    y ∈ Set.Ioc (x - L - u) (x - u) ↔ u ∈ Set.Ioc (x - L - y) (x - y) := by
  simp only [Set.mem_Ioc]
  constructor <;> rintro ⟨h1, h2⟩ <;> constructor <;> linarith

theorem stepInd_swap (x L u y : ℝ) : stepInd x L (u, y) = stepInd x L (y, u) := by
  unfold stepInd
  by_cases h : y ∈ Set.Ioc (x - L - u) (x - u)
  · rw [Set.indicator_of_mem h, Set.indicator_of_mem (mem_Ioc_swap.mp h)]
  · rw [Set.indicator_of_notMem h, Set.indicator_of_notMem (fun h' => h (mem_Ioc_swap.mpr h'))]

theorem measurable_stepInd (x L : ℝ) : Measurable (stepInd x L) := by
  have : stepInd x L
      = {p : ℝ × ℝ | x - L - p.1 < p.2 ∧ p.2 ≤ x - p.1}.indicator (fun _ => (1 : ℝ)) := by
    funext p; simp [stepInd, Set.indicator, Set.mem_Ioc]
  rw [this]
  exact measurable_const.indicator ((measurableSet_lt (by fun_prop) measurable_snd).inter
    (measurableSet_le measurable_snd (by fun_prop)))

theorem integrable_stepInd_prod (μ K : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure K]
    (x L : ℝ) : Integrable (stepInd x L) (μ.prod K) :=
  Integrable.of_bound (measurable_stepInd x L).aestronglyMeasurable 1
    (Filter.Eventually.of_forall (norm_stepInd_le_one x L))

/-- `∫ 1_{(x−L−y, x−y]}(u) du = L`: the step indicator has total mass `L` in its second slot. -/
theorem integral_stepInd_volume (x : ℝ) {L : ℝ} (hL : 0 ≤ L) (y : ℝ) :
    ∫ u, stepInd x L (u, y) = L := by
  have : (fun u => stepInd x L (u, y))
      = (Set.Ioc (x - L - y) (x - y)).indicator (fun _ => (1 : ℝ)) :=
    funext fun u => by rw [stepInd_swap]; rfl
  rw [this, integral_indicator_const _ measurableSet_Ioc, smul_eq_mul, mul_one, measureReal_def,
    Real.volume_Ioc, ENNReal.toReal_ofReal (by linarith)]
  ring

theorem integrable_stepInd_volume_prod (K : Measure ℝ) [IsProbabilityMeasure K] (x L : ℝ) :
    Integrable (stepInd x L) (volume.prod K) := by
  refine ⟨(measurable_stepInd x L).aestronglyMeasurable, ?_⟩
  rw [hasFiniteIntegral_iff_ofReal (Filter.Eventually.of_forall (stepInd_nonneg x L))]
  rw [lintegral_prod_symm _ ((measurable_stepInd x L).ennreal_ofReal).aemeasurable]
  have hinner : ∀ y : ℝ, ∫⁻ u, ENNReal.ofReal (stepInd x L (u, y)) = ENNReal.ofReal L := by
    intro y
    have : ∀ u : ℝ, ENNReal.ofReal (stepInd x L (u, y))
        = (Set.Ioc (x - L - y) (x - y)).indicator (fun _ => (1 : ℝ≥0∞)) u := by
      intro u
      rw [stepInd_swap]
      show ENNReal.ofReal ((Set.Ioc (x - L - y) (x - y)).indicator (fun _ => (1 : ℝ)) u) = _
      by_cases h : u ∈ Set.Ioc (x - L - y) (x - y)
      · rw [Set.indicator_of_mem h, Set.indicator_of_mem h, ENNReal.ofReal_one]
      · rw [Set.indicator_of_notMem h, Set.indicator_of_notMem h, ENNReal.ofReal_zero]
    simp_rw [this]
    rw [lintegral_indicator_const measurableSet_Ioc, Real.volume_Ioc, one_mul]
    congr 1; ring
  simp_rw [hinner]
  rw [lintegral_const, measure_univ, mul_one]
  exact ENNReal.ofReal_lt_top

/-- The kernel-smoothed indicator of the interval `(x − L, x]`:
`g(u) = K((x − L − u, x − u])`. -/
noncomputable def smoothStep (K : Measure ℝ) (x L u : ℝ) : ℝ :=
  K.real (Set.Ioc (x - L - u) (x - u))

theorem smoothStep_nonneg (K : Measure ℝ) (x L u : ℝ) : 0 ≤ smoothStep K x L u :=
  measureReal_nonneg

/-- The smooth step as an integral of the step indicator. -/
theorem smoothStep_eq_integral (K : Measure ℝ) [IsFiniteMeasure K] (x L u : ℝ) :
    smoothStep K x L u = ∫ y, stepInd x L (u, y) ∂K := by
  simp only [smoothStep, stepInd]
  rw [integral_indicator_const _ measurableSet_Ioc, smul_eq_mul, mul_one]

theorem Ioc_eq_Iic_diff (a b : ℝ) : Set.Ioc a b = Set.Iic b \ Set.Iic a := by
  ext z
  simp only [Set.mem_Ioc, Set.mem_sdiff, Set.mem_Iic, not_le]
  constructor <;> rintro ⟨h1, h2⟩ <;> exact ⟨h2, h1⟩

theorem smoothStep_eq_sub (K : Measure ℝ) [IsFiniteMeasure K] (x : ℝ) {L : ℝ} (hL : 0 ≤ L)
    (u : ℝ) : smoothStep K x L u = cdfR K (x - u) - cdfR K (x - L - u) := by
  unfold smoothStep cdfR
  rw [Ioc_eq_Iic_diff, measureReal_sdiff (Set.Iic_subset_Iic.mpr (by linarith)) measurableSet_Iic]

theorem smoothStep_eq_fun (K : Measure ℝ) [IsFiniteMeasure K] (x : ℝ) {L : ℝ} (hL : 0 ≤ L) :
    smoothStep K x L = fun u => cdfR K (x - u) - cdfR K (x - L - u) :=
  funext fun u => smoothStep_eq_sub K x hL u

theorem measurable_smoothStep (K : Measure ℝ) [IsFiniteMeasure K] (x : ℝ) {L : ℝ} (hL : 0 ≤ L) :
    Measurable (smoothStep K x L) := by
  rw [smoothStep_eq_fun K x hL]
  have hm : Measurable (cdfR K) := measurable_cdfR K
  exact (hm.comp (measurable_const.sub measurable_id)).sub
    (hm.comp (measurable_const.sub measurable_id))

theorem integrable_smoothStep (K : Measure ℝ) [IsProbabilityMeasure K] (x : ℝ) {L : ℝ}
    (hL : 0 ≤ L) : Integrable (smoothStep K x L) := by
  have h := (integrable_stepInd_volume_prod K x L).integral_prod_left
  have : smoothStep K x L = fun u => ∫ y, stepInd x L (u, y) ∂K :=
    funext (smoothStep_eq_integral K x L)
  rw [this]
  exact h

/-- `∫ g(u) du = L`: the smooth step has total mass `L`. -/
theorem integral_smoothStep (K : Measure ℝ) [IsProbabilityMeasure K] (x : ℝ) {L : ℝ} (hL : 0 ≤ L) :
    ∫ u, smoothStep K x L u = L := by
  simp_rw [smoothStep_eq_integral]
  rw [← integral_prod (stepInd x L) (integrable_stepInd_volume_prod K x L),
    integral_prod_symm _ (integrable_stepInd_volume_prod K x L)]
  simp_rw [integral_stepInd_volume x hL]
  rw [integral_const, measureReal_def, measure_univ, ENNReal.toReal_one, one_smul]

/-- The distribution function of an atomless measure is continuous. -/
theorem continuous_cdfR (K : Measure ℝ) [IsProbabilityMeasure K] [NullSingletonClass K] :
    Continuous (cdfR K) := by
  have hK : cdfR K = cdf K := funext fun z => (cdf_eq_real K z).symm
  rw [hK, continuous_iff_continuousAt]
  intro x
  rw [(cdf K).mono.continuousAt_iff_leftLim_eq_rightLim, (cdf K).rightLim_eq]
  have h := (cdf K).measure_singleton x
  rw [measure_cdf, measure_singleton, eq_comm, ENNReal.ofReal_eq_zero] at h
  exact le_antisymm ((cdf K).mono.leftLim_le le_rfl) (by linarith)

theorem continuous_smoothStep (K : Measure ℝ) [IsProbabilityMeasure K] [NullSingletonClass K]
    (x : ℝ) {L : ℝ} (hL : 0 ≤ L) : Continuous (smoothStep K x L) := by
  rw [smoothStep_eq_fun K x hL]
  have hc := continuous_cdfR K
  exact (hc.comp (continuous_const.sub continuous_id)).sub
    (hc.comp (continuous_const.sub continuous_id))

/-- The expectation of the smooth step under `μ` is the `K`-average of `F_μ(x−y) − F_μ(x−L−y)`. -/
theorem integral_smoothStep_eq (μ K : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure K]
    (x : ℝ) {L : ℝ} (hL : 0 ≤ L) :
    ∫ u, smoothStep K x L u ∂μ = ∫ y, (cdfR μ (x - y) - cdfR μ (x - L - y)) ∂K := by
  simp_rw [smoothStep_eq_integral]
  rw [integral_integral_swap (f := fun u y => stepInd x L (u, y)) (integrable_stepInd_prod μ K x L)]
  refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
  show ∫ u, stepInd x L (u, y) ∂μ = _
  simp_rw [stepInd_swap x L _ y]
  rw [← smoothStep_eq_integral μ x L y, smoothStep_eq_sub μ x hL y]

theorem integrable_cdfR_shift (ρ K : Measure ℝ) [IsProbabilityMeasure ρ] [IsProbabilityMeasure K]
    (z : ℝ) : Integrable (fun y => cdfR ρ (z - y)) K :=
  Integrable.of_bound ((measurable_cdfR ρ).comp (measurable_const.sub measurable_id)).aestronglyMeasurable
    1 (Filter.Eventually.of_forall fun y => by
      rw [Real.norm_of_nonneg (cdfR_nonneg _ _)]; exact cdfR_le_one _ _)

/-- The smoothed difference at `x` minus the one at `x − L` is the gap of expectations of the
smooth step. -/
theorem smoothDiff_sub_smoothDiff (μ ν K : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] [IsProbabilityMeasure K] (x : ℝ) {L : ℝ} (hL : 0 ≤ L) :
    smoothDiff μ ν K x - smoothDiff μ ν K (x - L)
      = ∫ u, smoothStep K x L u ∂μ - ∫ u, smoothStep K x L u ∂ν := by
  rw [integral_smoothStep_eq μ K x hL, integral_smoothStep_eq ν K x hL, smoothDiff, smoothDiff,
    integral_sub (integrable_cdfR_shift μ K x) (integrable_cdfR_shift ν K x),
    integral_sub (integrable_cdfR_shift μ K (x - L)) (integrable_cdfR_shift ν K (x - L)),
    integral_sub (integrable_cdfR_shift μ K x) (integrable_cdfR_shift μ K (x - L)),
    integral_sub (integrable_cdfR_shift ν K x) (integrable_cdfR_shift ν K (x - L))]
  ring

/-- The smoothed difference vanishes at `−∞`. -/
theorem tendsto_smoothDiff_atTop (μ ν K : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] [IsProbabilityMeasure K] (x : ℝ) :
    Tendsto (fun L => smoothDiff μ ν K (x - L)) atTop (𝓝 0) := by
  have hcdf : ∀ (ρ : Measure ℝ) [IsProbabilityMeasure ρ] (y : ℝ),
      Tendsto (fun L => cdfR ρ (x - L - y)) atTop (𝓝 0) := by
    intro ρ _ y
    have h1 : Tendsto (fun L : ℝ => x - L - y) atTop atBot := by
      have := tendsto_atBot_add_const_left atTop (x - y) (tendsto_neg_atTop_atBot (G := ℝ))
      exact this.congr fun L => by show x - y + -L = x - L - y; ring
    have hρ : cdfR ρ = cdf ρ := funext fun z => (cdf_eq_real ρ z).symm
    rw [hρ]
    exact (tendsto_cdf_atBot (μ := ρ)).comp h1
  have hmeas : ∀ L : ℝ, AEStronglyMeasurable (fun y => cdfR μ (x - L - y) - cdfR ν (x - L - y)) K :=
    fun L => ((integrable_cdfR_shift μ K (x - L)).sub (integrable_cdfR_shift ν K (x - L))).1
  have h := tendsto_integral_filter_of_dominated_convergence (μ := K)
    (F := fun L y => cdfR μ (x - L - y) - cdfR ν (x - L - y)) (f := fun _ => (0 : ℝ)) (fun _ => 2)
    (Filter.Eventually.of_forall hmeas)
    (Filter.Eventually.of_forall fun L => Filter.Eventually.of_forall fun y => by
      rw [Real.norm_eq_abs, abs_le]
      have := cdfR_nonneg μ (x - L - y); have := cdfR_le_one μ (x - L - y)
      have := cdfR_nonneg ν (x - L - y); have := cdfR_le_one ν (x - L - y)
      constructor <;> linarith)
    (integrable_const 2)
    (Filter.Eventually.of_forall fun y => by simpa using (hcdf μ y).sub (hcdf ν y))
  simpa [smoothDiff] using h

theorem norm_exp_mul_I_mul (a u : ℝ) : ‖Complex.exp (↑a * Complex.I * ↑u)‖ = 1 := by
  rw [show (↑a * Complex.I * ↑u : ℂ) = ↑(a * u) * Complex.I by push_cast; ring,
    Complex.norm_exp_ofReal_mul_I]

/-- **Fourier transform of the smooth step**, for `ξ ≠ 0`:
`𝓕g(ξ) = e^{−2πiξx} (e^{2πiξL} − 1)/(2πiξ) · φ_K(2πξ)`. -/
theorem fourier_smoothStep (K : Measure ℝ) [IsProbabilityMeasure K] (x : ℝ) {L : ℝ} (hL : 0 ≤ L)
    {ξ : ℝ} (hξ : ξ ≠ 0) :
    𝓕 (fun u => (smoothStep K x L u : ℂ)) ξ
      = Complex.exp (↑(-(2 * π * ξ * x)) * Complex.I)
        * (Complex.exp (↑(2 * π * ξ * L) * Complex.I) - 1) / (↑(2 * π * ξ) * Complex.I)
        * charFun K (2 * π * ξ) := by
  set c : ℂ := ↑(-(2 * π * ξ)) * Complex.I with hc
  have h2πξ : (2 * π * ξ : ℝ) ≠ 0 := by positivity
  have hc0 : c ≠ 0 := by
    rw [hc]
    refine mul_ne_zero ?_ Complex.I_ne_zero
    exact_mod_cast neg_ne_zero.mpr h2πξ
  have hd0 : (↑(2 * π * ξ) * Complex.I : ℂ) ≠ 0 :=
    mul_ne_zero (by exact_mod_cast h2πξ) Complex.I_ne_zero
  have hexp : ∀ u : ℝ, Complex.exp (↑(-2 * π * inner ℝ u ξ) * Complex.I) = Complex.exp (c * (u : ℂ)) := by
    intro u
    congr 1
    simp only [RCLike.inner_apply, conj_trivial, hc]
    push_cast
    ring
  have hmeas : AEStronglyMeasurable (fun p : ℝ × ℝ => Complex.exp (c * (p.1 : ℂ)) * (stepInd x L p : ℂ))
      (volume.prod K) := by
    refine (Continuous.aestronglyMeasurable (by fun_prop)).mul ?_
    exact (Complex.measurable_ofReal.comp (measurable_stepInd x L)).aestronglyMeasurable
  have hint : Integrable (Function.uncurry fun (u y : ℝ) => Complex.exp (c * (u : ℂ)) * (stepInd x L (u, y) : ℂ))
      (volume.prod K) := by
    refine (integrable_stepInd_volume_prod K x L).mono' hmeas
      (Filter.Eventually.of_forall fun p => ?_)
    show ‖Complex.exp (c * (p.1 : ℂ)) * (stepInd x L p : ℂ)‖ ≤ stepInd x L p
    rw [norm_mul, hc, norm_exp_mul_I_mul, one_mul, Complex.norm_real]
    exact le_of_eq (Real.norm_of_nonneg (stepInd_nonneg x L p))
  have hinner : ∀ y : ℝ, ∫ u : ℝ, Complex.exp (c * (u : ℂ)) * (stepInd x L (u, y) : ℂ)
      = Complex.exp (↑(-(2 * π * ξ * x)) * Complex.I)
        * (Complex.exp (↑(2 * π * ξ * L) * Complex.I) - 1) / (↑(2 * π * ξ) * Complex.I)
        * Complex.exp (↑(2 * π * ξ) * ↑y * Complex.I) := by
    intro y
    have hpt : ∀ u : ℝ, Complex.exp (c * (u : ℂ)) * (stepInd x L (u, y) : ℂ)
        = (Set.Ioc (x - L - y) (x - y)).indicator (fun u : ℝ => Complex.exp (c * (u : ℂ))) u := by
      intro u
      rw [stepInd_swap]
      show Complex.exp (c * (u : ℂ))
        * (((Set.Ioc (x - L - y) (x - y)).indicator (fun _ => (1 : ℝ)) u : ℝ) : ℂ) = _
      by_cases h : u ∈ Set.Ioc (x - L - y) (x - y)
      · rw [Set.indicator_of_mem h, Set.indicator_of_mem h]; push_cast; ring
      · rw [Set.indicator_of_notMem h, Set.indicator_of_notMem h]; push_cast; ring
    simp_rw [hpt]
    rw [integral_indicator measurableSet_Ioc, ← intervalIntegral.integral_of_le (by linarith),
      integral_exp_mul_complex hc0]
    have e1 : Complex.exp (c * ↑(x - y)) = Complex.exp (↑(-(2 * π * ξ * x)) * Complex.I)
        * Complex.exp (↑(2 * π * ξ) * ↑y * Complex.I) := by
      rw [← Complex.exp_add]; congr 1; rw [hc]; push_cast; ring
    have e2 : Complex.exp (c * ↑(x - L - y)) = Complex.exp (↑(-(2 * π * ξ * x)) * Complex.I)
        * Complex.exp (↑(2 * π * ξ * L) * Complex.I)
        * Complex.exp (↑(2 * π * ξ) * ↑y * Complex.I) := by
      rw [← Complex.exp_add, ← Complex.exp_add]; congr 1; rw [hc]; push_cast; ring
    rw [e1, e2, hc]
    push_cast
    field_simp
    ring
  calc 𝓕 (fun u => (smoothStep K x L u : ℂ)) ξ
      = ∫ u : ℝ, Complex.exp (c * (u : ℂ)) * ∫ y, (stepInd x L (u, y) : ℂ) ∂K := by
        rw [Real.fourier_eq']
        refine integral_congr_ae (Filter.Eventually.of_forall fun u => ?_)
        simp only [smul_eq_mul, hexp]
        rw [smoothStep_eq_integral, integral_complex_ofReal]
    _ = ∫ u : ℝ, ∫ y, Complex.exp (c * (u : ℂ)) * (stepInd x L (u, y) : ℂ) ∂K := by
        refine integral_congr_ae (Filter.Eventually.of_forall fun u => ?_)
        show Complex.exp (c * (u : ℂ)) * ∫ y, (stepInd x L (u, y) : ℂ) ∂K
          = ∫ y, Complex.exp (c * (u : ℂ)) * (stepInd x L (u, y) : ℂ) ∂K
        rw [integral_const_mul]
    _ = ∫ y, (∫ u : ℝ, Complex.exp (c * (u : ℂ)) * (stepInd x L (u, y) : ℂ)) ∂K :=
        integral_integral_swap (f := fun (u y : ℝ) => Complex.exp (c * (u : ℂ)) * (stepInd x L (u, y) : ℂ))
          hint
    _ = _ := by
        simp_rw [hinner]
        rw [integral_const_mul, charFun_apply_real]

/-- The Fourier transform of the smooth step decays like `|φ_K(2πξ)|/(π|ξ|)`. -/
theorem norm_fourier_smoothStep_le (K : Measure ℝ) [IsProbabilityMeasure K] (x : ℝ) {L : ℝ}
    (hL : 0 ≤ L) {ξ : ℝ} (hξ : ξ ≠ 0) :
    ‖𝓕 (fun u => (smoothStep K x L u : ℂ)) ξ‖ ≤ ‖charFun K (2 * π * ξ)‖ / (π * |ξ|) := by
  rw [fourier_smoothStep K x hL hξ, norm_mul, norm_div, norm_mul, Complex.norm_exp_ofReal_mul_I,
    one_mul, norm_mul, Complex.norm_I, mul_one, Complex.norm_real, Real.norm_eq_abs, abs_mul,
    abs_of_pos (by positivity : (0 : ℝ) < 2 * π)]
  have hpos : 0 < π * |ξ| := mul_pos Real.pi_pos (abs_pos.mpr hξ)
  have h2 : ‖Complex.exp (↑(2 * π * ξ * L) * Complex.I) - 1‖ ≤ 2 := by
    refine (norm_sub_le _ _).trans ?_
    rw [Complex.norm_exp_ofReal_mul_I, norm_one]; try norm_num
  calc ‖Complex.exp (↑(2 * π * ξ * L) * Complex.I) - 1‖ / (2 * π * |ξ|) * ‖charFun K (2 * π * ξ)‖
      ≤ 2 / (2 * π * |ξ|) * ‖charFun K (2 * π * ξ)‖ := by gcongr
    _ = ‖charFun K (2 * π * ξ)‖ / (π * |ξ|) := by field_simp; try ring

/-- The Fourier transform of the smooth step is bounded by its mass `L`. -/
theorem norm_fourier_smoothStep_le_mass (K : Measure ℝ) [IsProbabilityMeasure K] (x : ℝ) {L : ℝ}
    (hL : 0 ≤ L) (ξ : ℝ) : ‖𝓕 (fun u => (smoothStep K x L u : ℂ)) ξ‖ ≤ L := by
  refine (VectorFourier.norm_fourierIntegral_le_integral_norm _ _ _ _ _).trans (le_of_eq ?_)
  simp_rw [Complex.norm_real, Real.norm_of_nonneg (smoothStep_nonneg K x L _)]
  exact integral_smoothStep K x hL

theorem integrable_fourier_smoothStep (K : Measure ℝ) [IsProbabilityMeasure K] (x : ℝ) {L : ℝ}
    (hL : 0 ≤ L) (hKf : Integrable (fun ξ : ℝ => min 1 (‖charFun K (2 * π * ξ)‖ / |ξ|))) :
    Integrable (𝓕 (fun u => (smoothStep K x L u : ℂ))) := by
  have hgi : Integrable (fun u => (smoothStep K x L u : ℂ)) := (integrable_smoothStep K x hL).ofReal
  have hcont : Continuous (𝓕 (fun u => (smoothStep K x L u : ℂ))) :=
    VectorFourier.fourierIntegral_continuous Real.continuous_fourierChar (innerSL ℝ).continuous₂ hgi
  refine (hKf.const_mul (max L 1)).mono' hcont.aestronglyMeasurable ?_
  filter_upwards [Measure.ae_ne volume (0 : ℝ)] with ξ hξ
  have h1 := norm_fourier_smoothStep_le_mass K x hL ξ
  have h2 := norm_fourier_smoothStep_le K x hL hξ
  have h3 : ‖charFun K (2 * π * ξ)‖ / (π * |ξ|) ≤ ‖charFun K (2 * π * ξ)‖ / |ξ| := by
    apply div_le_div_of_nonneg_left (norm_nonneg _) (abs_pos.mpr hξ)
    exact le_mul_of_one_le_left (abs_nonneg _) (by linarith [Real.pi_gt_three])
  have hh : 0 ≤ ‖charFun K (2 * π * ξ)‖ / |ξ| := by positivity
  rcases le_or_gt (‖charFun K (2 * π * ξ)‖ / |ξ|) 1 with hh1 | hh1
  · rw [min_eq_right hh1]
    exact (h2.trans h3).trans (le_mul_of_one_le_left hh (le_max_right _ _))
  · rw [min_eq_left hh1.le, mul_one]
    exact h1.trans (le_max_left _ _)

/-- **The smoothed difference is bounded by the weighted characteristic-function gap.** -/
theorem abs_smoothDiff_le (μ ν K : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    [IsProbabilityMeasure K] [NullSingletonClass K]
    (hKf : Integrable (fun ξ : ℝ => min 1 (‖charFun K (2 * π * ξ)‖ / |ξ|)))
    (hI : Integrable (fun ξ : ℝ => ‖charFun K (2 * π * ξ)‖ / (π * |ξ|)
      * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖)) (x : ℝ) :
    |smoothDiff μ ν K x| ≤ ∫ ξ, ‖charFun K (2 * π * ξ)‖ / (π * |ξ|)
      * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖ := by
  set I := ∫ ξ, ‖charFun K (2 * π * ξ)‖ / (π * |ξ|)
    * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖ with hI_def
  have hΔc : Continuous fun ξ : ℝ => ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖ := by
    have h2 : Continuous fun ξ : ℝ => 2 * π * ξ := by fun_prop
    exact ((continuous_charFun.comp h2).sub (continuous_charFun.comp h2)).norm
  have hΔb : ∀ ξ : ℝ, ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖ ≤ 2 := by
    intro ξ
    have hμ : ‖charFun μ (2 * π * ξ)‖ ≤ 1 := norm_charFun_le_one _
    have hν : ‖charFun ν (2 * π * ξ)‖ ≤ 1 := norm_charFun_le_one _
    exact (norm_sub_le _ _).trans (by linarith)
  have key : ∀ L, 0 ≤ L → |smoothDiff μ ν K x| ≤ I + |smoothDiff μ ν K (x - L)| := by
    intro L hL
    have hg : Continuous (fun u => (smoothStep K x L u : ℂ)) :=
      Complex.continuous_ofReal.comp (continuous_smoothStep K x hL)
    have hgi : Integrable (fun u => (smoothStep K x L u : ℂ)) :=
      (integrable_smoothStep K x hL).ofReal
    have hgf : Integrable (𝓕 (fun u => (smoothStep K x L u : ℂ))) :=
      integrable_fourier_smoothStep K x hL hKf
    have h1 := norm_integral_sub_integral_le_fourier hg hgi hgf μ ν
    rw [integral_complex_ofReal, integral_complex_ofReal, ← Complex.ofReal_sub, Complex.norm_real,
      Real.norm_eq_abs, ← smoothDiff_sub_smoothDiff μ ν K x hL] at h1
    have h2 : ∫ ξ, ‖𝓕 (fun u => (smoothStep K x L u : ℂ)) ξ‖
        * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖ ≤ I := by
      refine integral_mono_ae ?_ hI ?_
      · refine (hgf.norm.mul_const 2).mono' (hgf.norm.aestronglyMeasurable.mul
          hΔc.aestronglyMeasurable) (Filter.Eventually.of_forall fun ξ => ?_)
        rw [norm_mul, norm_norm, norm_norm]
        exact mul_le_mul_of_nonneg_left (hΔb ξ) (norm_nonneg _)
      · filter_upwards [Measure.ae_ne volume (0 : ℝ)] with ξ hξ
        exact mul_le_mul_of_nonneg_right (norm_fourier_smoothStep_le K x hL hξ) (norm_nonneg _)
    calc |smoothDiff μ ν K x|
        = |(smoothDiff μ ν K x - smoothDiff μ ν K (x - L)) + smoothDiff μ ν K (x - L)| := by
          ring_nf
      _ ≤ |smoothDiff μ ν K x - smoothDiff μ ν K (x - L)| + |smoothDiff μ ν K (x - L)| :=
          abs_add_le _ _
      _ ≤ I + |smoothDiff μ ν K (x - L)| := by gcongr; exact h1.trans h2
  have hlim : Tendsto (fun L => I + |smoothDiff μ ν K (x - L)|) atTop (𝓝 (I + |0|)) :=
    tendsto_const_nhds.add (tendsto_smoothDiff_atTop μ ν K x).abs
  rw [abs_zero, add_zero] at hlim
  exact ge_of_tendsto hlim (Filter.eventually_atTop.mpr ⟨0, fun L hL => key L hL⟩)

/-- **Esseen's smoothing inequality** (kernel form): for probability measures `μ, ν, K` on `ℝ`
with `K` atomless, `F_ν` `m`-Lipschitz and `K` putting mass at most `ε` outside `[−a, a]`,
`|F_μ(x₀) − F_ν(x₀)| (1 − ε) ≤ 2 ∫ |φ_K(2πξ)|/(π|ξ|) · |φ_μ(2πξ) − φ_ν(2πξ)| dξ + 2ma + ε`. -/
theorem esseen_inequality (μ ν K : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    [IsProbabilityMeasure K] [NullSingletonClass K] {m : ℝ} (hm : 0 ≤ m)
    (hG : ∀ x y, x ≤ y → cdfR ν y - cdfR ν x ≤ m * (y - x)) {a ε : ℝ} (ha : 0 ≤ a)
    (hε : K.real {y | a < ‖y‖} ≤ ε)
    (hKf : Integrable (fun ξ : ℝ => min 1 (‖charFun K (2 * π * ξ)‖ / |ξ|)))
    (hI : Integrable (fun ξ : ℝ => ‖charFun K (2 * π * ξ)‖ / (π * |ξ|)
      * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖)) (x₀ : ℝ) :
    |cdfR μ x₀ - cdfR ν x₀| * (1 - ε)
      ≤ 2 * (∫ ξ, ‖charFun K (2 * π * ξ)‖ / (π * |ξ|)
          * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖) + 2 * m * a + ε := by
  have h := esseen_lemma μ ν K hm hG ha hε x₀
  have h1 := abs_smoothDiff_le μ ν K hKf hI (x₀ + a)
  have h2 := abs_smoothDiff_le μ ν K hKf hI (x₀ - a)
  linarith

end EsseenFourier

section EsseenGaussian

theorem norm_charFun_gaussianReal (v : NNReal) (t : ℝ) :
    ‖charFun (gaussianReal 0 v) t‖ = Real.exp (-(v * t ^ 2 / 2)) := by
  rw [charFun_gaussianReal, Complex.norm_exp]
  congr 1
  rw [show (↑t * ↑(0 : ℝ) * Complex.I - ↑↑v * ↑t ^ 2 / 2 : ℂ)
      = ((-((v : ℝ) * t ^ 2 / 2) : ℝ) : ℂ) by push_cast; ring, Complex.ofReal_re]

/-- The Gaussian kernel is atomless. -/
theorem nullSingletonClass_gaussianReal {v : NNReal} (hv : v ≠ 0) :
    NullSingletonClass (gaussianReal 0 v) := by
  rw [gaussianReal_of_var_ne_zero _ hv]; infer_instance

/-- Chebyshev tail of the Gaussian kernel: `K{|y| > a} ≤ v/a²`. -/
theorem gaussianReal_real_tail_le (v : NNReal) {a : ℝ} (ha : 0 < a) :
    (gaussianReal 0 v).real {y | a < ‖y‖} ≤ (v : ℝ) / a ^ 2 := by
  have h := meas_ge_le_variance_div_sq (μ := gaussianReal 0 v) (memLp_id_gaussianReal 2) ha
  rw [variance_id_gaussianReal] at h
  simp only [id_eq, integral_id_gaussianReal, sub_zero] at h
  have hsub : {y : ℝ | a < ‖y‖} ⊆ {y : ℝ | a ≤ |y|} := fun y hy => by
    simp only [Set.mem_setOf_eq, Real.norm_eq_abs] at hy ⊢; exact hy.le
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) ?_
  exact (measure_mono hsub).trans h

/-- The smoothing weight `min(1, |φ_K(2πξ)|/|ξ|)` is integrable for the Gaussian kernel. -/
theorem integrable_gaussian_weight {v : NNReal} (hv : v ≠ 0) :
    Integrable (fun ξ : ℝ => min 1 (‖charFun (gaussianReal 0 v) (2 * π * ξ)‖ / |ξ|)) := by
  have hv0 : (0 : ℝ) < v := NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr hv)
  have hb : (0 : ℝ) < 2 * π ^ 2 * v := by positivity
  have hmeas : Measurable fun ξ : ℝ => min 1 (‖charFun (gaussianReal 0 v) (2 * π * ξ)‖ / |ξ|) := by
    refine measurable_const.min ?_
    have h2 : Continuous fun ξ : ℝ => 2 * π * ξ := by fun_prop
    exact (continuous_charFun.comp h2).norm.measurable.div continuous_abs.measurable
  have hdom : Integrable (fun ξ : ℝ => (Set.Icc (-1 : ℝ) 1).indicator (fun _ => (1 : ℝ)) ξ
      + Real.exp (-(2 * π ^ 2 * v) * ξ ^ 2)) :=
    ((integrableOn_const (by rw [Real.volume_Icc]; exact ENNReal.ofReal_ne_top)).integrable_indicator
      measurableSet_Icc).add (integrable_exp_neg_mul_sq hb)
  refine hdom.mono' hmeas.aestronglyMeasurable (Filter.Eventually.of_forall fun ξ => ?_)
  have hh : 0 ≤ ‖charFun (gaussianReal 0 v) (2 * π * ξ)‖ / |ξ| := by positivity
  rw [Real.norm_of_nonneg (le_min zero_le_one hh)]
  have hexp : ‖charFun (gaussianReal 0 v) (2 * π * ξ)‖ = Real.exp (-(2 * π ^ 2 * v) * ξ ^ 2) := by
    rw [norm_charFun_gaussianReal]; congr 1; ring
  rcases le_or_gt |ξ| 1 with hξ | hξ
  · have hmem : ξ ∈ Set.Icc (-1 : ℝ) 1 := abs_le.mp hξ
    rw [Set.indicator_of_mem hmem]
    exact (min_le_left _ _).trans (le_add_of_nonneg_right (Real.exp_nonneg _))
  · have hmem : ξ ∉ Set.Icc (-1 : ℝ) 1 := fun h => absurd (abs_le.mpr h) (not_le.mpr hξ)
    rw [Set.indicator_of_notMem hmem, zero_add, ← hexp]
    exact (min_le_right _ _).trans (div_le_self (norm_nonneg _) hξ.le)

/-- **Esseen's inequality with a Gaussian kernel** `N(0, v)`: for probability measures `μ, ν`
on `ℝ` with `F_ν` `m`-Lipschitz, every `a > 0` and `v ≠ 0`,
`|F_μ(x₀) − F_ν(x₀)| (1 − v/a²) ≤ 2 ∫ e^{−v(2πξ)²/2}/(π|ξ|) · |φ_μ(2πξ) − φ_ν(2πξ)| dξ + 2ma + v/a²`.
-/
theorem esseen_inequality_gaussian (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] {m : ℝ} (hm : 0 ≤ m)
    (hG : ∀ x y, x ≤ y → cdfR ν y - cdfR ν x ≤ m * (y - x)) {v : NNReal} (hv : v ≠ 0)
    {a : ℝ} (ha : 0 < a)
    (hI : Integrable (fun ξ : ℝ => Real.exp (-(v * (2 * π * ξ) ^ 2 / 2)) / (π * |ξ|)
      * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖)) (x₀ : ℝ) :
    |cdfR μ x₀ - cdfR ν x₀| * (1 - v / a ^ 2)
      ≤ 2 * (∫ ξ, Real.exp (-(v * (2 * π * ξ) ^ 2 / 2)) / (π * |ξ|)
          * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖) + 2 * m * a + v / a ^ 2 := by
  haveI := nullSingletonClass_gaussianReal hv
  have h := esseen_inequality μ ν (gaussianReal 0 v) hm hG ha.le (gaussianReal_real_tail_le v ha)
    (integrable_gaussian_weight hv) (by simpa only [norm_charFun_gaussianReal] using hI) x₀
  simpa only [norm_charFun_gaussianReal] using h

end EsseenGaussian

