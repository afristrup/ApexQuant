/-!
# §43  The Fejér kernel: Esseen's inequality band-limited to `|t| < 2πT`

§42 proves Esseen's inequality for an arbitrary atomless kernel `K`.  A rate for a lattice law
such as the coordinate law of `H D_ε v` needs a kernel whose characteristic function vanishes
beyond a cutoff, so that only the range where the §40 bound holds enters.  This section builds
that kernel and sharpens the lemma so that the tail term scales with the gap itself.

* `esseen_lemma'`, `esseen_sup`: Esseen's lemma with the tail term `η ε`, `η` a bound on the
  distribution-function gap, and its supremum form
  `|F_μ(x₀) − F_ν(x₀)| (1 − 2ε) ≤ 2I + 2ma` with `I` a bound on the smoothed gap.  The
  supremum `⨆ x, |F_μ x − F_ν x|` replaces the crude bound `1` used in §42.
* `tri T ξ = (1 − |ξ|/T)₊` is the triangle; its Fourier transform is computed exactly by the
  fundamental theorem of calculus with the antiderivative `(α + βξ) e^{cξ}`:
  `𝓕 tri_T (y) = (1 − cos(2πTy))/(2π²Ty²)` for `y ≠ 0` (`fourier_triC`).
* `fejerDensity T y = ‖𝓕 tri_T(y)‖` is nonnegative, continuous, at most `2T`, at most
  `1/(π²Ty²)`, integrable, and integrates to `1` by Fourier inversion at the origin.
* `fejer T` is the probability measure with that density: atomless, with characteristic
  function the triangle `tri T (t/(2π))` (Fourier inversion, `charFun_fejer`), tail
  `K_T{|y| > a} ≤ 2/(π²Ta)` (`fejer_real_tail_le`), and integrable smoothing weight.
* `esseen_inequality_fejer`: the assembled inequality
  `|F_μ(x₀) − F_ν(x₀)| (1 − 4/(π²Ta)) ≤ 2 ∫ (1 − |ξ|/T)₊/(π|ξ|) · |φ_μ(2πξ) − φ_ν(2πξ)| dξ + 2ma`,
  in which only `|2πξ| < 2πT` enters.

What this does **not** yet give is the Kolmogorov–Smirnov rate for the coordinate law of
`H D_ε v`.  Feeding the §40 bound `t⁴ ∑ⱼ vⱼ⁴/(3d²)` into the integral with `2πT = √d` gives a
constant, not a rate: the §40 bound has no Gaussian damping and its range `|t| ≤ √d` is the
range `|t·max_j |vⱼ|| ≤ 1` only for the worst direction.  A rate needs the damped form
`|φ(t) − e^{−σ²t²/2}| ≤ C t⁴ ∑ⱼ aⱼ⁴ e^{−σ²t²/4}` on `|t aⱼ| ≤ 1`, which is not proved here.
-/

open Filter
open scoped Topology FourierTransform

section EsseenSup

theorem esseen_lemma' (μ ν K : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    [IsProbabilityMeasure K] {m : ℝ} (hm : 0 ≤ m)
    (hG : ∀ x y, x ≤ y → cdfR ν y - cdfR ν x ≤ m * (y - x)) {a ε : ℝ} (ha : 0 ≤ a)
    (hε : K.real {y | a < ‖y‖} ≤ ε) {η : ℝ} (hη : ∀ z, |cdfR μ z - cdfR ν z| ≤ η) (x₀ : ℝ) :
    |cdfR μ x₀ - cdfR ν x₀| * (1 - ε)
      ≤ |smoothDiff μ ν K (x₀ + a)| + |smoothDiff μ ν K (x₀ - a)| + 2 * m * a + η * ε := by
  set Δ : ℝ → ℝ := fun z => cdfR μ z - cdfR ν z with hΔ
  have hΔm : Measurable Δ := (measurable_cdfR μ).sub (measurable_cdfR ν)
  have hΔb : ∀ z, |Δ z| ≤ η := hη
  have hη0 : 0 ≤ η := (abs_nonneg _).trans (hη 0)
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
    Integrable.of_bound (hΔm.comp (measurable_const.sub measurable_id)).aestronglyMeasurable η
      (Filter.Eventually.of_forall fun y => by rw [Real.norm_eq_abs]; exact hΔb _)
  have hstep : ∀ c : ℝ, Integrable (fun y : ℝ => {y : ℝ | ‖y‖ ≤ a}.indicator (fun _ => c) y
      + {y : ℝ | a < ‖y‖}.indicator (fun _ => -η) y) K :=
    fun c => ((integrable_const c).indicator (measurableSet_le measurable_norm measurable_const)).add
      ((integrable_const _).indicator hset)
  have hstep_int : ∀ c : ℝ, ∫ y, ({y : ℝ | ‖y‖ ≤ a}.indicator (fun _ => c) y
      + {y : ℝ | a < ‖y‖}.indicator (fun _ => -η) y) ∂K = c * (1 - ε') - η * ε' := by
    intro c
    rw [integral_add ((integrable_const c).indicator (measurableSet_le measurable_norm measurable_const))
      ((integrable_const _).indicator hset), integral_indicator_const _
      (measurableSet_le measurable_norm measurable_const), integral_indicator_const _ hset, hcompl]
    simp only [smul_eq_mul]
    ring
  rcases le_or_gt 1 ε with hε1' | hε1'
  · -- trivial: the left side is nonpositive
    have : |Δ x₀| * (1 - ε) ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (abs_nonneg _) (by linarith)
    have h0 : 0 ≤ |smoothDiff μ ν K (x₀ + a)| + |smoothDiff μ ν K (x₀ - a)| + 2 * m * a + η * ε := by
      have := abs_nonneg (smoothDiff μ ν K (x₀ + a)); have := abs_nonneg (smoothDiff μ ν K (x₀ - a))
      have := mul_nonneg hm ha; have := mul_nonneg hη0 (hε0.trans hε); nlinarith
    linarith
  rcases le_or_gt 0 (Δ x₀) with hD | hD
  · -- `Δ x₀ ≥ 0`: look to the right
    have hpt : ∀ y, {y : ℝ | ‖y‖ ≤ a}.indicator (fun _ => Δ x₀ - 2 * m * a) y
        + {y : ℝ | a < ‖y‖}.indicator (fun _ => -η) y ≤ Δ (x₀ + a - y) := by
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
    have hle : Δ x₀ * (1 - ε') - 2 * m * a * (1 - ε') - η * ε' ≤ smoothDiff μ ν K (x₀ + a) := by
      have := integral_mono (hstep _) (hint (x₀ + a)) hpt
      rw [hstep_int] at this
      unfold smoothDiff
      linarith
    have habs : smoothDiff μ ν K (x₀ + a) ≤ |smoothDiff μ ν K (x₀ + a)| := le_abs_self _
    have h1 : |Δ x₀| = Δ x₀ := abs_of_nonneg hD
    rw [h1]
    have h2 : Δ x₀ * (1 - ε) ≤ Δ x₀ * (1 - ε') := mul_le_mul_of_nonneg_left (by linarith) hD
    have h3 : 0 ≤ |smoothDiff μ ν K (x₀ - a)| := abs_nonneg _
    nlinarith [mul_nonneg hm ha, mul_nonneg (mul_nonneg hm ha) hε0, mul_le_mul_of_nonneg_left hε hη0]
  · -- `Δ x₀ < 0`: look to the left
    have hpt : ∀ y, Δ (x₀ - a - y) ≤ {y : ℝ | ‖y‖ ≤ a}.indicator (fun _ => Δ x₀ + 2 * m * a) y
        + {y : ℝ | a < ‖y‖}.indicator (fun _ => η) y := by
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
        + {y : ℝ | a < ‖y‖}.indicator (fun _ => η) y) K :=
      ((integrable_const _).indicator (measurableSet_le measurable_norm measurable_const)).add
        ((integrable_const _).indicator hset)
    have hle : smoothDiff μ ν K (x₀ - a) ≤ (Δ x₀ + 2 * m * a) * (1 - ε') + η * ε' := by
      have := integral_mono (hint (x₀ - a)) hstep' hpt
      rw [integral_add ((integrable_const _).indicator (measurableSet_le measurable_norm measurable_const))
        ((integrable_const _).indicator hset), integral_indicator_const _
        (measurableSet_le measurable_norm measurable_const), integral_indicator_const _ hset,
        hcompl] at this
      simp only [smul_eq_mul] at this
      unfold smoothDiff
      linarith
    have habs : -|smoothDiff μ ν K (x₀ - a)| ≤ smoothDiff μ ν K (x₀ - a) := neg_abs_le _
    have h1 : |Δ x₀| = -Δ x₀ := abs_of_neg hD
    rw [h1]
    have h2 : -Δ x₀ * (1 - ε) ≤ -Δ x₀ * (1 - ε') :=
      mul_le_mul_of_nonneg_left (by linarith) (by linarith)
    have h3 : 0 ≤ |smoothDiff μ ν K (x₀ + a)| := abs_nonneg _
    nlinarith [mul_nonneg hm ha, mul_nonneg (mul_nonneg hm ha) hε0, mul_le_mul_of_nonneg_left hε hη0]


/-- **Esseen's lemma, supremum form**: with `I` a bound on the smoothed gap and `ε` the kernel tail,
`|F_μ(x₀) − F_ν(x₀)| (1 − 2ε) ≤ 2I + 2ma` for every `x₀`. -/
theorem esseen_sup (μ ν K : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    [IsProbabilityMeasure K] {m : ℝ} (hm : 0 ≤ m)
    (hG : ∀ x y, x ≤ y → cdfR ν y - cdfR ν x ≤ m * (y - x)) {a ε : ℝ} (ha : 0 ≤ a)
    (hε : K.real {y | a < ‖y‖} ≤ ε) {I : ℝ} (hD : ∀ x, |smoothDiff μ ν K x| ≤ I) (x₀ : ℝ) :
    |cdfR μ x₀ - cdfR ν x₀| * (1 - 2 * ε) ≤ 2 * I + 2 * m * a := by
  have hI : 0 ≤ I := (abs_nonneg _).trans (hD 0)
  have hma : 0 ≤ m * a := mul_nonneg hm ha
  rcases le_or_gt (1 - 2 * ε) 0 with hneg | hpos
  · have : |cdfR μ x₀ - cdfR ν x₀| * (1 - 2 * ε) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (abs_nonneg _) hneg
    linarith
  have hε1 : 0 < 1 - ε := by linarith
  set Δ : ℝ → ℝ := fun z => |cdfR μ z - cdfR ν z| with hΔ
  have hΔb : ∀ z, Δ z ≤ 1 := fun z => by
    rw [hΔ]; simp only
    rw [abs_le]
    have := cdfR_nonneg μ z; have := cdfR_le_one μ z
    have := cdfR_nonneg ν z; have := cdfR_le_one ν z
    constructor <;> linarith
  have hbdd : BddAbove (Set.range Δ) := ⟨1, by rintro _ ⟨z, rfl⟩; exact hΔb z⟩
  set η := ⨆ z, Δ z with hη
  have hle : ∀ z, Δ z ≤ η := fun z => le_ciSup hbdd z
  have hpt : ∀ x, Δ x ≤ (2 * I + 2 * m * a + η * ε) / (1 - ε) := by
    intro x
    rw [le_div_iff₀ hε1]
    have h := esseen_lemma' μ ν K hm hG ha hε hle x
    have h1 := hD (x + a); have h2 := hD (x - a)
    linarith
  have hsup : η ≤ (2 * I + 2 * m * a + η * ε) / (1 - ε) := ciSup_le hpt
  rw [le_div_iff₀ hε1] at hsup
  have hη' : η * (1 - 2 * ε) ≤ 2 * I + 2 * m * a := by linarith
  calc |cdfR μ x₀ - cdfR ν x₀| * (1 - 2 * ε) ≤ η * (1 - 2 * ε) :=
        mul_le_mul_of_nonneg_right (hle x₀) hpos.le
    _ ≤ 2 * I + 2 * m * a := hη'

end EsseenSup


section Fejer

/-- The triangle function of half-width `T`: `(1 − |ξ|/T)₊`. -/
noncomputable def tri (T ξ : ℝ) : ℝ := max 0 (1 - |ξ| / T)

theorem tri_nonneg (T ξ : ℝ) : 0 ≤ tri T ξ := le_max_left _ _

theorem tri_le_one {T : ℝ} (hT : 0 < T) (ξ : ℝ) : tri T ξ ≤ 1 :=
  max_le zero_le_one (by have := div_nonneg (abs_nonneg ξ) hT.le; linarith)

theorem tri_eq_zero {T : ℝ} (hT : 0 < T) {ξ : ℝ} (h : T ≤ |ξ|) : tri T ξ = 0 :=
  max_eq_left (by rw [sub_nonpos, le_div_iff₀ hT, one_mul]; exact h)

theorem tri_of_abs_le {T : ℝ} (hT : 0 < T) {ξ : ℝ} (h : |ξ| ≤ T) : tri T ξ = 1 - |ξ| / T :=
  max_eq_right (by rw [sub_nonneg, div_le_one hT]; exact h)

theorem tri_zero {T : ℝ} (hT : 0 < T) : tri T 0 = 1 := by
  rw [tri_of_abs_le hT (by rw [abs_zero]; exact hT.le), abs_zero, zero_div, sub_zero]

theorem continuous_tri (T : ℝ) : Continuous (tri T) :=
  continuous_const.max (continuous_const.sub (continuous_abs.div_const T))

theorem support_tri_subset {T : ℝ} (hT : 0 < T) :
    Function.support (tri T) ⊆ Set.Icc (-T) T := fun ξ hξ => by
  rw [Function.mem_support] at hξ
  by_contra h
  rw [Set.mem_Icc, ← abs_le, not_le] at h
  exact hξ (tri_eq_zero hT h.le)

theorem hasCompactSupport_tri {T : ℝ} (hT : 0 < T) : HasCompactSupport (tri T) :=
  HasCompactSupport.of_support_subset_isCompact isCompact_Icc (support_tri_subset hT)

/-- The triangle as a complex-valued function. -/
noncomputable def triC (T ξ : ℝ) : ℂ := (tri T ξ : ℂ)

theorem continuous_triC (T : ℝ) : Continuous (triC T) :=
  Complex.continuous_ofReal.comp (continuous_tri T)

theorem hasCompactSupport_triC {T : ℝ} (hT : 0 < T) : HasCompactSupport (triC T) :=
  (hasCompactSupport_tri hT).comp_left Complex.ofReal_zero

theorem integrable_triC {T : ℝ} (hT : 0 < T) : Integrable (triC T) :=
  (continuous_triC T).integrable_of_hasCompactSupport (hasCompactSupport_triC hT)

theorem norm_triC_le {T : ℝ} (hT : 0 < T) (ξ : ℝ) :
    ‖triC T ξ‖ ≤ (Set.Icc (-T) T).indicator (fun _ => (1 : ℝ)) ξ := by
  rw [triC, Complex.norm_real, Real.norm_of_nonneg (tri_nonneg T ξ)]
  by_cases h : ξ ∈ Set.Icc (-T) T
  · rw [Set.indicator_of_mem h]; exact tri_le_one hT ξ
  · rw [Set.indicator_of_notMem h]
    rw [Set.mem_Icc, ← abs_le, not_le] at h
    rw [tri_eq_zero hT h.le]

/-- Derivative of `(α + βξ) e^{cξ}` along the real line. -/
theorem hasDerivAt_affine_mul_exp (α β c : ℂ) (ξ : ℝ) :
    HasDerivAt (fun ξ : ℝ => (α + β * ξ) * Complex.exp (c * ξ))
      ((β + (α + β * ξ) * c) * Complex.exp (c * ξ)) ξ := by
  have h1 : HasDerivAt (fun ξ : ℝ => α + β * (ξ : ℂ)) β ξ := by
    have := ((hasDerivAt_id ξ).ofReal_comp.const_mul β).const_add α
    simpa using this
  have h2 : HasDerivAt (fun ξ : ℝ => Complex.exp (c * ξ)) (Complex.exp (c * ξ) * c) ξ := by
    have := ((Complex.hasDerivAt_exp (c * ↑ξ)).comp (↑ξ : ℂ)
      ((hasDerivAt_id (↑ξ : ℂ)).const_mul c)).comp_ofReal
    simpa [Function.comp] using this
  exact (h1.mul h2).congr_deriv (by ring)

/-- **Fourier transform of the triangle**, `y ≠ 0`:
`𝓕 tri_T (y) = (1 − cos(2πTy)) / (2π²Ty²)`. -/
theorem fourier_triC {T : ℝ} (hT : 0 < T) {y : ℝ} (hy : y ≠ 0) :
    𝓕 (triC T) y = (((1 - Real.cos (2 * π * T * y)) / (2 * π ^ 2 * T * y ^ 2) : ℝ) : ℂ) := by
  set c : ℂ := ↑(-(2 * π * y)) * Complex.I with hc
  have h2πy : (2 * π * y : ℝ) ≠ 0 := by positivity
  have hc0 : c ≠ 0 := by
    rw [hc]
    refine mul_ne_zero ?_ Complex.I_ne_zero
    exact_mod_cast neg_ne_zero.mpr h2πy
  have hT0 : (T : ℂ) ≠ 0 := by exact_mod_cast hT.ne'
  have hexp : ∀ ξ : ℝ, Complex.exp (↑(-2 * π * inner ℝ ξ y) * Complex.I) = Complex.exp (c * ξ) := by
    intro ξ
    congr 1
    simp only [RCLike.inner_apply, conj_trivial, hc]
    push_cast
    ring
  have hF : Continuous fun ξ : ℝ => Complex.exp (c * ξ) * triC T ξ :=
    (by fun_prop : Continuous fun ξ : ℝ => Complex.exp (c * ξ)).mul (continuous_triC T)
  -- the two halves, by the fundamental theorem of calculus
  have hright : ∫ ξ in (0 : ℝ)..T, Complex.exp (c * ξ) * triC T ξ
      = Complex.exp (c * T) / (T * c ^ 2) - (1 / c + 1 / (T * c ^ 2)) := by
    have hcongr : Set.EqOn (fun ξ : ℝ => Complex.exp (c * ξ) * triC T ξ)
        (fun ξ : ℝ => ((1 / c + 1 / (T * c ^ 2)) + (-1 / (T * c)) * ξ) * Complex.exp (c * ξ)
          * (0 : ℂ) + (1 + (-1 / T) * ξ) * Complex.exp (c * ξ)) (Set.uIcc 0 T) := by
      intro ξ hξ
      rw [Set.uIcc_of_le hT.le, Set.mem_Icc] at hξ
      simp only [triC]
      rw [tri_of_abs_le hT (by rw [abs_of_nonneg hξ.1]; exact hξ.2), abs_of_nonneg hξ.1]
      push_cast
      ring
    rw [intervalIntegral.integral_congr hcongr]
    have hder : ∀ ξ ∈ Set.uIcc (0 : ℝ) T,
        HasDerivAt (fun ξ : ℝ => ((1 / c + 1 / (T * c ^ 2)) + (-1 / (T * c)) * ξ) * Complex.exp (c * ξ))
          (((1 / c + 1 / (T * c ^ 2)) + (-1 / (T * c)) * ξ) * Complex.exp (c * ξ) * (0 : ℂ)
            + (1 + (-1 / T) * ξ) * Complex.exp (c * ξ)) ξ := by
      intro ξ _
      refine (hasDerivAt_affine_mul_exp _ _ c ξ).congr_deriv ?_
      field_simp
      ring
    rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hder (by
      refine Continuous.intervalIntegrable ?_ _ _
      fun_prop)]
    push_cast
    simp only [mul_zero, Complex.exp_zero]
    field_simp
    ring
  have hleft : ∫ ξ in (-T : ℝ)..0, Complex.exp (c * ξ) * triC T ξ
      = (1 / c - 1 / (T * c ^ 2)) - (1 / c - 1 / (T * c ^ 2) - 1 / c) * Complex.exp (c * (-T : ℝ)) := by
    have hcongr : Set.EqOn (fun ξ : ℝ => Complex.exp (c * ξ) * triC T ξ)
        (fun ξ : ℝ => ((1 / c - 1 / (T * c ^ 2)) + (1 / (T * c)) * ξ) * Complex.exp (c * ξ)
          * (0 : ℂ) + (1 + (1 / T) * ξ) * Complex.exp (c * ξ)) (Set.uIcc (-T) 0) := by
      intro ξ hξ
      rw [Set.uIcc_of_le (by linarith), Set.mem_Icc] at hξ
      simp only [triC]
      rw [tri_of_abs_le hT (by rw [abs_of_nonpos hξ.2]; linarith), abs_of_nonpos hξ.2]
      push_cast
      ring
    rw [intervalIntegral.integral_congr hcongr]
    have hder : ∀ ξ ∈ Set.uIcc (-T : ℝ) 0,
        HasDerivAt (fun ξ : ℝ => ((1 / c - 1 / (T * c ^ 2)) + (1 / (T * c)) * ξ) * Complex.exp (c * ξ))
          (((1 / c - 1 / (T * c ^ 2)) + (1 / (T * c)) * ξ) * Complex.exp (c * ξ) * (0 : ℂ)
            + (1 + (1 / T) * ξ) * Complex.exp (c * ξ)) ξ := by
      intro ξ _
      refine (hasDerivAt_affine_mul_exp _ _ c ξ).congr_deriv ?_
      field_simp
      ring
    rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hder (by
      refine Continuous.intervalIntegrable ?_ _ _
      fun_prop)]
    push_cast
    simp only [mul_zero, Complex.exp_zero]
    field_simp
    ring
  have hsplit : 𝓕 (triC T) y = ∫ ξ in (-T : ℝ)..T, Complex.exp (c * ξ) * triC T ξ := by
    rw [Real.fourier_eq']
    simp_rw [hexp, smul_eq_mul]
    rw [← setIntegral_eq_integral_of_forall_compl_eq_zero (s := Set.Icc (-T) T) (fun ξ hξ => by
      have : tri T ξ = 0 := by
        rw [Set.mem_Icc, ← abs_le, not_le] at hξ
        exact tri_eq_zero hT hξ.le
      simp [triC, this]), integral_Icc_eq_integral_Ioc, ← intervalIntegral.integral_of_le (by linarith)]
  rw [hsplit, ← intervalIntegral.integral_add_adjacent_intervals (hF.intervalIntegrable _ _)
    (hF.intervalIntegrable _ _), hleft, hright]
  -- `e^{cT} + e^{−cT} = 2 cos(2πTy)`
  have hcos : Complex.exp (c * T) + Complex.exp (c * (-T : ℝ)) = 2 * Complex.cos (↑(2 * π * T * y)) := by
    rw [Complex.two_cos]
    have e1 : c * (T : ℂ) = -↑(2 * π * T * y) * Complex.I := by rw [hc]; push_cast; ring
    have e2 : c * ((-T : ℝ) : ℂ) = ↑(2 * π * T * y) * Complex.I := by rw [hc]; push_cast; ring
    rw [e1, e2, add_comm]
  have hc2 : c ^ 2 = -((4 * π ^ 2 * y ^ 2 : ℝ) : ℂ) := by
    rw [hc]; push_cast; ring_nf; rw [Complex.I_sq]; ring
  have hy2 : (4 * π ^ 2 * y ^ 2 : ℂ) ≠ 0 := by
    have : (4 * π ^ 2 * y ^ 2 : ℝ) ≠ 0 := by positivity
    exact_mod_cast this
  have hden : (2 * π ^ 2 * T * y ^ 2 : ℂ) ≠ 0 := by
    have : (2 * π ^ 2 * T * y ^ 2 : ℝ) ≠ 0 := by positivity
    exact_mod_cast this
  have key : (1 / c - 1 / (T * c ^ 2)) - (1 / c - 1 / (T * c ^ 2) - 1 / c) * Complex.exp (c * (-T : ℝ))
      + (Complex.exp (c * T) / (T * c ^ 2) - (1 / c + 1 / (T * c ^ 2)))
      = (Complex.exp (c * T) + Complex.exp (c * (-T : ℝ)) - 2) / (T * c ^ 2) := by
    field_simp
    ring
  rw [key, hcos, hc2]
  push_cast
  field_simp
  ring

/-- The Fejér density, `‖𝓕 tri_T(y)‖`. -/
noncomputable def fejerDensity (T y : ℝ) : ℝ := ‖𝓕 (triC T) y‖

theorem fejerDensity_nonneg (T y : ℝ) : 0 ≤ fejerDensity T y := norm_nonneg _

theorem continuous_fejerDensity {T : ℝ} (hT : 0 < T) : Continuous (fejerDensity T) :=
  (VectorFourier.fourierIntegral_continuous Real.continuous_fourierChar (innerSL ℝ).continuous₂
    (integrable_triC hT)).norm

theorem fejerDensity_eq {T : ℝ} (hT : 0 < T) {y : ℝ} (hy : y ≠ 0) :
    fejerDensity T y = (1 - Real.cos (2 * π * T * y)) / (2 * π ^ 2 * T * y ^ 2) := by
  rw [fejerDensity, fourier_triC hT hy, Complex.norm_real, Real.norm_of_nonneg]
  exact div_nonneg (by linarith [Real.cos_le_one (2 * π * T * y)]) (by positivity)

theorem ofReal_fejerDensity {T : ℝ} (hT : 0 < T) {y : ℝ} (hy : y ≠ 0) :
    ((fejerDensity T y : ℝ) : ℂ) = 𝓕 (triC T) y := by
  rw [fejerDensity_eq hT hy, fourier_triC hT hy]

/-- The Fejér density decays like `1/(π²T y²)`. -/
theorem fejerDensity_le_inv {T : ℝ} (hT : 0 < T) {y : ℝ} (hy : y ≠ 0) :
    fejerDensity T y ≤ 1 / (π ^ 2 * T) * (y ^ 2)⁻¹ := by
  rw [fejerDensity_eq hT hy]
  have h1 : 1 - Real.cos (2 * π * T * y) ≤ 2 := by linarith [Real.neg_one_le_cos (2 * π * T * y)]
  have hp : 0 < 2 * π ^ 2 * T * y ^ 2 := by positivity
  calc (1 - Real.cos (2 * π * T * y)) / (2 * π ^ 2 * T * y ^ 2) ≤ 2 / (2 * π ^ 2 * T * y ^ 2) :=
        div_le_div_of_nonneg_right h1 hp.le
    _ = 1 / (π ^ 2 * T) * (y ^ 2)⁻¹ := by field_simp

theorem fejerDensity_le_two_mul {T : ℝ} (hT : 0 < T) (y : ℝ) : fejerDensity T y ≤ 2 * T := by
  refine (VectorFourier.norm_fourierIntegral_le_integral_norm _ _ _ _ _).trans ?_
  calc ∫ ξ, ‖triC T ξ‖ ≤ ∫ ξ, (Set.Icc (-T) T).indicator (fun _ => (1 : ℝ)) ξ :=
        integral_mono (integrable_triC hT).norm ((integrableOn_const (by
          rw [Real.volume_Icc]; exact ENNReal.ofReal_ne_top)).integrable_indicator measurableSet_Icc)
          (norm_triC_le hT)
    _ = 2 * T := by
        rw [integral_indicator_const _ measurableSet_Icc, smul_eq_mul, mul_one, measureReal_def,
          Real.volume_Icc, ENNReal.toReal_ofReal (by linarith)]
        ring

theorem integrable_fejerDensity {T : ℝ} (hT : 0 < T) : Integrable (fejerDensity T) := by
  set C := max (4 * T) (2 / (π ^ 2 * T)) with hC
  refine (integrable_inv_one_add_sq.const_mul C).mono'
    (continuous_fejerDensity hT).aestronglyMeasurable (Filter.Eventually.of_forall fun y => ?_)
  rw [Real.norm_of_nonneg (fejerDensity_nonneg T y)]
  have h1y : 0 < 1 + y ^ 2 := by positivity
  rcases le_or_gt |y| 1 with hy | hy
  · have hy2 : y ^ 2 ≤ 1 := by rw [← sq_abs]; nlinarith [abs_nonneg y]
    calc fejerDensity T y ≤ 2 * T := fejerDensity_le_two_mul hT y
      _ ≤ 4 * T * (1 + y ^ 2)⁻¹ := by rw [le_mul_inv_iff₀ h1y]; nlinarith
      _ ≤ C * (1 + y ^ 2)⁻¹ := by gcongr; exact le_max_left _ _
  · have hy0 : y ≠ 0 := by intro h; rw [h, abs_zero] at hy; linarith
    have hy2 : 1 ≤ y ^ 2 := by rw [← sq_abs]; nlinarith
    have hπT : 0 < π ^ 2 * T := by positivity
    calc fejerDensity T y ≤ 1 / (π ^ 2 * T) * (y ^ 2)⁻¹ := fejerDensity_le_inv hT hy0
      _ ≤ 2 / (π ^ 2 * T) * (1 + y ^ 2)⁻¹ := by
          rw [show 1 / (π ^ 2 * T) * (y ^ 2)⁻¹ = 1 / (π ^ 2 * T * y ^ 2) by field_simp,
            show 2 / (π ^ 2 * T) * (1 + y ^ 2)⁻¹ = 2 / (π ^ 2 * T * (1 + y ^ 2)) by field_simp,
            div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith
      _ ≤ C * (1 + y ^ 2)⁻¹ := by gcongr; exact le_max_right _ _

theorem integrable_fourier_triC {T : ℝ} (hT : 0 < T) : Integrable (𝓕 (triC T)) := by
  refine (integrable_fejerDensity hT).ofReal.congr ?_
  filter_upwards [Measure.ae_ne volume (0 : ℝ)] with y hy
  exact ofReal_fejerDensity hT hy

/-- Fourier inversion for the triangle. -/
theorem fourierInv_fourier_triC {T : ℝ} (hT : 0 < T) : 𝓕⁻ (𝓕 (triC T)) = triC T :=
  (continuous_triC T).fourierInv_fourier_eq (integrable_triC hT) (integrable_fourier_triC hT)

/-- The Fejér density integrates to `1` (Fourier inversion at the origin). -/
theorem integral_fejerDensity {T : ℝ} (hT : 0 < T) : ∫ y, fejerDensity T y = 1 := by
  have h := congrFun (fourierInv_fourier_triC hT) 0
  rw [Real.fourierInv_eq'] at h
  simp only [inner_zero_right, mul_zero, Complex.ofReal_zero, zero_mul, Complex.exp_zero,
    one_smul] at h
  have h2 : ∫ v, 𝓕 (triC T) v = ((∫ y, fejerDensity T y : ℝ) : ℂ) := by
    rw [← integral_complex_ofReal]
    refine integral_congr_ae ?_
    filter_upwards [Measure.ae_ne volume (0 : ℝ)] with y hy
    exact (ofReal_fejerDensity hT hy).symm
  rw [h2, triC, tri_zero hT] at h
  exact_mod_cast h

/-- **The Fejér kernel** of half-width `T`: the probability measure with density
`(1 − cos(2πTy))/(2π²Ty²)`, whose characteristic function is the triangle `(1 − |t|/(2πT))₊`. -/
noncomputable def fejer (T : ℝ) : Measure ℝ :=
  volume.withDensity fun y => ((fejerDensity T y).toNNReal : ℝ≥0∞)

instance (T : ℝ) : NullSingletonClass (fejer T) := by unfold fejer; infer_instance

theorem integral_fejer {T : ℝ} (hT : 0 < T) (g : ℝ → ℂ) :
    ∫ y, g y ∂(fejer T) = ∫ y, ((fejerDensity T y : ℝ) : ℂ) * g y := by
  rw [fejer, integral_withDensity_eq_integral_smul
    (continuous_fejerDensity hT).measurable.real_toNNReal]
  refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
  show (fejerDensity T y).toNNReal • g y = _
  rw [NNReal.smul_def, Real.coe_toNNReal _ (fejerDensity_nonneg T y), Complex.real_smul]

theorem integral_fejer_eq_fourier {T : ℝ} (hT : 0 < T) (g : ℝ → ℂ) :
    ∫ y, g y ∂(fejer T) = ∫ y, 𝓕 (triC T) y * g y := by
  rw [integral_fejer hT]
  refine integral_congr_ae ?_
  filter_upwards [Measure.ae_ne volume (0 : ℝ)] with y hy
  rw [ofReal_fejerDensity hT hy]

/-- The characteristic function of the Fejér kernel is the triangle. -/
theorem charFun_fejer {T : ℝ} (hT : 0 < T) (t : ℝ) :
    charFun (fejer T) t = triC T (t / (2 * π)) := by
  have h := congrFun (fourierInv_fourier_triC hT) (t / (2 * π))
  rw [Real.fourierInv_eq'] at h
  rw [charFun_apply_real, integral_fejer_eq_fourier hT, ← h]
  refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
  simp only [smul_eq_mul]
  rw [mul_comm]
  congr 2
  simp only [RCLike.inner_apply, conj_trivial]
  push_cast
  field_simp
  try ring

theorem norm_charFun_fejer {T : ℝ} (hT : 0 < T) (ξ : ℝ) :
    ‖charFun (fejer T) (2 * π * ξ)‖ = tri T ξ := by
  rw [charFun_fejer hT, triC, Complex.norm_real, Real.norm_of_nonneg (tri_nonneg _ _),
    show 2 * π * ξ / (2 * π) = ξ by field_simp]

theorem fejer_univ {T : ℝ} (hT : 0 < T) : fejer T Set.univ = 1 := by
  rw [fejer, withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ]
  have : ∫⁻ y, ((fejerDensity T y).toNNReal : ℝ≥0∞) = ∫⁻ y, ENNReal.ofReal (fejerDensity T y) := rfl
  rw [this, ← ofReal_integral_eq_lintegral_ofReal (integrable_fejerDensity hT)
    (Filter.Eventually.of_forall (fejerDensity_nonneg T)), integral_fejerDensity hT,
    ENNReal.ofReal_one]

theorem isProbabilityMeasure_fejer {T : ℝ} (hT : 0 < T) : IsProbabilityMeasure (fejer T) :=
  ⟨fejer_univ hT⟩

theorem fejer_real_eq {T : ℝ} (hT : 0 < T) {s : Set ℝ} (hs : MeasurableSet s) :
    (fejer T).real s = ∫ y in s, fejerDensity T y := by
  rw [measureReal_def, fejer, withDensity_apply _ hs]
  have : ∫⁻ y in s, ((fejerDensity T y).toNNReal : ℝ≥0∞)
      = ∫⁻ y in s, ENNReal.ofReal (fejerDensity T y) := rfl
  rw [this, ← ofReal_integral_eq_lintegral_ofReal (integrable_fejerDensity hT).integrableOn
    (Filter.Eventually.of_forall (fejerDensity_nonneg T)),
    ENNReal.toReal_ofReal (setIntegral_nonneg hs fun y _ => fejerDensity_nonneg T y)]

theorem integral_Ioi_inv_sq {a : ℝ} (ha : 0 < a) : ∫ y in Set.Ioi a, (y ^ 2)⁻¹ = a⁻¹ := by
  have h := integral_Ioi_of_hasDerivAt_of_nonneg (g := fun y : ℝ => -y⁻¹) (g' := fun y => (y ^ 2)⁻¹)
    (a := a) (l := 0) (continuousAt_inv₀ ha.ne').neg.continuousWithinAt
    (fun y hy => by
      have hy0 : y ≠ 0 := by have : a < y := hy; linarith
      exact (hasDerivAt_inv hy0).neg.congr_deriv (by ring))
    (fun y _ => by positivity)
    (by simpa using (tendsto_inv_atTop_zero : Tendsto (fun r : ℝ => r⁻¹) atTop (𝓝 0)).neg)
  rw [h]; ring

theorem integrableOn_inv_sq_Iic {a : ℝ} (ha : 0 < a) :
    IntegrableOn (fun y : ℝ => (y ^ 2)⁻¹) (Set.Iic (-a)) := by
  refine (integrable_inv_one_add_sq.const_mul (1 + (a ^ 2)⁻¹)).integrableOn.mono' ?_ ?_
  · exact ((continuous_pow 2).continuousOn.inv₀ (fun y hy => by
      have : y ≤ -a := hy; exact pow_ne_zero 2 (by linarith))).aestronglyMeasurable measurableSet_Iic
  · rw [ae_restrict_iff' measurableSet_Iic]
    refine Filter.Eventually.of_forall fun y hy => ?_
    have hy : y ≤ -a := hy
    have hy2 : a ^ 2 ≤ y ^ 2 := by nlinarith
    have ha2 : 0 < a ^ 2 := by positivity
    have hy2' : 0 < y ^ 2 := by linarith
    rw [Real.norm_of_nonneg (by positivity)]
    rw [show (1 + (a ^ 2)⁻¹) * (1 + y ^ 2)⁻¹ = (1 + a ^ 2) / (a ^ 2 * (1 + y ^ 2)) by
      field_simp; try ring,
      inv_eq_one_div, div_le_div_iff₀ hy2' (by positivity)]
    nlinarith

theorem integral_Iic_inv_sq {a : ℝ} (ha : 0 < a) : ∫ y in Set.Iic (-a), (y ^ 2)⁻¹ = a⁻¹ := by
  have h := integral_Iic_of_hasDerivAt_of_tendsto (f := fun y : ℝ => -y⁻¹) (f' := fun y => (y ^ 2)⁻¹)
    (a := -a) (m := 0) (continuousAt_inv₀ (by linarith : (-a : ℝ) ≠ 0)).neg.continuousWithinAt
    (fun y hy => by
      have hy0 : y ≠ 0 := by have : y < -a := hy; linarith
      exact (hasDerivAt_inv hy0).neg.congr_deriv (by ring))
    (integrableOn_inv_sq_Iic ha)
    (by simpa using (tendsto_inv_atBot_zero : Tendsto (fun r : ℝ => r⁻¹) atBot (𝓝 0)).neg)
  rw [h]; simp

/-- **Tail of the Fejér kernel**: `K_T{|y| > a} ≤ 2/(π²Ta)`. -/
theorem fejer_real_tail_le {T : ℝ} (hT : 0 < T) {a : ℝ} (ha : 0 < a) :
    (fejer T).real {y | a < ‖y‖} ≤ 2 / (π ^ 2 * T * a) := by
  haveI := isProbabilityMeasure_fejer hT
  have hsub : {y : ℝ | a < ‖y‖} ⊆ Set.Iic (-a) ∪ Set.Ioi a := by
    intro y hy
    simp only [Set.mem_setOf_eq, Real.norm_eq_abs] at hy
    rcases lt_abs.mp hy with h | h
    · exact Or.inr h
    · exact Or.inl (by simp only [Set.mem_Iic]; linarith)
  have hbound : ∀ s : Set ℝ, MeasurableSet s → (∀ y ∈ s, y ≠ 0)
      → IntegrableOn (fun y : ℝ => (y ^ 2)⁻¹) s
      → (fejer T).real s ≤ 1 / (π ^ 2 * T) * ∫ y in s, (y ^ 2)⁻¹ := by
    intro s hs hs0 hint
    rw [fejer_real_eq hT hs, ← integral_const_mul]
    exact setIntegral_mono_on (integrable_fejerDensity hT).integrableOn (hint.const_mul _) hs
      fun y hy => fejerDensity_le_inv hT (hs0 y hy)
  have h1 := hbound (Set.Iic (-a)) measurableSet_Iic (fun y hy => by
    have : y ≤ -a := hy; intro h0; linarith) (integrableOn_inv_sq_Iic ha)
  have h2 := hbound (Set.Ioi a) measurableSet_Ioi (fun y hy => by
    have : a < y := hy; intro h0; linarith)
    (integrableOn_Ioi_deriv_of_nonneg (g := fun y : ℝ => -y⁻¹) (l := 0)
      (continuousAt_inv₀ ha.ne').neg.continuousWithinAt
      (fun y hy => by
        have hy0 : y ≠ 0 := by have : a < y := hy; linarith
        exact (hasDerivAt_inv hy0).neg.congr_deriv (by ring))
      (fun y _ => by positivity)
      (by simpa using (tendsto_inv_atTop_zero : Tendsto (fun r : ℝ => r⁻¹) atTop (𝓝 0)).neg))
  rw [integral_Iic_inv_sq ha] at h1
  rw [integral_Ioi_inv_sq ha] at h2
  calc (fejer T).real {y | a < ‖y‖} ≤ (fejer T).real (Set.Iic (-a) ∪ Set.Ioi a) :=
        measureReal_mono hsub
    _ ≤ (fejer T).real (Set.Iic (-a)) + (fejer T).real (Set.Ioi a) := measureReal_union_le _ _
    _ ≤ 1 / (π ^ 2 * T) * a⁻¹ + 1 / (π ^ 2 * T) * a⁻¹ := add_le_add h1 h2
    _ = 2 / (π ^ 2 * T * a) := by field_simp; ring

/-- The smoothing weight is integrable for the Fejér kernel: it vanishes beyond `|ξ| ≥ T`. -/
theorem integrable_fejer_weight {T : ℝ} (hT : 0 < T) :
    Integrable (fun ξ : ℝ => min 1 (‖charFun (fejer T) (2 * π * ξ)‖ / |ξ|)) := by
  simp_rw [norm_charFun_fejer hT]
  have hmeas : Measurable fun ξ : ℝ => min 1 (tri T ξ / |ξ|) :=
    measurable_const.min ((continuous_tri T).measurable.div continuous_abs.measurable)
  have hind : Integrable ((Set.Icc (-T) T).indicator fun _ => (1 : ℝ)) :=
    (integrableOn_const (C := (1 : ℝ)) (s := Set.Icc (-T) T) (μ := volume) (by
      rw [Real.volume_Icc]; exact ENNReal.ofReal_ne_top)).integrable_indicator measurableSet_Icc
  refine hind.mono' hmeas.aestronglyMeasurable (Filter.Eventually.of_forall fun ξ => ?_)
  rw [Real.norm_of_nonneg (le_min zero_le_one (div_nonneg (tri_nonneg _ _) (abs_nonneg _)))]
  by_cases h : ξ ∈ Set.Icc (-T) T
  · rw [Set.indicator_of_mem h]; exact min_le_left _ _
  · rw [Set.indicator_of_notMem h]
    rw [Set.mem_Icc, ← abs_le, not_le] at h
    rw [tri_eq_zero hT h.le, zero_div, min_eq_right zero_le_one]

/-- **Esseen's smoothing inequality with the Fejér kernel** of half-width `T`: for probability
measures `μ, ν` on `ℝ` with `F_ν` `m`-Lipschitz and every `a > 0`,
`|F_μ(x₀) − F_ν(x₀)| (1 − 4/(π²Ta)) ≤ 2 ∫_{|ξ|<T} (1 − |ξ|/T)/(π|ξ|) · |φ_μ(2πξ) − φ_ν(2πξ)| dξ + 2ma`.
Only the characteristic functions on `|t| < 2πT` enter. -/
theorem esseen_inequality_fejer (μ ν : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    {m : ℝ} (hm : 0 ≤ m) (hG : ∀ x y, x ≤ y → cdfR ν y - cdfR ν x ≤ m * (y - x)) {T : ℝ}
    (hT : 0 < T) {a : ℝ} (ha : 0 < a)
    (hI : Integrable (fun ξ : ℝ => tri T ξ / (π * |ξ|)
      * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖)) (x₀ : ℝ) :
    |cdfR μ x₀ - cdfR ν x₀| * (1 - 2 * (2 / (π ^ 2 * T * a)))
      ≤ 2 * (∫ ξ, tri T ξ / (π * |ξ|) * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖)
        + 2 * m * a := by
  haveI := isProbabilityMeasure_fejer hT
  have hI' : Integrable (fun ξ : ℝ => ‖charFun (fejer T) (2 * π * ξ)‖ / (π * |ξ|)
      * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖) := by
    simpa only [norm_charFun_fejer hT] using hI
  have hD : ∀ x, |smoothDiff μ ν (fejer T) x|
      ≤ ∫ ξ, tri T ξ / (π * |ξ|) * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖ := by
    intro x
    have := abs_smoothDiff_le μ ν (fejer T) (integrable_fejer_weight hT) hI' x
    simpa only [norm_charFun_fejer hT] using this
  exact esseen_sup μ ν (fejer T) hm hG ha.le (fejer_real_tail_le hT ha) hD x₀

end Fejer

