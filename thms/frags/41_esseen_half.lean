/-! ## §41  Esseen's smoothing inequality, first half: test functions with integrable Fourier
transform

The passage from the characteristic-function bound of §40 to a statement about distributions
is Esseen's smoothing argument.  Its first half is Fourier inversion: for `g` continuous and
integrable with integrable Fourier transform, and any probability measure `μ` on `ℝ`,
`∫ g dμ = ∫ 𝓕g(ξ) φ_μ(2πξ) dξ` (`integral_eq_integral_fourier_mul_charFun`, from Mathlib's
`Continuous.fourierInv_fourier_eq` and Fubini), so two laws differ on `g` by at most
`∫ |𝓕g(ξ)| |φ_μ − φ_ν|(2πξ) dξ` (`norm_integral_sub_integral_le_fourier`).  The coordinate law
of `H D_ε v` is made a measure (`coordLaw`, the image of the uniform law on sign patterns,
`uniformSigns`), its characteristic function is the real product `∏ⱼ cos(t aⱼ)` of §40
(`charFun_coordLaw`), and the bound of §40 becomes a bound on its distance from the Gaussian
`N(0, ‖v‖²/d)` in characteristic function (`norm_charFun_coordLaw_sub_gaussian_le`).  Hence, for
every test function `g` band-limited to `|2πξ| ≤ √d`,
`|E g((H D_ε v)ᵢ) − E g(Z)| ≤ ∑ⱼ vⱼ⁴/(3d²) · ∫ |𝓕g(ξ)| (2πξ)⁴ dξ`
(`norm_integral_coordLaw_sub_gaussian_le`).  The second half of Esseen's argument, from such
test functions to indicators of half-lines with the Gaussian density bound, is not formalised;
that is the remaining step to a Kolmogorov-distance statement. -/

open scoped FourierTransform

section FourierSmoothing

/-- **Fourier representation of an expectation**: for `g` continuous, integrable, with
integrable Fourier transform, `∫ g dμ = ∫ 𝓕g(ξ) · φ_μ(2πξ) dξ`. -/
theorem integral_eq_integral_fourier_mul_charFun {g : ℝ → ℂ} (hg : Continuous g)
    (hgi : Integrable g) (hgf : Integrable (𝓕 g)) (μ : Measure ℝ) [IsProbabilityMeasure μ] :
    ∫ x, g x ∂μ = ∫ ξ, 𝓕 g ξ * charFun μ (2 * π * ξ) := by
  have hinv : ∀ x, g x = ∫ ξ, Complex.exp (↑(2 * π * ξ * x) * Complex.I) * 𝓕 g ξ := by
    intro x
    conv_lhs => rw [← hg.fourierInv_fourier_eq hgi hgf]
    rw [Real.fourierInv_eq']
    refine integral_congr_ae (Filter.Eventually.of_forall fun ξ => ?_)
    simp only [smul_eq_mul]
    congr 2
    simp only [RCLike.inner_apply, conj_trivial]
    push_cast
    ring
  have hmeas : AEStronglyMeasurable (fun p : ℝ × ℝ =>
      Complex.exp (↑(2 * π * p.2 * p.1) * Complex.I) * 𝓕 g p.2) (μ.prod volume) := by
    refine (Continuous.aestronglyMeasurable (by fun_prop)).mul ?_
    exact hgf.aestronglyMeasurable.comp_snd
  have hint : Integrable (Function.uncurry fun (x ξ : ℝ) =>
      Complex.exp (↑(2 * π * ξ * x) * Complex.I) * 𝓕 g ξ) (μ.prod volume) := by
    refine (hgf.norm.comp_snd (μ := μ)).mono' hmeas (Filter.Eventually.of_forall fun p => ?_)
    show ‖Complex.exp (↑(2 * π * p.2 * p.1) * Complex.I) * 𝓕 g p.2‖ ≤ ‖𝓕 g p.2‖
    rw [norm_mul, Complex.norm_exp_ofReal_mul_I, one_mul]
  calc ∫ x, g x ∂μ
      = ∫ x, (∫ ξ, Complex.exp (↑(2 * π * ξ * x) * Complex.I) * 𝓕 g ξ) ∂μ :=
        integral_congr_ae (μ := μ) (Filter.Eventually.of_forall fun x => hinv x)
    _ = ∫ ξ, (∫ x, Complex.exp (↑(2 * π * ξ * x) * Complex.I) * 𝓕 g ξ ∂μ) :=
        integral_integral_swap (f := fun (x ξ : ℝ) =>
          Complex.exp (↑(2 * π * ξ * x) * Complex.I) * 𝓕 g ξ) hint
    _ = ∫ ξ, 𝓕 g ξ * charFun μ (2 * π * ξ) := by
        refine integral_congr_ae (Filter.Eventually.of_forall fun ξ => ?_)
        show ∫ x, Complex.exp (↑(2 * π * ξ * x) * Complex.I) * 𝓕 g ξ ∂μ
          = 𝓕 g ξ * charFun μ (2 * π * ξ)
        rw [integral_mul_const, charFun_apply_real, mul_comm]
        congr 1
        refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
        push_cast
        ring_nf

/-- **The smoothing bound for test functions**: the difference of two expectations is bounded
by the characteristic-function gap weighted by `|𝓕 g|`. -/
theorem norm_integral_sub_integral_le_fourier {g : ℝ → ℂ} (hg : Continuous g)
    (hgi : Integrable g) (hgf : Integrable (𝓕 g)) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] :
    ‖∫ x, g x ∂μ - ∫ x, g x ∂ν‖
      ≤ ∫ ξ, ‖𝓕 g ξ‖ * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖ := by
  rw [integral_eq_integral_fourier_mul_charFun hg hgi hgf μ,
    integral_eq_integral_fourier_mul_charFun hg hgi hgf ν]
  have hcf : ∀ (ρ : Measure ℝ) [IsProbabilityMeasure ρ],
      Integrable (fun ξ => 𝓕 g ξ * charFun ρ (2 * π * ξ)) volume := by
    intro ρ _
    have hm : AEStronglyMeasurable (fun ξ => 𝓕 g ξ * charFun ρ (2 * π * ξ)) volume :=
      hgf.aestronglyMeasurable.mul
        ((continuous_charFun.comp (by fun_prop : Continuous fun ξ : ℝ => 2 * π * ξ))
          |>.aestronglyMeasurable)
    refine hgf.norm.mono' hm (Filter.Eventually.of_forall fun ξ => ?_)
    show ‖𝓕 g ξ * charFun ρ (2 * π * ξ)‖ ≤ ‖𝓕 g ξ‖
    rw [norm_mul]
    exact mul_le_of_le_one_right (norm_nonneg _) (norm_charFun_le_one _)
  have h1 := hcf μ
  have h2 := hcf ν
  rw [← integral_sub h1 h2]
  refine (norm_integral_le_integral_norm _).trans (le_of_eq ?_)
  refine integral_congr_ae (Filter.Eventually.of_forall fun ξ => ?_)
  show ‖𝓕 g ξ * charFun μ (2 * π * ξ) - 𝓕 g ξ * charFun ν (2 * π * ξ)‖
    = ‖𝓕 g ξ‖ * ‖charFun μ (2 * π * ξ) - charFun ν (2 * π * ξ)‖
  rw [← mul_sub, norm_mul]

end FourierSmoothing

namespace Hadamard

open Orthogonal


variable {d : ℕ}

/-- The uniform law on sign patterns. -/
noncomputable def uniformSigns (d : ℕ) : Measure (Signs d) :=
  ((2 : ℝ≥0∞) ^ d)⁻¹ • Measure.count

theorem card_signs_ennreal (d : ℕ) : ((ENat.card (Signs d) : ℕ∞) : ℝ≥0∞) = (2 : ℝ≥0∞) ^ d := by
  rw [ENat.card_eq_coe_fintype_card, card_signs]
  simp

instance : IsProbabilityMeasure (uniformSigns d) := by
  constructor
  rw [uniformSigns, Measure.smul_apply, Measure.count_univ, card_signs_ennreal, smul_eq_mul,
    ENNReal.inv_mul_cancel (by positivity) (by simp)]

theorem integral_uniformSigns (f : Signs d → ℝ) : ∫ s, f s ∂(uniformSigns d) = signAvg f := by
  rw [integral_fintype Integrable.of_finite, signAvg, Finset.sum_div]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [uniformSigns, measureReal_def, Measure.smul_apply, Measure.count_singleton]
  simp only [smul_eq_mul, mul_one]
  rw [ENNReal.toReal_inv, ENNReal.toReal_pow, ENNReal.toReal_ofNat]
  ring

/-- The law of the coordinate `(H D_ε v)ᵢ` under uniform signs. -/
noncomputable def coordLaw (H : Od d) (v : Ed d) (i : Fin d) : Measure ℝ :=
  (uniformSigns d).map fun s => (hadRot H s • v) i

instance (H : Od d) (v : Ed d) (i : Fin d) : IsProbabilityMeasure (coordLaw H v i) := by
  constructor
  rw [coordLaw, Measure.map_apply (measurable_of_countable _) MeasurableSet.univ,
    Set.preimage_univ, measure_univ]

theorem integral_coordLaw (H : Od d) (v : Ed d) (i : Fin d) {g : ℝ → ℝ} (hg : Measurable g) :
    ∫ x, g x ∂(coordLaw H v i) = signAvg fun s => g ((hadRot H s • v) i) := by
  rw [coordLaw, integral_map (measurable_of_countable _).aemeasurable hg.aestronglyMeasurable,
    integral_uniformSigns]

/-- The characteristic function of the coordinate law is real: `∏ⱼ cos(t aⱼ)`. -/
theorem charFun_coordLaw (H : Od d) (v : Ed d) (i : Fin d) (t : ℝ) :
    charFun (coordLaw H v i) t
      = ((∏ j, Real.cos (t * ((H : Matrix (Fin d) (Fin d) ℝ) i j * v j)) : ℝ) : ℂ) := by
  rw [charFun_apply_real, coordLaw, integral_map (measurable_of_countable _).aemeasurable
    (Continuous.aestronglyMeasurable (by fun_prop))]
  have hcoord : ∀ s, (hadRot H s • v) i = radSum (fun j => (H : Matrix (Fin d) (Fin d) ℝ) i j * v j) s :=
    fun s => hadRot_apply_eq_radSum H s v i
  simp_rw [hcoord]
  have hexp : ∀ s, Complex.exp (↑t * ↑(radSum (fun j => (H : Matrix (Fin d) (Fin d) ℝ) i j * v j) s)
      * Complex.I) = ((Real.cos (t * radSum (fun j => (H : Matrix (Fin d) (Fin d) ℝ) i j * v j) s) : ℝ) : ℂ)
        + ((Real.sin (t * radSum (fun j => (H : Matrix (Fin d) (Fin d) ℝ) i j * v j) s) : ℝ) : ℂ)
          * Complex.I := by
    intro s
    rw [← Complex.ofReal_mul, Complex.exp_mul_I, Complex.ofReal_cos, Complex.ofReal_sin]
  simp_rw [hexp]
  rw [integral_add, integral_mul_const, integral_complex_ofReal, integral_complex_ofReal,
    integral_uniformSigns, integral_uniformSigns, signAvg_cos_radSum,
    signAvg_odd _ (F := fun x => Real.sin (t * x)) (fun x => by rw [mul_neg, Real.sin_neg])]
  · simp
  · exact Integrable.of_finite
  · exact Integrable.of_finite

/-- The characteristic-function gap between the coordinate law and the Gaussian
`N(0, ‖v‖²/d)`, for `|t| ≤ √d`. -/
theorem norm_charFun_coordLaw_sub_gaussian_le {H : Od d} (hH : IsFlat H) {v : Ed d}
    (hv : ‖v‖ ≤ 1) (i : Fin d) {t : ℝ} (ht : |t| ≤ Real.sqrt d) :
    ‖charFun (coordLaw H v i) t - charFun (gaussianReal 0 (Real.toNNReal (‖v‖ ^ 2 / d))) t‖
      ≤ t ^ 4 * (∑ j, v j ^ 4) / (3 * (d : ℝ) ^ 2) := by
  rw [charFun_coordLaw, charFun_gaussianReal]
  have h := abs_signAvg_cos_coord_sub_exp_le hH hv i ht
  have hcoord : ∀ s, (hadRot H s • v) i = radSum (fun j => (H : Matrix (Fin d) (Fin d) ℝ) i j * v j) s :=
    fun s => hadRot_apply_eq_radSum H s v i
  simp_rw [hcoord, signAvg_cos_radSum] at h
  have hnn : (0 : ℝ) ≤ ‖v‖ ^ 2 / d := by positivity
  have hg : Complex.exp (↑t * ↑(0 : ℝ) * Complex.I - ↑↑(Real.toNNReal (‖v‖ ^ 2 / d)) * ↑t ^ 2 / 2)
      = ((Real.exp (-(t ^ 2 * ‖v‖ ^ 2 / d) / 2) : ℝ) : ℂ) := by
    rw [Complex.ofReal_exp]
    congr 1
    rw [Real.coe_toNNReal _ hnn]
    push_cast
    ring
  rw [hg, ← Complex.ofReal_sub, Complex.norm_real, Real.norm_eq_abs]
  exact h

/-- **Esseen's smoothing bound, first half, for the deployed rotation**: for any band-limited
test function `g` (with `𝓕 g` vanishing beyond `|2πξ| ≤ √d`), the expectation of `g` under the
coordinate law of `H D_ε v` differs from its Gaussian expectation by at most
`∑ⱼ vⱼ⁴/(3d²) · ∫ |𝓕g(ξ)| (2πξ)⁴ dξ`. -/
theorem norm_integral_coordLaw_sub_gaussian_le {H : Od d} (hH : IsFlat H) {v : Ed d}
    (hv : ‖v‖ ≤ 1) (i : Fin d) {g : ℝ → ℂ} (hg : Continuous g) (hgi : Integrable g)
    (hgf : Integrable (𝓕 g)) (hband : ∀ ξ : ℝ, Real.sqrt d < |2 * π * ξ| → 𝓕 g ξ = 0) :
    ‖∫ x, g x ∂(coordLaw H v i) - ∫ x, g x ∂(gaussianReal 0 (Real.toNNReal (‖v‖ ^ 2 / d)))‖
      ≤ (∑ j, v j ^ 4) / (3 * (d : ℝ) ^ 2) * ∫ ξ, ‖𝓕 g ξ‖ * (2 * π * ξ) ^ 4 := by
  refine (norm_integral_sub_integral_le_fourier hg hgi hgf _ _).trans ?_
  have hpt : ∀ ξ : ℝ, ‖𝓕 g ξ‖ * ‖charFun (coordLaw H v i) (2 * π * ξ)
        - charFun (gaussianReal 0 (Real.toNNReal (‖v‖ ^ 2 / d))) (2 * π * ξ)‖
      ≤ (∑ j, v j ^ 4) / (3 * (d : ℝ) ^ 2) * (‖𝓕 g ξ‖ * (2 * π * ξ) ^ 4) := by
    intro ξ
    rcases le_or_gt |2 * π * ξ| (Real.sqrt d) with hξ | hξ
    · have := norm_charFun_coordLaw_sub_gaussian_le hH hv i hξ
      calc ‖𝓕 g ξ‖ * ‖charFun (coordLaw H v i) (2 * π * ξ)
            - charFun (gaussianReal 0 (Real.toNNReal (‖v‖ ^ 2 / d))) (2 * π * ξ)‖
          ≤ ‖𝓕 g ξ‖ * ((2 * π * ξ) ^ 4 * (∑ j, v j ^ 4) / (3 * (d : ℝ) ^ 2)) :=
            mul_le_mul_of_nonneg_left this (norm_nonneg _)
        _ = _ := by ring
    · rw [hband ξ hξ, norm_zero, zero_mul, zero_mul, mul_zero]
  have hint : Integrable (fun ξ : ℝ => ‖𝓕 g ξ‖ * (2 * π * ξ) ^ 4) volume := by
    refine (hgf.norm.const_mul ((d : ℝ) ^ 2)).mono'
      (hgf.norm.aestronglyMeasurable.mul (by fun_prop))
      (Filter.Eventually.of_forall fun ξ => ?_)
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity), mul_comm]
    rcases le_or_gt |2 * π * ξ| (Real.sqrt d) with hξ | hξ
    · refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
      have h1 : (2 * π * ξ) ^ 4 = (|2 * π * ξ| ^ 2) ^ 2 := by rw [sq_abs]; ring
      have h2 : |2 * π * ξ| ^ 2 ≤ (d : ℝ) := by
        calc |2 * π * ξ| ^ 2 ≤ Real.sqrt d ^ 2 := pow_le_pow_left₀ (abs_nonneg _) hξ 2
          _ = d := Real.sq_sqrt (Nat.cast_nonneg _)
      rw [h1]
      exact pow_le_pow_left₀ (sq_nonneg _) h2 2
    · rw [hband ξ hξ, norm_zero]; simp
  calc ∫ ξ, ‖𝓕 g ξ‖ * ‖charFun (coordLaw H v i) (2 * π * ξ)
        - charFun (gaussianReal 0 (Real.toNNReal (‖v‖ ^ 2 / d))) (2 * π * ξ)‖
      ≤ ∫ ξ, (∑ j, v j ^ 4) / (3 * (d : ℝ) ^ 2) * (‖𝓕 g ξ‖ * (2 * π * ξ) ^ 4) :=
        integral_mono_of_nonneg (Filter.Eventually.of_forall fun ξ => by positivity)
          (hint.const_mul _) (Filter.Eventually.of_forall hpt)
    _ = _ := integral_const_mul _ _

end Hadamard
