/-! ## §57  The spherical-versus-Gaussian leg, by coupling rather than by sup-norm

§55 bounds a distortion gap by `2 · η · spread cbk · (b − a)` out of a *sup-norm* bound `η` on
`cdf μ − cdf ν`, and §56 multiplies that by `d`.  That shape cannot be made to decay: any
Kolmogorov-Smirnov `η` between a coordinate law and its limit is `Θ(1/√d)` at best, so the printed
bound grows like `√d`.  The remedy is not a better `η` but a different quantity.  §55's
`abs_sum_fin_weighted_le` only ever needs a uniform bound on the *summands*, and those summands are
values of `cdfGapInt`, the once-integrated CDF difference of §52.  A uniform bound on `cdfGapInt`
is a Wasserstein-1 quantity: it carries no `(b − a)` factor and it obeys a coupling bound.

This section builds the spherical-versus-Gaussian leg in exactly that form.  It contains three
independent blocks.

* **A characteristic-function bridge.**  `gaussianE`, the development's own standard Gaussian on a
  finite-dimensional inner product space, is identified with Mathlib's `stdGaussian`
  (`gaussianE_eq_stdGaussian`) through the Fourier transform of the raw density
  (`integral_cexp_gaussRaw`, `charFun_gaussianE`).  Reading Mathlib's coordinate marginal back
  through that identification gives `gaussianE_map_ofLp`: a coordinate of a Gaussian vector on
  `Ed d` is `N(0,1)`.  This says nothing about how the `gaussianE`s of different ambient dimension
  relate, and nothing about coordinates in a non-orthonormal basis.
* **A support-free coupling bound on `cdfGapInt`.**  `cdfGapInt μ ν a x` is the difference of the
  two laws' expectations of the clamp `t ↦ (x − t)⁺ − (a − t)⁺`
  (`Threshold.cdfGapInt_eq_integral_clamp_sub`), and that clamp is `1`-Lipschitz
  (`Threshold.abs_clamp_sub_clamp_le`), so any coupling of `μ` and `ν` bounds `|cdfGapInt μ ν a x|`
  by its `L¹` cost (`Threshold.abs_cdfGapInt_le_integral_abs_sub_of_coupling`).  Unlike §55's
  estimate this needs no bounded support and no bound on the CDF difference itself; unlike the
  compactly supported variant recorded alongside it, it applies to a Gaussian second law.  It
  names no coupling and asserts nothing about how small the `L¹` cost is.
* **The coupling, and the rate it buys.**  Take `Z ~ gaussianE (Ed d)` and couple
  `u = Z/‖Z‖` with `Z/√d`.  Coordinatewise `u_i − Z_i/√d = u_i · (√d − ‖Z‖)/√d` exactly, so
  Cauchy-Schwarz splits the `L¹` cost into `√(E[u_i²]) · √(E[(√d − ‖Z‖)²])/√d`.  The first factor
  is `1/√d` by Lemma 1 (§4's `integral_sq_sphericalCoordLaw`).  The second is
  `Var(‖Z‖) + (√d − E‖Z‖)²`, and `Var(‖Z‖) ≤ 1` **independently of the dimension** by the Gaussian
  Poincaré inequality of §32/§39 (`variance_norm_gaussianE_le`), which in turn forces
  `E‖Z‖ ≥ √(d − 1)` and hence `(√d − E‖Z‖)² ≤ 1/(4(d − 1))`.  The product is
  `√(1 + 1/(4(d−1)))/d`.

The result is `abs_cdfGapInt_sphericalCoordLaw_gaussian_le`:

  `|cdfGapInt (sphericalCoordLaw d) (N(0, 1/d)) a x| ≤ √(1 + 1/(4(d−1)))/d`

for every `d ≥ 2` and **every** pair `a ≤ x` (and trivially when `x < a`), with no hypothesis
relating `a` and `x` to the supports.

**What this does not do.**  It is one leg only: the pair compared is `sphericalCoordLaw d` against
the Gaussian `N(0, 1/d)`, not against the deployed coordinate law `coordLaw H v i`, and nothing
here composes it with §44's Kolmogorov-Smirnov bound for the other leg.  It is *not* a sup-norm
bound: no `η` bounding `|cdf μ − cdf ν|` is produced, so §55's and §56's statements are not
discharged by it, and in particular `abs_hadRate_sub_rhoSq_le` remains conditional exactly as
before.  Rewiring §55's threshold sum to consume a uniform `cdfGapInt` bound instead of `η·(b−a)`
is not done here.  No asymptotic statement is made, and no claim is made that the constant
`√(1 + 1/(4(d−1)))` is optimal; only that it is dimension-free. -/

section CharacteristicBridge

open Complex
open scoped InnerProductSpace

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

omit [Nontrivial E] in
/-- The raw (unnormalised) Gaussian's Fourier transform: `∫ e^{i⟪x,t⟫} e^{-‖x‖²/2} dx
  = (2π)^{d/2} e^{-‖t‖²/2}`, obtained from Mathlib's Gaussian-Fourier-transform machinery
  (which itself proceeds by factorising the exponential over an orthonormal basis). This does
  NOT assume `gaussianE` is normalised; the constant `(2π)^{d/2}` is exactly the total mass of
  `gaussRaw`. -/
theorem integral_cexp_gaussRaw (t : E) :
    ∫ x : E, Complex.exp ((⟪x, t⟫_ℝ : ℂ) * Complex.I) ∂(gaussRaw E)
      = (2 * (Real.pi : ℂ)) ^ ((Module.finrank ℝ E : ℂ) / 2)
        * Complex.exp (-((‖t‖:ℂ)) ^ 2 / 2) := by
  have hfmeas : AEMeasurable (fun x : E => (Real.exp (-(‖x‖ ^ 2) / 2)).toNNReal) volume := by
    fun_prop
  have hgd : (gaussDensity : E → ℝ≥0∞)
      = fun x => ((fun x : E => (Real.exp (-(‖x‖ ^ 2) / 2)).toNNReal) x : ℝ≥0∞) := by
    funext x; unfold Progressive.gaussDensity; rw [ENNReal.ofReal]
  rw [gaussRaw, hgd, integral_withDensity_eq_integral_smul₀ hfmeas]
  have hpt : ∀ x : E, ((Real.exp (-(‖x‖ ^ 2) / 2)).toNNReal : ℝ≥0)
        • Complex.exp ((⟪x, t⟫_ℝ : ℂ) * Complex.I)
      = Complex.exp (-((‖x‖:ℂ)) ^ 2 / 2) * Complex.exp ((⟪x, t⟫_ℝ : ℂ) * Complex.I) := by
    intro x
    rw [NNReal.smul_def, Real.coe_toNNReal _ (Real.exp_pos _).le,
      Complex.real_smul, Complex.ofReal_exp]
    congr 2
    push_cast
    ring
  simp_rw [hpt]
  have h := GaussianFourier.integral_cexp_neg_mul_sq_norm_add (V := E) (b := (1/2 : ℂ))
    (by norm_num) Complex.I t
  have hcongr : (fun x : E => Complex.exp (-((‖x‖:ℂ)) ^ 2 / 2)
        * Complex.exp ((⟪x, t⟫_ℝ : ℂ) * Complex.I))
      = (fun x : E => Complex.exp (-(1/2 : ℂ) * (‖x‖:ℂ) ^ 2 + Complex.I * (⟪t, x⟫_ℝ : ℂ))) := by
    funext x
    rw [← Complex.exp_add, real_inner_comm x t]
    ring_nf
  rw [hcongr, h]
  congr 1
  · rw [show (Real.pi:ℂ) / (1/2 : ℂ) = 2 * Real.pi by ring]
  · rw [show Complex.I ^ 2 * (‖t‖:ℂ) ^ 2 / (4 * (1/2 : ℂ)) = -((‖t‖:ℂ))^2/2 by
      rw [Complex.I_sq]; ring]

/-- **The characteristic function of `gaussianE`.**  Matches Mathlib's `charFun_stdGaussian`,
obtained by normalising the raw Gaussian Fourier transform `integral_cexp_gaussRaw` at `t = 0`
against `IsProbabilityMeasure (gaussianE E)`; this sidesteps ever computing the normalising
constant `gaussRaw E Set.univ` explicitly. -/
theorem charFun_gaussianE (t : E) :
    charFun (gaussianE E) t = Complex.exp (-((‖t‖:ℂ)) ^ 2 / 2) := by
  set A : ℂ := ((gaussRaw E Set.univ)⁻¹.toReal : ℂ) with hA
  set B : ℂ := (2 * (Real.pi : ℂ)) ^ ((Module.finrank ℝ E : ℂ) / 2) with hB
  have hscale : ∀ s : E, charFun (gaussianE E) s = A * charFun (gaussRaw E) s := by
    intro s
    rw [hA, charFun_apply, charFun_apply, gaussianE, integral_smul_measure, Complex.real_smul]
  have hraw : ∀ s : E, charFun (gaussRaw E) s = B * Complex.exp (-((‖s‖:ℂ))^2/2) := by
    intro s
    rw [charFun_apply, hB, ← integral_cexp_gaussRaw s]
  have h1 : A * B = 1 := by
    have heq := hscale 0
    rw [hraw 0, charFun_zero] at heq
    have huniv : (gaussianE E).real Set.univ = 1 := by
      rw [measureReal_def, measure_univ, ENNReal.toReal_one]
    rw [huniv] at heq
    norm_num at heq
    exact heq.symm
  rw [hscale t, hraw t, ← mul_assoc, h1, one_mul]

/-- **`gaussianE` is Mathlib's `stdGaussian`.**  Both are probability measures on a
finite-dimensional real inner product space with the same characteristic function
`t ↦ exp(-‖t‖²/2)`, so `Measure.ext_of_charFun` identifies them. -/
theorem gaussianE_eq_stdGaussian :
    gaussianE E = ProbabilityTheory.stdGaussian E := by
  apply Measure.ext_of_charFun
  funext t
  rw [charFun_gaussianE, ProbabilityTheory.charFun_stdGaussian]

end

end CharacteristicBridge

noncomputable section CoordinateMarginal

open Orthogonal

/-- The coordinate marginal of Mathlib's `stdGaussian` on `EuclideanSpace` is `gaussianReal 0 1`.
A pure Mathlib computation: the projection is a norm-one continuous linear functional, so
`IsGaussian.map_eq_gaussianReal` applies and `variance_dual_stdGaussian` evaluates the variance. -/
theorem stdGaussian_map_proj {d : ℕ} (i : Fin d) :
    (stdGaussian (EuclideanSpace ℝ (Fin d))).map (EuclideanSpace.proj i) = gaussianReal 0 1 := by
  have hnorm : ‖EuclideanSpace.proj (𝕜 := ℝ) i‖ = 1 := by
    refine le_antisymm (ContinuousLinearMap.opNorm_le_bound _ zero_le_one fun x => ?_) ?_
    · simpa using PiLp.norm_apply_le x i
    · have := (EuclideanSpace.proj (𝕜 := ℝ) i).le_opNorm (EuclideanSpace.single i (1:ℝ))
      simpa using this
  rw [IsGaussian.map_eq_gaussianReal (EuclideanSpace.proj i)]
  congr 1
  · rw [ContinuousLinearMap.integral_comp_id_comm IsGaussian.integrable_id,
      integral_id_stdGaussian]; simp
  · rw [variance_dual_stdGaussian, hnorm]; simp

/-- **A coordinate of a Gaussian vector is `N(0,1)`.**  The coordinate marginal of the
development's own `gaussianE` on `Ed d`, obtained by identifying `gaussianE` with Mathlib's
`stdGaussian` (`gaussianE_eq_stdGaussian`, via characteristic functions) and then reading off
`stdGaussian_map_proj`.  This does NOT show that the `gaussianE`s of different ambient dimension
are compatible under inclusion, nor does it say anything about coordinates in a non-orthonormal
basis. -/
theorem gaussianE_map_ofLp {d : ℕ} (i : Fin d) :
    (gaussianE (Ed d)).map (fun z => z.ofLp i) = gaussianReal 0 1 := by
  haveI : Nontrivial (Ed d) := by
    refine ⟨WithLp.toLp 2 (Pi.single i (1:ℝ)), 0, ?_⟩
    intro h
    have := congrFun (congrArg WithLp.ofLp h) i
    simp at this
  have hfun : (fun z : Ed d => z.ofLp i) = ⇑(EuclideanSpace.proj (𝕜 := ℝ) i) := by
    funext z; rfl
  rw [hfun, gaussianE_eq_stdGaussian, stdGaussian_map_proj]

/-- The law of `Z_i/√d` when `Z_i` is `N(0,1)`: the Gaussian `N(0, 1/d)`.  A rescaling of
Mathlib's `gaussianReal_map_const_mul`, recorded in the `1/√d` spelling the coupling below uses. -/
theorem gaussianReal_map_div_sqrt {d : ℕ} (hd : 2 ≤ d) :
    (gaussianReal 0 1).map (fun t : ℝ => t / Real.sqrt (d : ℝ))
      = gaussianReal 0 (Real.toNNReal (1 / (d : ℝ))) := by
  have hdpos : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hfun : (fun t : ℝ => t / Real.sqrt (d : ℝ)) = ((Real.sqrt (d : ℝ))⁻¹ * ·) := by
    funext t; rw [div_eq_inv_mul]
  rw [hfun, gaussianReal_map_const_mul]
  congr 1
  · simp
  · rw [mul_one]
    ext
    rw [Real.coe_toNNReal _ (by positivity)]
    show ((Real.sqrt (d : ℝ))⁻¹) ^ 2 = 1 / (d : ℝ)
    rw [inv_pow, Real.sq_sqrt hdpos.le, one_div]

end CoordinateMarginal

/-! ### §57a  A coupling bound on `cdfGapInt`, with no support hypothesis

The three lemmas below are about `cdfGapInt` and couplings and nothing else: no codebook, no
quantiser, no distortion appears.  The key identity is that `cdfGapInt μ ν a x` is the difference
of the two laws' expectations of the clamp `t ↦ (x − t)⁺ − (a − t)⁺`.  That clamp is bounded by
`x − a` and is `1`-Lipschitz, so the difference is controlled by the `L¹` cost of any coupling,
with no hypothesis on the supports of `μ` and `ν`. -/

namespace Threshold

/-- The length of `Ioc a x ∩ Ici s`, in closed form: the positive-part difference
`(x − s)⁺ − (a − s)⁺`.  This is the elementary real-number identity underlying the Fubini step
below; it needs `a ≤ x` because otherwise `Ioc a x` is empty while the right side need not be
zero. -/
theorem length_Ioc_inter_Ici {a x : ℝ} (hax : a ≤ x) (s : ℝ) :
    (volume : Measure ℝ).real (Set.Ioc a x ∩ Set.Ici s) = max (x - s) 0 - max (a - s) 0 := by
  rcases lt_or_ge a s with h1 | h1
  · rcases lt_or_ge x s with h2 | h2
    · have he : Set.Ioc a x ∩ Set.Ici s = (∅ : Set ℝ) := by
        ext u
        simp only [Set.mem_inter_iff, Set.mem_Ioc, Set.mem_Ici, Set.mem_empty_iff_false, iff_false]
        rintro ⟨⟨-, hu2⟩, hu3⟩
        linarith
      rw [he, measureReal_empty, max_eq_right (by linarith), max_eq_right (by linarith)]
      ring
    · have he : Set.Ioc a x ∩ Set.Ici s = Set.Icc s x := by
        ext u
        simp only [Set.mem_inter_iff, Set.mem_Ioc, Set.mem_Ici, Set.mem_Icc]
        constructor
        · exact fun h => ⟨h.2, h.1.2⟩
        · intro h; exact ⟨⟨h1.trans_le h.1, h.2⟩, h.1⟩
      rw [he, Real.volume_real_Icc, max_eq_left (by linarith),
        max_eq_right (by linarith)]
      ring
  · have he : Set.Ioc a x ∩ Set.Ici s = Set.Ioc a x := by
      ext u
      simp only [Set.mem_inter_iff, Set.mem_Ioc, Set.mem_Ici]
      constructor
      · exact fun h => h.1
      · intro h; exact ⟨h, h1.trans h.1.le⟩
    rw [he, Real.volume_real_Ioc, max_eq_left (by linarith), max_eq_left (by linarith),
      max_eq_left (by linarith)]
    ring

/-- Fubini step: the integral of the cdf over `(a, x]` equals the `μ`-expectation of the clamp
function `(x − t)⁺ − (a − t)⁺`.  No support hypothesis on `μ` is used; this is a general fact
about any probability law on `ℝ`. -/
theorem cdf_Ioc_integral_eq_integral_diff (μ : Measure ℝ) [IsProbabilityMeasure μ] {a x : ℝ}
    (hax : a ≤ x) :
    (∫ u in Set.Ioc a x, (cdf μ) u) = ∫ t, (max (x - t) 0 - max (a - t) 0) ∂μ := by
  haveI hfin : IsFiniteMeasure ((volume : Measure ℝ).restrict (Set.Ioc a x)) := by
    constructor
    rw [Measure.restrict_apply_univ, Real.volume_Ioc]
    exact ENNReal.ofReal_lt_top
  set F : ℝ → ℝ → ℝ := fun u t => (Set.Iic u).indicator (1 : ℝ → ℝ) t with hF
  have hmeas : Measurable (Function.uncurry F) := by
    have hset : MeasurableSet {p : ℝ × ℝ | p.2 ≤ p.1} :=
      isClosed_le continuous_snd continuous_fst |>.measurableSet
    have heq : Function.uncurry F = Set.indicator {p : ℝ × ℝ | p.2 ≤ p.1} (fun _ => (1 : ℝ)) := by
      funext p
      simp [hF, Function.uncurry, Set.indicator, Set.mem_Iic]
    rw [heq]
    exact measurable_const.indicator hset
  have hInt : Integrable (Function.uncurry F) ((volume.restrict (Set.Ioc a x)).prod μ) := by
    refine (integrable_const (1:ℝ)).mono' hmeas.aestronglyMeasurable ?_
    filter_upwards with p
    rw [Real.norm_eq_abs, hF]
    simp only [Function.uncurry, Set.indicator, Set.mem_Iic]
    split <;> norm_num
  have hswap := integral_integral_swap (μ := (volume.restrict (Set.Ioc a x))) (ν := μ)
    (f := F) hInt
  have hL : ∀ u, ∫ t, F u t ∂μ = (cdf μ) u := by
    intro u
    rw [hF]
    simp only
    rw [integral_indicator_one measurableSet_Iic, ← cdf_eq_real]
  have hR : ∀ t, ∫ u in Set.Ioc a x, F u t ∂volume = max (x - t) 0 - max (a - t) 0 := by
    intro t
    have hswap2 : ∀ u : ℝ, F u t = (Set.Ici t).indicator (1 : ℝ → ℝ) u := by
      intro u
      rw [hF]
      simp only [Set.indicator, Set.mem_Iic, Set.mem_Ici]
      congr 1
    simp_rw [hswap2]
    rw [setIntegral_indicator measurableSet_Ici]
    simp only [Pi.one_apply, setIntegral_const, smul_eq_mul, mul_one]
    rw [length_Ioc_inter_Ici hax]
  simp only [hL] at hswap
  simp only [hR] at hswap
  exact hswap

/-- With `μ` supported on `[a, ∞)`, the integral of its cdf over `(a, x]` is the `μ`-expectation
of the positive part `(x − t)⁺`, with no boundary term at `a`.  Only the compactly supported leg
uses this; the Gaussian leg below cannot. -/
theorem integral_cdf_Ioc_eq_integral_posPart (μ : Measure ℝ) [IsProbabilityMeasure μ] {a x : ℝ}
    (hax : a ≤ x) (hsupp : μ (Set.Iio a) = 0) :
    (∫ u in Set.Ioc a x, (cdf μ) u) = ∫ t, max (x - t) 0 ∂μ := by
  rw [cdf_Ioc_integral_eq_integral_diff μ hax]
  have hae : ∀ᵐ t ∂μ, a ≤ t := by
    rw [ae_iff]
    have : {t : ℝ | ¬ a ≤ t} = Set.Iio a := by ext y; simp
    rw [this]; exact hsupp
  refine integral_congr_ae ?_
  filter_upwards [hae] with t ht
  rw [max_eq_right (by linarith : a - t ≤ 0), sub_zero]

/-- **The clamp form of `cdfGapInt`.**  For `a ≤ x` and any two probability laws,
`cdfGapInt μ ν a x` is the difference of their expectations of the clamp
`t ↦ (x − t)⁺ − (a − t)⁺`.  No hypothesis is placed on the supports, which is what lets the
Gaussian leg below use it. -/
theorem cdfGapInt_eq_integral_clamp_sub (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] {a x : ℝ} (hax : a ≤ x) :
    cdfGapInt μ ν a x = (∫ t, (max (x - t) 0 - max (a - t) 0) ∂μ)
      - ∫ t, (max (x - t) 0 - max (a - t) 0) ∂ν := by
  rw [cdfGapInt_eq_sub, cdf_Ioc_integral_eq_integral_diff μ hax,
    cdf_Ioc_integral_eq_integral_diff ν hax]

/-- The same identity, for laws that put no mass below `a`: the boundary term drops and the clamp
becomes the bare positive part `(x − t)⁺`. -/
theorem cdfGapInt_eq_integral_posPart_sub (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] {a x : ℝ} (hax : a ≤ x) (hsupp_mu : μ (Set.Iio a) = 0)
    (hsupp_nu : ν (Set.Iio a) = 0) :
    cdfGapInt μ ν a x = (∫ t, max (x - t) 0 ∂μ) - ∫ t, max (x - t) 0 ∂ν := by
  rw [cdfGapInt_eq_sub, integral_cdf_Ioc_eq_integral_posPart μ hax hsupp_mu,
    integral_cdf_Ioc_eq_integral_posPart ν hax hsupp_nu]

/-- The clamp is `1`-Lipschitz.  Both `(x − ·)⁺` and `(a − ·)⁺` are nonincreasing and
`1`-Lipschitz, so their difference moves by at most `|s − t|` even though a naive triangle
inequality would give `2|s − t|`.  No relation between `a` and `x` is needed. -/
theorem abs_clamp_sub_clamp_le (a x : ℝ) (s t : ℝ) :
    |(max (x - s) 0 - max (a - s) 0) - (max (x - t) 0 - max (a - t) 0)| ≤ |s - t| := by
  have key : ∀ u v : ℝ, u ≤ v →
      |(max (x - u) 0 - max (a - u) 0) - (max (x - v) 0 - max (a - v) 0)| ≤ |u - v| := by
    intro u v huv
    have h1 : 0 ≤ max (x - u) 0 - max (x - v) 0 ∧ max (x - u) 0 - max (x - v) 0 ≤ v - u := by
      simp only [max_def]; split_ifs <;> constructor <;> linarith
    have h2 : 0 ≤ max (a - u) 0 - max (a - v) 0 ∧ max (a - u) 0 - max (a - v) 0 ≤ v - u := by
      simp only [max_def]; split_ifs <;> constructor <;> linarith
    rw [abs_sub_comm u v, abs_of_nonneg (by linarith : (0:ℝ) ≤ v - u), abs_le]
    constructor <;> linarith [h1.1, h1.2, h2.1, h2.2]
  rcases le_total s t with h | h
  · exact key s t h
  · rw [abs_sub_comm ((max (x - s) 0 - max (a - s) 0)), abs_sub_comm s t]; exact key t s h

/-- The clamp is bounded by `x − a`, which makes it integrable against any probability law. -/
theorem abs_clamp_le {a x : ℝ} (hax : a ≤ x) (t : ℝ) :
    |max (x - t) 0 - max (a - t) 0| ≤ x - a := by
  rw [abs_le]; simp only [max_def]; split_ifs <;> constructor <;> linarith

/-- **The coupling bound on `cdfGapInt`.**  If `X, Y : Ω → ℝ` are integrable random variables on
a common probability space with laws `μ, ν`, then `|cdfGapInt μ ν a x|` is at most the `L¹` cost
`E|X − Y|` of the coupling, for every `a ≤ x`.  No support hypothesis: `μ` and `ν` may both have
full support, which is what §55's sup-norm route could not accommodate.  Nothing here asserts a
particular coupling is good; it says what any coupling buys. -/
theorem abs_cdfGapInt_le_integral_abs_sub_of_coupling
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    {X Y : Ω → ℝ} (hX : AEMeasurable X P) (hY : AEMeasurable Y P)
    (hIntX : Integrable X P) (hIntY : Integrable Y P)
    {μ ν : Measure ℝ} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hXeq : Measure.map X P = μ) (hYeq : Measure.map Y P = ν)
    {a x : ℝ} (hax : a ≤ x) :
    |cdfGapInt μ ν a x| ≤ ∫ ω, |X ω - Y ω| ∂P := by
  set g : ℝ → ℝ := fun t => max (x - t) 0 - max (a - t) 0 with hg
  have hgc : Continuous g := by fun_prop
  rw [cdfGapInt_eq_integral_clamp_sub μ ν hax]
  have hMX : (∫ t, g t ∂μ) = ∫ ω, g (X ω) ∂P := by
    rw [← hXeq]; exact integral_map hX hgc.aestronglyMeasurable
  have hMY : (∫ t, g t ∂ν) = ∫ ω, g (Y ω) ∂P := by
    rw [← hYeq]; exact integral_map hY hgc.aestronglyMeasurable
  rw [hMX, hMY]
  have hmX : AEMeasurable (fun ω => g (X ω)) P := hgc.measurable.comp_aemeasurable hX
  have hmY : AEMeasurable (fun ω => g (Y ω)) P := hgc.measurable.comp_aemeasurable hY
  have hIX : Integrable (fun ω => g (X ω)) P := by
    refine (integrable_const (x - a)).mono' hmX.aestronglyMeasurable ?_
    filter_upwards with ω
    rw [Real.norm_eq_abs]
    exact abs_clamp_le hax _
  have hIY : Integrable (fun ω => g (Y ω)) P := by
    refine (integrable_const (x - a)).mono' hmY.aestronglyMeasurable ?_
    filter_upwards with ω
    rw [Real.norm_eq_abs]
    exact abs_clamp_le hax _
  rw [← integral_sub hIX hIY]
  calc |∫ ω, (g (X ω) - g (Y ω)) ∂P|
      ≤ ∫ ω, |g (X ω) - g (Y ω)| ∂P := abs_integral_le_integral_abs
    _ ≤ ∫ ω, |X ω - Y ω| ∂P := by
        refine integral_mono_ae (hIX.sub hIY).abs ((hIntX.sub hIntY).abs) ?_
        filter_upwards with ω
        exact abs_clamp_sub_clamp_le a x (X ω) (Y ω)

end Threshold

/-! ### §57b  The radius of a Gaussian vector

`Var(‖Z‖) ≤ 1` with a constant independent of the dimension is the one nontrivial input, and it is
a two-line consequence of the Gaussian Poincaré inequality §32/§39 already proves.  Everything else
here is bookkeeping around `E‖Z‖² = d`. -/

noncomputable section GaussianRadius

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- `‖Z‖` is integrable, from `‖x‖ ≤ 1 + ‖x‖²` and `integrable_normSq_gaussianE` (§39). -/
theorem integrable_norm_gaussianE : Integrable (fun x : E => ‖x‖) (gaussianE E) := by
  refine ((integrable_const (1:ℝ)).add (integrable_normSq_gaussianE (E := E))).mono'
    (by fun_prop) ?_
  filter_upwards with x
  simp only [Pi.add_apply]
  rw [Real.norm_eq_abs, abs_of_nonneg (norm_nonneg x)]
  nlinarith [norm_nonneg x, sq_nonneg (‖x‖ - 1)]

/-- **`Var(‖Z‖) ≤ 1`** for the standard Gaussian on any space of dimension at least two, with
constant `1` independent of the dimension. From the Gaussian Poincaré inequality
(`gaussPoincare_gaussianE`, §39) applied to `f = ‖·‖` with the constant slope bound `G = 1/2`:
`|‖x‖ − ‖y‖| ≤ ‖x − y‖ = ‖x − y‖·(1/2 + 1/2)` is a valid two-point bound, so
`Var(‖·‖) ≤ ∫ (2·(1/2))² = 1`.  Does not bound `Var(‖Z‖²)`, and gives no matching lower bound on
the variance itself. -/
theorem variance_norm_gaussianE_le (hd : 2 ≤ Module.finrank ℝ E) :
    ProbabilityTheory.variance (fun x : E => ‖x‖) (gaussianE E) ≤ 1 := by
  have hGP := gaussPoincare_gaussianE (E := E)
  have hf : MemLp (fun x : E => ‖x‖) 2 (gaussianE E) := by
    rw [memLp_two_iff_integrable_sq (by fun_prop)]
    simpa using integrable_normSq_gaussianE (E := E)
  have hint : Integrable (fun _ : E => ((1:ℝ)/2) ^ 2) (gaussianE E) := integrable_const _
  have hbound := hGP.bound hd (fun x : E => ‖x‖) (fun _ => (1:ℝ)/2)
    (continuousOn_const) (fun x y _ _ => by
      have h1 := abs_norm_sub_norm_le x y
      calc |‖x‖ - ‖y‖| ≤ ‖x - y‖ := h1
        _ = ‖x - y‖ * (1/2 + 1/2) := by ring) hf hint
  have heq : (∫ _x : E, (2 * ((1:ℝ)/2)) ^ 2 ∂(gaussianE E)) = 1 := by simp
  linarith [hbound, heq]

/-- **`E‖Z‖ ≥ √(d − 1)`**, from `(E‖Z‖)² = E‖Z‖² − Var(‖Z‖) = d − Var(‖Z‖) ≥ d − 1`
(`variance_norm_gaussianE_le`, `integral_normSq_gaussianE`) and `E‖Z‖ ≥ 0`.  Gives no matching
upper bound on `E‖Z‖`; that direction is not needed by the coupling argument this feeds. -/
theorem le_integral_norm_gaussianE (hd : 2 ≤ Module.finrank ℝ E) :
    Real.sqrt ((Module.finrank ℝ E : ℝ) - 1) ≤ ∫ x, ‖x‖ ∂(gaussianE E) := by
  have hf : MemLp (fun x : E => ‖x‖) 2 (gaussianE E) := by
    rw [memLp_two_iff_integrable_sq (by fun_prop)]
    simpa using integrable_normSq_gaussianE (E := E)
  have hvar := variance_norm_gaussianE_le hd
  have hveq := ProbabilityTheory.variance_eq_sub hf
  simp only [Pi.pow_apply] at hveq
  rw [integral_normSq_gaussianE] at hveq
  have hnn : 0 ≤ ∫ x, ‖x‖ ∂(gaussianE E) := integral_nonneg (fun x => norm_nonneg _)
  have hkey : (Module.finrank ℝ E : ℝ) - 1 ≤ (∫ x, ‖x‖ ∂(gaussianE E)) ^ 2 := by linarith
  calc Real.sqrt ((Module.finrank ℝ E : ℝ) - 1)
      ≤ Real.sqrt ((∫ x, ‖x‖ ∂(gaussianE E)) ^ 2) := Real.sqrt_le_sqrt hkey
    _ = ∫ x, ‖x‖ ∂(gaussianE E) := Real.sqrt_sq hnn

/-- `c − ‖Z‖` is square-integrable, so Cauchy-Schwarz may be applied to it. -/
theorem memLp_two_const_sub_norm (c : ℝ) : MemLp (fun x : E => c - ‖x‖) 2 (gaussianE E) := by
  rw [memLp_two_iff_integrable_sq (by fun_prop)]
  have hIn : Integrable (fun x : E => ‖x‖) (gaussianE E) := integrable_norm_gaussianE
  have h : Integrable (fun x : E => (c ^ 2 - 2 * c * ‖x‖) + ‖x‖ ^ 2) (gaussianE E) :=
    ((integrable_const (c ^ 2)).sub (hIn.const_mul (2 * c))).add
      (integrable_normSq_gaussianE (E := E))
  exact h.congr (Filter.Eventually.of_forall fun x => by ring)

/-- `E[(c − ‖Z‖)²] = c² − 2c·E‖Z‖ + d`: expanding the square and using `E‖Z‖² = d` (§39). -/
theorem integral_sq_const_sub_norm (c : ℝ) :
    ∫ x : E, (c - ‖x‖) ^ 2 ∂(gaussianE E)
      = c ^ 2 - 2 * c * (∫ x : E, ‖x‖ ∂(gaussianE E)) + (Module.finrank ℝ E : ℝ) := by
  have hIn : Integrable (fun x : E => ‖x‖) (gaussianE E) := integrable_norm_gaussianE
  have h1 : ∫ x : E, (c - ‖x‖) ^ 2 ∂(gaussianE E)
      = ∫ x : E, ((c ^ 2 - 2 * c * ‖x‖) + ‖x‖ ^ 2) ∂(gaussianE E) :=
    integral_congr_ae (Filter.Eventually.of_forall fun x => by ring)
  have h2 : ∫ x : E, ((c ^ 2 - 2 * c * ‖x‖) + ‖x‖ ^ 2) ∂(gaussianE E)
      = (∫ x : E, (c ^ 2 - 2 * c * ‖x‖) ∂(gaussianE E)) + ∫ x : E, ‖x‖ ^ 2 ∂(gaussianE E) :=
    integral_add ((integrable_const (c ^ 2)).sub (hIn.const_mul (2 * c)))
      (integrable_normSq_gaussianE (E := E))
  have h3 : ∫ x : E, (c ^ 2 - 2 * c * ‖x‖) ∂(gaussianE E)
      = (∫ _x : E, c ^ 2 ∂(gaussianE E)) - ∫ x : E, (2 * c * ‖x‖) ∂(gaussianE E) :=
    integral_sub (integrable_const (c ^ 2)) (hIn.const_mul (2 * c))
  rw [h1, h2, h3, integral_const_mul, integral_const, integral_normSq_gaussianE,
    measureReal_def, measure_univ, ENNReal.toReal_one, one_smul]

/-- The arithmetic behind the radius estimate, isolated from any measure theory: if `m` is a
number with `m² ≤ n`, `n − m² ≤ 1` and `√(n − 1) ≤ m`, then `2n − 2√n·m ≤ 1 + 1/(4(n−1))`.  The
two summands are `n − m²` (the variance, bounded by `1`) and `(√n − m)²`, and the latter is at
most `(√n − √(n−1))² = 1/(√n + √(n−1))² ≤ 1/(4(n−1))`. -/
theorem two_mul_sub_two_mul_sqrt_mul_le {n m : ℝ} (hn2 : 2 ≤ n) (hm0 : 0 ≤ m) (hvnn : m ^ 2 ≤ n)
    (hvar : n - m ^ 2 ≤ 1) (hmr : Real.sqrt (n - 1) ≤ m) :
    2 * n - 2 * Real.sqrt n * m ≤ 1 + 1 / (4 * (n - 1)) := by
  have hs2 : Real.sqrt n ^ 2 = n := Real.sq_sqrt (by linarith)
  have hr2 : Real.sqrt (n - 1) ^ 2 = n - 1 := Real.sq_sqrt (by linarith)
  have hr0 : 0 ≤ Real.sqrt (n - 1) := Real.sqrt_nonneg _
  have hs0 : 0 ≤ Real.sqrt n := Real.sqrt_nonneg _
  have hr1 : 1 ≤ Real.sqrt (n - 1) := by nlinarith
  have hrs : Real.sqrt (n - 1) ≤ Real.sqrt n := Real.sqrt_le_sqrt (by linarith)
  have hms : m ≤ Real.sqrt n := by nlinarith
  have hsm : 0 ≤ Real.sqrt n - m := by linarith
  have hkey : 2 * Real.sqrt (n - 1) * (Real.sqrt n - m) ≤ 1 := by nlinarith
  have hknn : 0 ≤ 2 * Real.sqrt (n - 1) * (Real.sqrt n - m) := by positivity
  have hsq : (Real.sqrt n - m) ^ 2 * (4 * (n - 1)) ≤ 1 := by nlinarith
  have hpos : (0:ℝ) < 4 * (n - 1) := by linarith
  have hdiv : (Real.sqrt n - m) ^ 2 ≤ 1 / (4 * (n - 1)) := by
    rw [le_div_iff₀ hpos]; exact hsq
  nlinarith [hdiv, hvar, hs2]

/-- **`E[(√d − ‖Z‖)²] ≤ 1 + 1/(4(d−1))`.**  The decomposition
`E[(√d − ‖Z‖)²] = Var(‖Z‖) + (√d − E‖Z‖)²`, with `Var(‖Z‖) ≤ 1` dimension-free
(`variance_norm_gaussianE_le`) and `√(d−1) ≤ E‖Z‖ ≤ √d` (`le_integral_norm_gaussianE` and
`E‖Z‖² = d`).  The bound tends to `1`, not to `0`: the radius does not concentrate at scale
better than `O(1)`, and nothing here claims it does. -/
theorem integral_sq_sqrt_finrank_sub_norm_le (hd : 2 ≤ Module.finrank ℝ E) :
    ∫ x : E, (Real.sqrt (Module.finrank ℝ E : ℝ) - ‖x‖) ^ 2 ∂(gaussianE E)
      ≤ 1 + 1 / (4 * ((Module.finrank ℝ E : ℝ) - 1)) := by
  have hn2 : (2:ℝ) ≤ (Module.finrank ℝ E : ℝ) := by exact_mod_cast hd
  have hf : MemLp (fun x : E => ‖x‖) 2 (gaussianE E) := by
    rw [memLp_two_iff_integrable_sq (by fun_prop)]
    simpa using integrable_normSq_gaussianE (E := E)
  have hveq := ProbabilityTheory.variance_eq_sub hf
  simp only [Pi.pow_apply] at hveq
  rw [integral_normSq_gaussianE] at hveq
  have hvar := variance_norm_gaussianE_le (E := E) hd
  have hvnn := ProbabilityTheory.variance_nonneg (fun x : E => ‖x‖) (gaussianE E)
  have hm0 : 0 ≤ ∫ x : E, ‖x‖ ∂(gaussianE E) := integral_nonneg fun x => norm_nonneg _
  have hmr := le_integral_norm_gaussianE (E := E) hd
  rw [integral_sq_const_sub_norm, Real.sq_sqrt (by linarith)]
  have := two_mul_sub_two_mul_sqrt_mul_le (n := (Module.finrank ℝ E : ℝ))
    (m := ∫ x : E, ‖x‖ ∂(gaussianE E)) hn2 hm0 (by linarith) (by linarith) hmr
  linarith

end GaussianRadius

/-! ### §57c  The coupling, and the leg it proves

`u = Z/‖Z‖` for `Z ~ gaussianE (Ed d)` is uniform on the sphere (§26's `gaussianE_map_dir`), so
`u_i` has law `sphericalCoordLaw d` (§26's `gaussianE_map_dir_coord`); and `Z_i/√d` has law
`N(0, 1/d)` (`gaussianE_map_ofLp` and `gaussianReal_map_div_sqrt` above).  That pair is the
coupling.  Its `L¹` cost is estimated by Cauchy-Schwarz and §57b, and §57a turns the cost into a
bound on `cdfGapInt`. -/

noncomputable section CouplingLeg

open Orthogonal

/-- Cauchy-Schwarz for the integral, in the `Real.sqrt` spelling: Mathlib's Hölder inequality
`integral_mul_le_Lp_mul_Lq_of_nonneg` at the conjugate pair `(2, 2)`, with the `rpow`s converted
to squares and square roots. -/
theorem integral_abs_mul_le_sqrt_mul_sqrt {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    {f g : Ω → ℝ} (hf : MemLp f 2 P) (hg : MemLp g 2 P) :
    ∫ ω, |f ω| * |g ω| ∂P
      ≤ Real.sqrt (∫ ω, f ω ^ 2 ∂P) * Real.sqrt (∫ ω, g ω ^ 2 ∂P) := by
  have h2 : ENNReal.ofReal (2:ℝ) = 2 := by
    rw [show (2:ℝ) = ((2:ℕ):ℝ) by norm_num, ENNReal.ofReal_natCast]; rfl
  have hf' : MemLp (fun ω => |f ω|) (ENNReal.ofReal (2:ℝ)) P := by rw [h2]; exact hf.abs
  have hg' : MemLp (fun ω => |g ω|) (ENNReal.ofReal (2:ℝ)) P := by rw [h2]; exact hg.abs
  have hCS := integral_mul_le_Lp_mul_Lq_of_nonneg (μ := P) (p := 2) (q := 2)
    Real.HolderConjugate.two_two (f := fun ω => |f ω|) (g := fun ω => |g ω|)
    (Filter.Eventually.of_forall fun ω => abs_nonneg _)
    (Filter.Eventually.of_forall fun ω => abs_nonneg _) hf' hg'
  simp only [Real.rpow_two, sq_abs] at hCS
  rwa [← Real.sqrt_eq_rpow, ← Real.sqrt_eq_rpow] at hCS

/-- `Ed d` is nontrivial as soon as it has a coordinate; §26's `measurable_dir` and §30's
`norm_dir_le_one` are stated on a nontrivial space, so this is the instance they need. -/
theorem nontrivial_Ed {d : ℕ} (i : Fin d) : Nontrivial (Ed d) :=
  Module.nontrivial_of_finrank_pos (R := ℝ) (by simpa [Ed] using i.pos)

/-- A coordinate of the direction of a Gaussian vector is bounded by `1`. -/
theorem abs_dir_coord_le_one {d : ℕ} (i : Fin d) (z : Ed d) : |(dir z).ofLp i| ≤ 1 := by
  haveI := nontrivial_Ed i
  exact le_trans (by simpa using PiLp.norm_apply_le (dir z) i) (norm_dir_le_one z)

/-- A coordinate of the direction of a Gaussian vector is measurable. -/
theorem measurable_dir_coord {d : ℕ} (i : Fin d) :
    Measurable (fun z : Ed d => (dir z).ofLp i) := by
  haveI := nontrivial_Ed i
  exact ((measurable_pi_apply i).comp (WithLp.measurable_ofLp 2 _)).comp measurable_dir

/-- `E[u_i²] = 1/d` for `u = Z/‖Z‖`: the pushforward of `gaussianE (Ed d)` along
`z ↦ (dir z)_i` is `sphericalCoordLaw d` (§26), whose second moment is `1/d` (§4). -/
theorem integral_sq_dir_coord {d : ℕ} (hd : 2 ≤ d) (i : Fin d) :
    ∫ z, ((dir z).ofLp i) ^ 2 ∂(gaussianE (Ed d)) = 1 / (d : ℝ) := by
  haveI : Nontrivial (Ed d) := Module.nontrivial_of_finrank_pos (R := ℝ) (by
    have : (0 : ℕ) < d := by omega
    simpa [Ed] using this)
  rw [← integral_sq_sphericalCoordLaw hd, ← gaussianE_map_dir_coord hd i,
    integral_map (measurable_dir_coord i).aemeasurable (by fun_prop)]

/-- **The `L¹` cost of the coupling.**  `E|u_i − Z_i/√d| ≤ √(1 + 1/(4(d−1)))/d`.  The
coordinatewise identity `u_i − Z_i/√d = u_i·(√d − ‖Z‖)/√d` is exact (both sides vanish at
`Z = 0`), Cauchy-Schwarz splits the expectation, `E[u_i²] = 1/d` by Lemma 1 and
`E[(√d − ‖Z‖)²] ≤ 1 + 1/(4(d−1))` by §57b.  The rate is `Θ(1/d)`, not `Θ(1/√d)`: the gain over
the crude bound `|Z_i| ≤ ‖Z‖` is exactly the factor `√(E[u_i²]) = 1/√d`.  This is an `L¹`
statement about one coupling; it is not a bound on the Kolmogorov-Smirnov distance of the two
laws, and no such bound follows from it. -/
theorem integral_abs_dir_coord_sub_coord_div_sqrt_le {d : ℕ} (hd : 2 ≤ d) (i : Fin d) :
    ∫ z, |(dir z).ofLp i - z.ofLp i / Real.sqrt (d : ℝ)| ∂(gaussianE (Ed d))
      ≤ Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) := by
  haveI : Nontrivial (Ed d) := Module.nontrivial_of_finrank_pos (R := ℝ) (by
    have : (0 : ℕ) < d := by omega
    simpa [Ed] using this)
  have hdn : Module.finrank ℝ (Ed d) = d := by simp [Ed]
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have hs : (0:ℝ) < Real.sqrt (d:ℝ) := Real.sqrt_pos.mpr hd0
  have hmf : Measurable (fun z : Ed d => (dir z).ofLp i) := measurable_dir_coord i
  -- the pointwise factorisation of the coupling error
  have hpt : ∀ z : Ed d, |(dir z).ofLp i - z.ofLp i / Real.sqrt (d:ℝ)|
      = (Real.sqrt (d:ℝ))⁻¹ * (|(dir z).ofLp i| * |Real.sqrt (d:ℝ) - ‖z‖|) := by
    intro z
    have hfac : (dir z).ofLp i - z.ofLp i / Real.sqrt (d:ℝ)
        = (Real.sqrt (d:ℝ))⁻¹ * ((dir z).ofLp i * (Real.sqrt (d:ℝ) - ‖z‖)) := by
      rcases eq_or_ne z 0 with h | h
      · subst h; simp [dir]
      · have hz : ‖z‖ ≠ 0 := norm_ne_zero_iff.mpr h
        show ‖z‖⁻¹ * z.ofLp i - z.ofLp i / Real.sqrt (d:ℝ)
          = (Real.sqrt (d:ℝ))⁻¹ * (‖z‖⁻¹ * z.ofLp i * (Real.sqrt (d:ℝ) - ‖z‖))
        field_simp
    rw [hfac, abs_mul, abs_mul, abs_of_nonneg (by positivity : (0:ℝ) ≤ (Real.sqrt (d:ℝ))⁻¹)]
  -- square-integrability of the two factors
  have hMf : MemLp (fun z : Ed d => (dir z).ofLp i) 2 (gaussianE (Ed d)) := by
    rw [memLp_two_iff_integrable_sq hmf.aestronglyMeasurable]
    refine (integrable_const (1:ℝ)).mono' (hmf.pow_const 2).aestronglyMeasurable ?_
    filter_upwards with z
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _), ← sq_abs]
    nlinarith [abs_dir_coord_le_one i z, abs_nonneg ((dir z).ofLp i)]
  have hMg : MemLp (fun z : Ed d => Real.sqrt (d:ℝ) - ‖z‖) 2 (gaussianE (Ed d)) :=
    memLp_two_const_sub_norm _
  -- the two second moments
  have hA : ∫ z, ((dir z).ofLp i) ^ 2 ∂(gaussianE (Ed d)) = 1 / (d : ℝ) :=
    integral_sq_dir_coord hd i
  have hB : ∫ z : Ed d, (Real.sqrt (d:ℝ) - ‖z‖) ^ 2 ∂(gaussianE (Ed d))
      ≤ 1 + 1 / (4 * ((d : ℝ) - 1)) := by
    have h := integral_sq_sqrt_finrank_sub_norm_le (E := Ed d) (by rw [hdn]; exact hd)
    rwa [hdn] at h
  have hsqrtA : Real.sqrt (1 / (d:ℝ)) = (Real.sqrt (d:ℝ))⁻¹ := by
    rw [one_div, Real.sqrt_inv]
  have hBnn : (0:ℝ) ≤ ∫ z : Ed d, (Real.sqrt (d:ℝ) - ‖z‖) ^ 2 ∂(gaussianE (Ed d)) :=
    integral_nonneg fun z => sq_nonneg _
  calc ∫ z, |(dir z).ofLp i - z.ofLp i / Real.sqrt (d:ℝ)| ∂(gaussianE (Ed d))
      = ∫ z, (Real.sqrt (d:ℝ))⁻¹
          * (|(dir z).ofLp i| * |Real.sqrt (d:ℝ) - ‖z‖|) ∂(gaussianE (Ed d)) :=
        integral_congr_ae (Filter.Eventually.of_forall hpt)
    _ = (Real.sqrt (d:ℝ))⁻¹
          * ∫ z, |(dir z).ofLp i| * |Real.sqrt (d:ℝ) - ‖z‖| ∂(gaussianE (Ed d)) :=
        integral_const_mul _ _
    _ ≤ (Real.sqrt (d:ℝ))⁻¹
          * (Real.sqrt (∫ z, ((dir z).ofLp i) ^ 2 ∂(gaussianE (Ed d)))
            * Real.sqrt (∫ z : Ed d, (Real.sqrt (d:ℝ) - ‖z‖) ^ 2 ∂(gaussianE (Ed d)))) := by
        refine mul_le_mul_of_nonneg_left ?_ (by positivity)
        exact integral_abs_mul_le_sqrt_mul_sqrt _ hMf hMg
    _ ≤ (Real.sqrt (d:ℝ))⁻¹
          * ((Real.sqrt (d:ℝ))⁻¹ * Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1)))) := by
        refine mul_le_mul_of_nonneg_left ?_ (by positivity)
        rw [hA, hsqrtA]
        exact mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hB) (by positivity)
    _ = Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) := by
        rw [← mul_assoc, ← mul_inv, Real.mul_self_sqrt hd0.le, inv_mul_eq_div]

/-- **The spherical-versus-Gaussian leg, in `cdfGapInt` form.**  For every `d ≥ 2` and every pair
`a, x` of reals,

  `|cdfGapInt (sphericalCoordLaw d) (N(0, 1/d)) a x| ≤ √(1 + 1/(4(d−1)))/d`.

The coupling is `u = Z/‖Z‖` against `Z/√d` for a single Gaussian vector `Z ~ gaussianE (Ed d)`;
§57a converts its `L¹` cost into this bound, and §57c estimates the cost.  The bound is uniform in
`a` and `x`, is `Θ(1/d)` in the dimension, and needs no support hypothesis on either law.

What it is **not**.  It is not a Kolmogorov-Smirnov bound: `|cdf μ − cdf ν|` is not bounded here,
and no `η` for §55 or §56 is produced, so `Threshold.abs_hadRate_sub_rhoSq_le` stays exactly as
conditional as it was.  It is one of the two legs the triangle inequality would need: the other
leg, `coordLaw H v i` against the same Gaussian, is §44's business and is not composed with this
one anywhere.  Rewiring §55's threshold sum so that it consumes a uniform `cdfGapInt` bound rather
than `η·(b − a)` is not carried out here, so nothing downstream is improved yet. -/
theorem abs_cdfGapInt_sphericalCoordLaw_gaussian_le {d : ℕ} (hd : 2 ≤ d) (a x : ℝ) :
    |Threshold.cdfGapInt (sphericalCoordLaw d)
        (gaussianReal 0 (Real.toNNReal (1 / (d : ℝ)))) a x|
      ≤ Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) := by
  haveI := isProbabilityMeasure_sphericalCoordLaw hd
  haveI : Nontrivial (Ed d) := Module.nontrivial_of_finrank_pos (R := ℝ) (by
    have : (0 : ℕ) < d := by omega
    simpa [Ed] using this)
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hrhs : (0:ℝ) ≤ Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) := by positivity
  rcases le_or_gt a x with hax | hax
  · have hi : (0:ℕ) < d := by omega
    set i : Fin d := ⟨0, hi⟩ with hidef
    have hmX : Measurable (fun z : Ed d => (dir z).ofLp i) := measurable_dir_coord i
    have hcoord : Measurable (fun z : Ed d => z.ofLp i) :=
      (measurable_pi_apply i).comp (WithLp.measurable_ofLp 2 _)
    have hmY : Measurable (fun z : Ed d => z.ofLp i / Real.sqrt (d:ℝ)) := hcoord.div_const _
    have hXlaw : Measure.map (fun z : Ed d => (dir z).ofLp i) (gaussianE (Ed d))
        = sphericalCoordLaw d := gaussianE_map_dir_coord hd i
    have hYlaw : Measure.map (fun z : Ed d => z.ofLp i / Real.sqrt (d:ℝ)) (gaussianE (Ed d))
        = gaussianReal 0 (Real.toNNReal (1 / (d : ℝ))) := by
      have h1 : (fun z : Ed d => z.ofLp i / Real.sqrt (d:ℝ))
          = (fun t : ℝ => t / Real.sqrt (d:ℝ)) ∘ (fun z : Ed d => z.ofLp i) := rfl
      have hdiv : Measurable (fun t : ℝ => t / Real.sqrt (d:ℝ)) := measurable_id.div_const _
      rw [h1, ← Measure.map_map hdiv hcoord, gaussianE_map_ofLp i,
        gaussianReal_map_div_sqrt hd]
    have hIX : Integrable (fun z : Ed d => (dir z).ofLp i) (gaussianE (Ed d)) := by
      refine (integrable_const (1:ℝ)).mono' hmX.aestronglyMeasurable ?_
      filter_upwards with z
      rw [Real.norm_eq_abs]
      exact abs_dir_coord_le_one i z
    have hIY : Integrable (fun z : Ed d => z.ofLp i / Real.sqrt (d:ℝ)) (gaussianE (Ed d)) := by
      refine ((integrable_norm_gaussianE (E := Ed d)).div_const
        (Real.sqrt (d:ℝ))).mono' hmY.aestronglyMeasurable ?_
      filter_upwards with z
      rw [Real.norm_eq_abs, abs_div, abs_of_nonneg (Real.sqrt_nonneg _)]
      gcongr
      simpa using PiLp.norm_apply_le z i
    calc |Threshold.cdfGapInt (sphericalCoordLaw d)
            (gaussianReal 0 (Real.toNNReal (1 / (d : ℝ)))) a x|
        ≤ ∫ z, |(dir z).ofLp i - z.ofLp i / Real.sqrt (d:ℝ)| ∂(gaussianE (Ed d)) :=
          Threshold.abs_cdfGapInt_le_integral_abs_sub_of_coupling _ hmX.aemeasurable
            hmY.aemeasurable hIX hIY hXlaw hYlaw hax
      _ ≤ Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) :=
          integral_abs_dir_coord_sub_coord_div_sqrt_le hd i
  · have hempty : Set.Ioc a x = (∅ : Set ℝ) := Set.Ioc_eq_empty (by linarith)
    rw [Threshold.cdfGapInt, hempty, Measure.restrict_empty, integral_zero_measure, abs_zero]
    exact hrhs

end CouplingLeg
