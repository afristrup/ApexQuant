
/-! ## §58  The deployed-versus-Gaussian leg, by Fourier inversion of the integrated CDF gap

§57 builds one leg of the comparison the paper needs, `sphericalCoordLaw d` against `N(0, 1/d)`,
in `cdfGapInt` form and by coupling.  This section builds the other, `coordLaw H v i` against the
same Gaussian, and it cannot use a coupling: the deployed coordinate law is supported on `2^d`
points and this development exhibits no coupling of it with a Gaussian.  What it does have is the
characteristic-function comparison of §40 and §44.  The route is therefore Fourier inversion, but
of the **once-integrated** CDF difference rather than of the CDF difference itself, which is what
lets a `u^{-2}` weight appear where Esseen's route carries `u^{-1}`.

Six blocks.  §58a and §58b are independent; §58a′ refines §58a; §58c composes §58a with §58b,
and §58c′, §58d and §58d′ are the same composition with one or both constants sharpened.

* **§58a, the kernel and the inversion.**  `dirConst = ∫₀^∞ (1 − cos u)/u² du` is finite and
  `∫₀^∞ (1 − cos(su))/u² du = dirConst · |s|` (`integral_kernel`).  Its exact value is the
  Dirichlet integral `π/2`, and that **is** now formalised (`dirConst_eq`), by reading §43's Fejér
  density at half-width `1/(2π)` and halving by evenness; Mathlib carries `Real.sinc` but no
  integral of it, so the value is not imported.  The earlier lower bound `9/8 ≤ dirConst`
  (`dirConst_ge`, from `sin x > x − x³/6` on `(0, 3]`) is retained, and the §58c statements are
  still phrased through it, so their constants are loose by `(π/2)/(9/8) ≈ 1.40`; §58d repeats
  them with the exact value.  On top of that,
  `abs_cdfGapInt_le_of_charFun` states

    `|cdfGapInt μ ν a x| ≤ (∫₀^∞ ‖φ_μ(u) − φ_ν(u)‖/u² du) / dirConst`

  for any two probability laws with finite first absolute moments, uniformly in `a` and `x` and
  with no support hypothesis.  The identity behind it is
  `(x − t)⁺ − (a − t)⁺ = (x − a)/2 + (|x − t| − |a − t|)/2`, whose constant part cancels between
  the two laws; the mean absolute deviations `∫|y − t| dρ` are then read off the kernel by Tonelli
  (`integrableOn_and_integral_kernel_measure`).  No equality of means is used: what makes the
  `u → 0` end harmless is the cancellation of the two `1`s in `1 − cos`.  A factor `2` is lost at
  `|G(x) − G(a)| ≤ |G(x)| + |G(a)|`.
* **§58a′, the same inversion with that cancellation kept.**  The Fourier representation gives
  `G(x) − G(a)` directly, and by `cos α − cos β = 2 sin((α+β)/2) sin((β−α)/2)` the kernel for the
  difference carries a factor `2 sin((x − a)u/2)`.  `abs_cdfGapInt_le_of_charFun_sin` keeps it:

    `|cdfGapInt μ ν a x| ≤ (∫₀^∞ |sin((x − a)u/2)|·‖φ_μ(u) − φ_ν(u)‖/u² du)/dirConst`,

  which implies `abs_cdfGapInt_le_of_charFun` (`integral_sinWeight_le`) and is strictly better at
  every `u` where `|sin((x − a)u/2)| < 1`.  It does **not** improve the constant uniformly in `a`
  and `x`: for a weight concentrated near one frequency `u₀` the choice `x − a = π/u₀` makes the
  sine factor `≈ 1` across the band, and such weights are realised, so the factor `2` above is
  **sharp** as a statement about arbitrary pairs of laws.  At the particular weight §58b produces
  the supremum is `0.72` of the unweighted integral at `d = 512` and a spread direction, a factor
  `1.39` rather than `2`, and collecting it needs `sup_{c>0} c·∫_c^∞|sin w|/w² dw = 0.7787… < 1`,
  which is **not** proved here.  Both numbers are numerical.  §58c and §58c′ below are therefore
  still proved from `abs_cdfGapInt_le_of_charFun` and carry the factor `2`.
* **§58b, the two-range estimate.**  Split at `√d` and at `√d/M`, where `M` is any bound on the
  coordinates of `v`.  Below `√d/M` §44's damped bound
  `‖φ_μ − φ_ν‖ ≤ (u⁴ ∑ⱼvⱼ⁴/(2d²)) · e^{−u²‖v‖²/(4d)}` applies, and `e^{−z} ≤ 2/z²` makes its tail
  integrable; beyond `√d/M` this development knows nothing about the atomic law's characteristic
  function and the trivial `‖φ_μ − φ_ν‖ ≤ 2` is used.  The two ranges integrate to
  `(97/6 · ∑ⱼvⱼ⁴ + 2M)/√d` (`integral_gapWeight_le`).  The constant `2` in `e^{−z} ≤ 2/z²` is
  very lossy: `max_{z>0} z²e^{−z} = 4e^{−2} = 0.5413…`, so `e^{−z} ≤ (5/9)/z²` holds
  (`exp_neg_le_five_ninths_div_sq`) and is within `2.6%` of the best possible.  Substituting it
  gives `(83/18 · ∑ⱼvⱼ⁴ + 2M)/√d` (`integral_gapWeight_le_sharp`), a factor `3.51` on the
  spikiness term.  The `2M` term is **not** improved by it, and at a spread direction that term
  dominates.
* **§58c, leg 1.**  Composing the two, with `M = (∑ⱼvⱼ⁴)^{1/4}` when no coordinate bound is
  supplied, gives `abs_cdfGapInt_coordLaw_gaussian_le`:

    `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ (15 · ∑ⱼvⱼ⁴ + 2 · (∑ⱼvⱼ⁴)^{1/4})/√d`

  for `d ≥ 2`, a flat `H`, a **unit** direction `v`, any coordinate `i` and every pair `a`, `x`.
* **§58c′, leg 1 with the sharp damping constant.**  The same three statements over
  `integral_gapWeight_le_sharp`, giving `41/10` in place of `15`
  (`abs_cdfGapInt_coordLaw_gaussian_le_sharp` and its two companions).  The `2 · (∑ⱼvⱼ⁴)^{1/4}` is
  unchanged.  At a spread direction the composed bound would read `(41/10/√d + 2)/d` in place of
  `(15/√d + 2)/d`; nothing in §59 is rewired to use it here, and the originals are kept.
* **§58d, leg 1 with the constant evaluated.**  The same three statements, primed, with `π/2` in
  place of `9/8`: `15` becomes `21/2` and `2` becomes `13/10`.  Nothing else changes, and the
  unprimed statements are retained unchanged.
* **§58d′, both constants at once.**  §58b's sharp damping and §58d's exact kernel constant are
  independent, and taking both gives `abs_cdfGapInt_coordLaw_gaussian_le_sharp'`:

    `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ (3 · ∑ⱼvⱼ⁴ + (13/10)·(∑ⱼvⱼ⁴)^{1/4})/√d`,

  and at a spread direction `(3/√d + 13/10)/d` in place of `(15/√d + 2)/d`, a factor `1.86` at
  `d = 512`.  This is the smallest constant proved here.  All twelve leg-1 statements are kept;
  none weakens another, and §59 is not rewired to any of them by this section.

**What this does not do.**  It is not a Kolmogorov-Smirnov bound: `|cdf μ − cdf ν|` is bounded
nowhere here, no `η` is produced, and `Threshold.abs_hadRate_sub_rhoSq_le` in its sup-norm form is
untouched by it.  It is one leg only: the second law is the Gaussian `N(0, 1/d)` and not
`sphericalCoordLaw d`, and nothing here composes this bound with §57's
`abs_cdfGapInt_sphericalCoordLaw_gaussian_le` by the triangle inequality.  The rate is **not** the
`d^{-3/2}` that the damped range alone would give: the `2M/√d` term, coming from the range
`|u| > √d/M` on which the atomic law's characteristic function returns near `1`, dominates, and
the lattice structure that would control that range is not exploited.  The constants are not
claimed optimal: the §58c statements enter `dirConst` only through `9/8 ≤ dirConst` and use the
crude `e^{−z} ≤ 2/z²`, and even §58d′, which repairs both, still carries the factor `2` lost at
`|G(x) − G(a)| ≤ |G(x)| + |G(a)|`.  That factor is **sharp** for arbitrary pairs of laws (§58a′),
so it is not slack there, but at the weight §58b produces it is worth about `1.39` and collecting
even that needs an oscillation estimate this development does not have.  The two largest pieces
of slack left are therefore the `M/√d` term, where `‖φ_μ − φ_ν‖ ≤ 2` is used with no information
about the atomic law at all, and the gap between `40/9` and the `√π` that the damped range really
contributes.  Nothing here is asymptotic, and no claim is made about the deployed rotation
attaining the Haar rate `ρ²`. -/

noncomputable section FourierLeg

/-! ### §58a  The kernel `(1 − cos u)/u²` and the inversion bound

Nothing in this block mentions a codebook, a quantiser, a rotation or a dimension: it is about two
arbitrary probability laws on `ℝ` with finite first absolute moments.  `dirConst` is evaluated
here: the Dirichlet integral `∫₀^∞ (1 − cos u)/u² du = π/2` is `dirConst_eq`, proved from §43 and
not imported from Mathlib, which does not contain it. -/

namespace FourierGap

/-- **The kernel constant** `∫₀^∞ (1 − cos u)/u² du`.  It is finite
(`integrableOn_kernel'`) and its exact value is the Dirichlet integral `π/2`, proved as
`dirConst_eq`.  The weaker `9/8 ≤ dirConst` (`dirConst_ge`) is retained because the §58c
statements are phrased through it; those are therefore at most `(π/2)/(9/8) ≈ 1.40` times the
sharp constant, and §58d repeats them with `dirConst_eq`.  Evaluating this one number does not
make any downstream constant sharp: see `dirConst_eq` for what is still lost. -/
def dirConst : ℝ := ∫ u in Set.Ioi (0:ℝ), (1 - Real.cos u) / u ^ 2

theorem oneSubCos_nonneg (u : ℝ) : 0 ≤ 1 - Real.cos u := by linarith [Real.cos_le_one u]

theorem kernel_nonneg (s u : ℝ) : 0 ≤ (1 - Real.cos (s * u)) / u ^ 2 :=
  div_nonneg (oneSubCos_nonneg _) (sq_nonneg u)

theorem measurable_kernel (s : ℝ) :
    Measurable fun u : ℝ => (1 - Real.cos (s * u)) / u ^ 2 := by fun_prop

theorem rpow_neg_two {u : ℝ} (hu : 0 < u) : u ^ (-2 : ℝ) = 1 / u ^ 2 := by
  rw [Real.rpow_neg hu.le, show ((2:ℝ)) = ((2:ℕ):ℝ) by norm_num, Real.rpow_natCast, one_div]

/-- `1 − cos u ≤ u²/2`, so the kernel is bounded near `0`. -/
theorem oneSubCos_le (u : ℝ) : 1 - Real.cos u ≤ u ^ 2 / 2 := by
  linarith [Real.one_sub_sq_div_two_le_cos (x := u)]

theorem integrableOn_kernel_one (s : ℝ) :
    IntegrableOn (fun u : ℝ => (1 - Real.cos (s * u)) / u ^ 2) (Set.Ioi 1) := by
  have hbase : IntegrableOn (fun u : ℝ => u ^ (-2 : ℝ)) (Set.Ioi 1) :=
    integrableOn_Ioi_rpow_of_lt (by norm_num) one_pos
  refine (hbase.const_mul 2).mono' (measurable_kernel s).aestronglyMeasurable ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
  have hu0 : (0:ℝ) < u := lt_trans one_pos hu
  rw [Real.norm_eq_abs, abs_of_nonneg (kernel_nonneg s u), rpow_neg_two hu0, mul_one_div]
  have h2 : 1 - Real.cos (s * u) ≤ 2 := by linarith [Real.neg_one_le_cos (s * u)]
  gcongr

theorem integrableOn_kernel_zero (s : ℝ) :
    IntegrableOn (fun u : ℝ => (1 - Real.cos (s * u)) / u ^ 2) (Set.Ioc 0 1) := by
  refine Measure.integrableOn_of_bounded (M := s ^ 2 / 2) (by simp)
    (measurable_kernel s).aestronglyMeasurable ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
  rw [Real.norm_eq_abs, abs_of_nonneg (kernel_nonneg s u)]
  rw [div_le_iff₀ (pow_pos hu.1 2)]
  nlinarith [oneSubCos_le (s * u), sq_nonneg u, hu.1, hu.2, sq_nonneg (s * u)]

theorem integrableOn_kernel (s : ℝ) :
    IntegrableOn (fun u : ℝ => (1 - Real.cos (s * u)) / u ^ 2) (Set.Ioi 0) := by
  have hsplit : Set.Ioi (0:ℝ) = Set.Ioc 0 1 ∪ Set.Ioi 1 := by
    ext u; simp only [Set.mem_Ioi, Set.mem_union, Set.mem_Ioc]
    constructor
    · intro h; rcases le_or_gt u 1 with h1 | h1
      · exact Or.inl ⟨h, h1⟩
      · exact Or.inr h1
    · rintro (⟨h, -⟩ | h)
      · exact h
      · linarith
  rw [hsplit]
  exact (integrableOn_kernel_zero s).union (integrableOn_kernel_one s)


theorem integrableOn_kernel' :
    IntegrableOn (fun u : ℝ => (1 - Real.cos u) / u ^ 2) (Set.Ioi 0) := by
  simpa using integrableOn_kernel 1

/-- `(1 − cos u)/u² ≥ 1/2 − u²/24` on `(0, 3]`, from `sin x > x − x³/6`. -/
theorem kernel_ge {u : ℝ} (hu0 : 0 < u) (hu3 : u ≤ 3) :
    1 / 2 - u ^ 2 / 24 ≤ (1 - Real.cos u) / u ^ 2 := by
  have hs : u / 2 - (u / 2) ^ 3 / 6 < Real.sin (u / 2) := Real.sin_gt_sub_cube (by linarith)
  have hu2 : u ^ 2 ≤ 9 := by nlinarith
  have hpos : 0 ≤ u / 2 - (u / 2) ^ 3 / 6 := by
    nlinarith [mul_nonneg hu0.le (show (0:ℝ) ≤ 24 - u ^ 2 by linarith)]
  have hsq : (u / 2 - (u / 2) ^ 3 / 6) ^ 2 ≤ Real.sin (u / 2) ^ 2 := by nlinarith
  have hid : 1 - Real.cos u = 2 * Real.sin (u / 2) ^ 2 := by
    have h := Real.sin_sq_eq_half_sub (u / 2)
    rw [show 2 * (u / 2) = u by ring] at h
    linarith
  have hexp : 2 * (u / 2 - (u / 2) ^ 3 / 6) ^ 2 = u ^ 2 / 2 - u ^ 4 / 24 + u ^ 6 / 1152 := by
    ring
  rw [hid, le_div_iff₀ (pow_pos hu0 2)]
  nlinarith [pow_nonneg hu0.le 6]

theorem integral_affine_Ioc : ∫ u in Set.Ioc (0:ℝ) 3, (1 / 2 - u ^ 2 / 24) = 9 / 8 := by
  rw [← intervalIntegral.integral_of_le (by norm_num : (0:ℝ) ≤ 3)]
  rw [intervalIntegral.integral_sub intervalIntegrable_const
    ((intervalIntegral.intervalIntegrable_pow 2).div_const 24)]
  simp only [intervalIntegral.integral_div, integral_pow, intervalIntegral.integral_const,
    smul_eq_mul]
  norm_num

/-- **The kernel constant is at least `9/8`**, by `sin x > x − x³/6` on `(0, 3]`.  Its exact value
`π/2 ≈ 1.5708` is proved separately as `dirConst_eq`; this weaker bound is kept because the
unprimed §58c and §59 statements are phrased through it, and it is elementary where `dirConst_eq`
routes through §43's Fourier inversion. -/
theorem dirConst_ge : (9:ℝ) / 8 ≤ dirConst := by
  have hIoc : IntegrableOn (fun u : ℝ => (1 - Real.cos u) / u ^ 2) (Set.Ioc (0:ℝ) 3) :=
    integrableOn_kernel'.mono_set (fun u hu => hu.1)
  have haff : IntegrableOn (fun u : ℝ => 1 / 2 - u ^ 2 / 24) (Set.Ioc (0:ℝ) 3) := by
    apply Continuous.integrableOn_Ioc
    fun_prop
  have h1 : (9:ℝ) / 8 ≤ ∫ u in Set.Ioc (0:ℝ) 3, (1 - Real.cos u) / u ^ 2 := by
    rw [← integral_affine_Ioc]
    exact setIntegral_mono_on haff hIoc measurableSet_Ioc fun u hu => kernel_ge hu.1 hu.2
  refine h1.trans ?_
  refine setIntegral_mono_set integrableOn_kernel' ?_ ?_
  · filter_upwards with u using div_nonneg (oneSubCos_nonneg u) (sq_nonneg u)
  · exact Filter.Eventually.of_forall fun u hu => hu.1

theorem dirConst_pos : 0 < dirConst := lt_of_lt_of_le (by norm_num) dirConst_ge

/-- **The whole-line kernel integral** `∫_ℝ (1 − cos u)/u² du = π`.  This is §43's Fejér density
read at half-width `T = 1/(2π)`, where `(1 − cos(2πTy))/(2π²Ty²)` collapses to `(1 − cos y)/(πy²)`,
combined with `integral_fejerDensity`, which says that density integrates to `1`.  That identity is
Fourier inversion of the triangle at the origin, so all the analytic content is §43's
`fourier_triC`; nothing new is assumed here and no Laplace-transform or contour argument is used.
Mathlib carries `Real.sinc` but no integral of it, so the value is not imported. -/
theorem integral_kernel_univ : ∫ u : ℝ, (1 - Real.cos u) / u ^ 2 = π := by
  have hπ : (0:ℝ) < π := Real.pi_pos
  have hT : (0:ℝ) < 1 / (2 * π) := by positivity
  have h1 := integral_fejerDensity hT
  have h2 : ∫ y : ℝ, fejerDensity (1 / (2 * π)) y
      = (∫ y : ℝ, (1 - Real.cos y) / y ^ 2) / π := by
    rw [← integral_div]
    refine integral_congr_ae ?_
    filter_upwards [Measure.ae_ne volume (0:ℝ)] with y hy
    rw [fejerDensity_eq hT hy]
    rw [show 2 * π * (1 / (2 * π)) * y = y by field_simp]
    rw [show 2 * π ^ 2 * (1 / (2 * π)) * y ^ 2 = y ^ 2 * π by field_simp]
    rw [div_div]
  rw [h2] at h1
  field_simp at h1
  linarith [h1]

/-- **The kernel constant is exactly `π/2`**: the Dirichlet integral, evaluated.  The integrand is
even, so `integral_comp_abs` halves `integral_kernel_univ`.

This supersedes `dirConst_ge`, which is kept because the unprimed statements below are phrased
through it; those therefore carry constants loose by the factor `(π/2)/(9/8) ≈ 1.396`, and the
primed statements at the end of §58 and in §59 carry the sharpened ones.

**What this does not do.**  It evaluates one real number.  No rate in this development becomes
sharp because of it: the factor `2` at `|G(x) − G(a)| ≤ |G(x)| + |G(a)|`, the bound `e^{−z} ≤ 2/z²`
inside §44, and §59's `3/2` for `√(1 + 1/(4(d−1)))` are untouched and remain unquantified losses,
so no constant downstream is claimed near-sharp even after this substitution. -/
theorem dirConst_eq : dirConst = π / 2 := by
  have h := integral_kernel_univ
  have heven : ∫ u : ℝ, (1 - Real.cos u) / u ^ 2
      = 2 * ∫ u in Set.Ioi (0:ℝ), (1 - Real.cos u) / u ^ 2 := by
    rw [← integral_comp_abs (f := fun x : ℝ => (1 - Real.cos x) / x ^ 2)]
    refine integral_congr_ae (Filter.Eventually.of_forall fun u => ?_)
    simp only [Real.cos_abs, sq_abs]
  rw [heven] at h
  rw [dirConst]
  linarith

/-- **The scaling law**: `∫₀^∞ (1 − cos (s u))/u² du = dirConst · |s|`. -/
theorem integral_kernel (s : ℝ) :
    ∫ u in Set.Ioi (0:ℝ), (1 - Real.cos (s * u)) / u ^ 2 = dirConst * |s| := by
  have key : ∀ r : ℝ, 0 < r →
      ∫ u in Set.Ioi (0:ℝ), (1 - Real.cos (r * u)) / u ^ 2 = dirConst * r := by
    intro r hr
    have hg : ∀ u : ℝ, (1 - Real.cos (r * u)) / u ^ 2
        = r ^ 2 * ((1 - Real.cos (r * u)) / (r * u) ^ 2) := by
      intro u
      rcases eq_or_ne u 0 with rfl | hu
      · simp
      · field_simp
    simp_rw [hg]
    rw [integral_comp_mul_left_Ioi (fun w : ℝ => r ^ 2 * ((1 - Real.cos w) / w ^ 2)) 0 hr,
      mul_zero, integral_const_mul, smul_eq_mul, ← dirConst]
    field_simp
  rcases lt_trichotomy s 0 with hs | hs | hs
  · have hcos : ∀ u : ℝ, Real.cos (s * u) = Real.cos (-s * u) := by
      intro u; rw [show -s * u = -(s * u) by ring, Real.cos_neg]
    simp_rw [hcos]
    rw [key (-s) (by linarith), abs_of_neg hs]
  · subst hs; simp
  · rw [key s hs, abs_of_pos hs]


/-! ### The kernel representation of the mean absolute deviation -/

/-- **Tonelli for the kernel.**  For a probability law `ρ` with finite mean absolute deviation at
`y`, the `u`-integrand `u ↦ (∫ (1 − cos((y − t)u)) dρ(t))/u²` is integrable on `(0, ∞)` and its
integral is `dirConst · ∫ |y − t| dρ(t)`.  Both statements come from the same product-integrability
of the nonnegative kernel; no cancellation is involved. -/
theorem integrableOn_and_integral_kernel_measure (ρ : Measure ℝ) [IsProbabilityMeasure ρ] (y : ℝ)
    (hρ : Integrable (fun t : ℝ => |y - t|) ρ) :
    IntegrableOn (fun u : ℝ => (∫ t, (1 - Real.cos ((y - t) * u)) ∂ρ) / u ^ 2) (Set.Ioi 0)
      ∧ (∫ u in Set.Ioi (0:ℝ), (∫ t, (1 - Real.cos ((y - t) * u)) ∂ρ) / u ^ 2)
          = dirConst * ∫ t, |y - t| ∂ρ := by
  set F : ℝ × ℝ → ℝ := fun p => (1 - Real.cos ((y - p.1) * p.2)) / p.2 ^ 2 with hFdef
  have hFm : Measurable F := by fun_prop
  have hFnn : ∀ p : ℝ × ℝ, 0 ≤ F p := fun p => kernel_nonneg (y - p.1) p.2
  have hrow : ∀ t : ℝ, (∫ u in Set.Ioi (0:ℝ), ‖F (t, u)‖) = dirConst * |y - t| := by
    intro t
    have hcongr : (fun u : ℝ => ‖F (t, u)‖)
        = fun u : ℝ => (1 - Real.cos ((y - t) * u)) / u ^ 2 := by
      funext u
      rw [Real.norm_eq_abs, abs_of_nonneg (hFnn (t, u))]
    rw [hcongr]
    exact integral_kernel (y - t)
  have hInt : Integrable F (ρ.prod (volume.restrict (Set.Ioi 0))) := by
    rw [integrable_prod_iff hFm.aestronglyMeasurable]
    refine ⟨Filter.Eventually.of_forall fun t => integrableOn_kernel (y - t), ?_⟩
    exact (hρ.const_mul dirConst).congr (Filter.Eventually.of_forall fun t => (hrow t).symm)
  have hrow2 : ∀ t : ℝ, (∫ u in Set.Ioi (0:ℝ), F (t, u)) = dirConst * |y - t| :=
    fun t => integral_kernel (y - t)
  have hcol : ∀ u : ℝ, (∫ t, F (t, u) ∂ρ)
      = (∫ t, (1 - Real.cos ((y - t) * u)) ∂ρ) / u ^ 2 := by
    intro u
    simp only [hFdef]
    rw [integral_div]
  constructor
  · have h := ((integrable_prod_iff' hFm.aestronglyMeasurable).mp hInt).2
    refine h.congr (Filter.Eventually.of_forall fun u => ?_)
    show (∫ t, ‖F (t, u)‖ ∂ρ) = (∫ t, (1 - Real.cos ((y - t) * u)) ∂ρ) / u ^ 2
    rw [← hcol u]
    refine integral_congr_ae (Filter.Eventually.of_forall fun t => ?_)
    show ‖F (t, u)‖ = F (t, u)
    rw [Real.norm_eq_abs, abs_of_nonneg (hFnn (t, u))]
  · have hswap := integral_integral_swap (f := fun t u => F (t, u)) hInt
    have hgoal : (∫ u in Set.Ioi (0:ℝ), (∫ t, (1 - Real.cos ((y - t) * u)) ∂ρ) / u ^ 2)
        = ∫ u in Set.Ioi (0:ℝ), (∫ t, F (t, u) ∂ρ) :=
      integral_congr_ae (Filter.Eventually.of_forall fun u => (hcol u).symm)
    rw [hgoal, ← hswap]
    simp_rw [hrow2]
    exact integral_const_mul _ _


/-! ### From the kernel to the characteristic functions -/

/-- `cos (u ·)` is integrable against any probability law, being bounded by `1`. -/
theorem integrable_cos (ρ : Measure ℝ) [IsProbabilityMeasure ρ] (u : ℝ) :
    Integrable (fun t : ℝ => Real.cos (u * t)) ρ :=
  (integrable_const (1:ℝ)).mono' (by fun_prop)
    (Filter.Eventually.of_forall fun t => by
      rw [Real.norm_eq_abs]; exact Real.abs_cos_le_one _)

theorem integrable_sin (ρ : Measure ℝ) [IsProbabilityMeasure ρ] (u : ℝ) :
    Integrable (fun t : ℝ => Real.sin (u * t)) ρ :=
  (integrable_const (1:ℝ)).mono' (by fun_prop)
    (Filter.Eventually.of_forall fun t => by
      rw [Real.norm_eq_abs]; exact Real.abs_sin_le_one _)

/-- **The characteristic function in real and imaginary parts.**  Nothing but `e^{iθ} = cos θ +
i sin θ` integrated term by term; both terms are bounded, hence integrable. -/
theorem charFun_eq_cos_add_sin (ρ : Measure ℝ) [IsProbabilityMeasure ρ] (u : ℝ) :
    charFun ρ u = ((∫ t, Real.cos (u * t) ∂ρ : ℝ) : ℂ)
      + ((∫ t, Real.sin (u * t) ∂ρ : ℝ) : ℂ) * Complex.I := by
  rw [charFun_apply_real]
  have hexp : ∀ t : ℝ, Complex.exp ((u : ℂ) * (t : ℂ) * Complex.I)
      = ((Real.cos (u * t) : ℝ) : ℂ) + ((Real.sin (u * t) : ℝ) : ℂ) * Complex.I := by
    intro t
    rw [← Complex.ofReal_mul, Complex.exp_mul_I, Complex.ofReal_cos, Complex.ofReal_sin]
  simp_rw [hexp]
  have hc : Integrable (fun t : ℝ => ((Real.cos (u * t) : ℝ) : ℂ)) ρ :=
    (integrable_cos ρ u).ofReal
  have hs : Integrable (fun t : ℝ => ((Real.sin (u * t) : ℝ) : ℂ) * Complex.I) ρ :=
    ((integrable_sin ρ u).ofReal).mul_const _
  rw [integral_add hc hs, integral_mul_const, integral_complex_ofReal, integral_complex_ofReal]

/-- **The shifted cosine average against the characteristic-function gap.**  For any two
probability laws and any `y`, `u`, the gap between the two averages of `cos((y − t)u)` is at most
`‖φ_μ(u) − φ_ν(u)‖`.  The `y` enters only through `cos(yu)` and `sin(yu)`, whose squares sum to
`1`; this is Cauchy-Schwarz and nothing more. -/
theorem abs_integral_cos_shift_sub_le (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (y u : ℝ) :
    |(∫ t, Real.cos ((y - t) * u) ∂μ) - ∫ t, Real.cos ((y - t) * u) ∂ν|
      ≤ ‖charFun μ u - charFun ν u‖ := by
  have hsplit : ∀ ρ : Measure ℝ, ∀ _ : IsProbabilityMeasure ρ,
      (∫ t, Real.cos ((y - t) * u) ∂ρ)
        = Real.cos (y * u) * (∫ t, Real.cos (u * t) ∂ρ)
          + Real.sin (y * u) * ∫ t, Real.sin (u * t) ∂ρ := by
    intro ρ hρ
    have hpt : ∀ t : ℝ, Real.cos ((y - t) * u)
        = Real.cos (y * u) * Real.cos (u * t) + Real.sin (y * u) * Real.sin (u * t) := by
      intro t
      rw [show (y - t) * u = y * u - u * t by ring, Real.cos_sub]
    simp_rw [hpt]
    rw [integral_add ((integrable_cos ρ u).const_mul _) ((integrable_sin ρ u).const_mul _),
      integral_const_mul, integral_const_mul]
  set A : ℝ := (∫ t, Real.cos (u * t) ∂μ) - ∫ t, Real.cos (u * t) ∂ν with hA
  set B : ℝ := (∫ t, Real.sin (u * t) ∂μ) - ∫ t, Real.sin (u * t) ∂ν with hB
  have hgap : (∫ t, Real.cos ((y - t) * u) ∂μ) - ∫ t, Real.cos ((y - t) * u) ∂ν
      = Real.cos (y * u) * A + Real.sin (y * u) * B := by
    rw [hsplit μ ‹_›, hsplit ν ‹_›, hA, hB]; ring
  have hnorm : ‖charFun μ u - charFun ν u‖ = Real.sqrt (A ^ 2 + B ^ 2) := by
    rw [charFun_eq_cos_add_sin μ u, charFun_eq_cos_add_sin ν u]
    rw [show (((∫ t, Real.cos (u * t) ∂μ : ℝ) : ℂ)
          + ((∫ t, Real.sin (u * t) ∂μ : ℝ) : ℂ) * Complex.I)
        - (((∫ t, Real.cos (u * t) ∂ν : ℝ) : ℂ)
          + ((∫ t, Real.sin (u * t) ∂ν : ℝ) : ℂ) * Complex.I)
        = ((A : ℝ) : ℂ) + ((B : ℝ) : ℂ) * Complex.I by rw [hA, hB]; push_cast; ring]
    exact Complex.norm_add_mul_I A B
  rw [hgap, hnorm]
  have hpyth : Real.cos (y * u) ^ 2 + Real.sin (y * u) ^ 2 = 1 := by
    rw [add_comm]; exact Real.sin_sq_add_cos_sq _
  have hsq : (Real.cos (y * u) * A + Real.sin (y * u) * B) ^ 2 ≤ A ^ 2 + B ^ 2 := by
    nlinarith [sq_nonneg (Real.sin (y * u) * A - Real.cos (y * u) * B)]
  calc |Real.cos (y * u) * A + Real.sin (y * u) * B|
      = Real.sqrt ((Real.cos (y * u) * A + Real.sin (y * u) * B) ^ 2) :=
        (Real.sqrt_sq_eq_abs _).symm
    _ ≤ Real.sqrt (A ^ 2 + B ^ 2) := Real.sqrt_le_sqrt hsq


/-! ### The inversion bound on `cdfGapInt` -/

theorem integrable_cos_shift (ρ : Measure ℝ) [IsProbabilityMeasure ρ] (y u : ℝ) :
    Integrable (fun t : ℝ => Real.cos ((y - t) * u)) ρ :=
  (integrable_const (1:ℝ)).mono' (by fun_prop)
    (Filter.Eventually.of_forall fun t => by
      rw [Real.norm_eq_abs]; exact Real.abs_cos_le_one _)

/-- **The mean absolute deviation gap, bounded by the characteristic-function integral.**  For two
probability laws with finite first absolute moments, and at every `y`,

  `|∫ |y − t| dμ − ∫ |y − t| dν| ≤ (∫₀^∞ ‖φ_μ(u) − φ_ν(u)‖/u² du) / dirConst`.

The `1` in `1 − cos` cancels between the two laws, which is why no hypothesis beyond the two
first moments is needed; no equality of means is used here. -/
theorem abs_integral_abs_sub_le (μ ν : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (y : ℝ) (hμ : Integrable (fun t : ℝ => |y - t|) μ) (hν : Integrable (fun t : ℝ => |y - t|) ν)
    (hcf : IntegrableOn (fun u : ℝ => ‖charFun μ u - charFun ν u‖ / u ^ 2) (Set.Ioi 0)) :
    |(∫ t, |y - t| ∂μ) - ∫ t, |y - t| ∂ν|
      ≤ (∫ u in Set.Ioi (0:ℝ), ‖charFun μ u - charFun ν u‖ / u ^ 2) / dirConst := by
  obtain ⟨hIμ, hJμ⟩ := integrableOn_and_integral_kernel_measure μ y hμ
  obtain ⟨hIν, hJν⟩ := integrableOn_and_integral_kernel_measure ν y hν
  have hdiff : dirConst * ((∫ t, |y - t| ∂μ) - ∫ t, |y - t| ∂ν)
      = ∫ u in Set.Ioi (0:ℝ), ((∫ t, (1 - Real.cos ((y - t) * u)) ∂μ) / u ^ 2
          - (∫ t, (1 - Real.cos ((y - t) * u)) ∂ν) / u ^ 2) := by
    rw [integral_sub hIμ hIν, hJμ, hJν]; ring
  have hone : ∀ ρ : Measure ℝ, ∀ _ : IsProbabilityMeasure ρ, ∀ u : ℝ,
      (∫ t, (1 - Real.cos ((y - t) * u)) ∂ρ) = 1 - ∫ t, Real.cos ((y - t) * u) ∂ρ := by
    intro ρ hρ u
    haveI := hρ
    rw [integral_sub (integrable_const 1) (integrable_cos_shift ρ y u)]
    simp
  have hpt : ∀ u ∈ Set.Ioi (0:ℝ),
      |(∫ t, (1 - Real.cos ((y - t) * u)) ∂μ) / u ^ 2
          - (∫ t, (1 - Real.cos ((y - t) * u)) ∂ν) / u ^ 2|
        ≤ ‖charFun μ u - charFun ν u‖ / u ^ 2 := by
    intro u hu
    have hu2 : (0:ℝ) < u ^ 2 := pow_pos hu 2
    rw [div_sub_div_same, abs_div, abs_of_pos hu2]
    have hnum : |(∫ t, (1 - Real.cos ((y - t) * u)) ∂μ)
        - ∫ t, (1 - Real.cos ((y - t) * u)) ∂ν| ≤ ‖charFun μ u - charFun ν u‖ := by
      rw [hone μ ‹_› u, hone ν ‹_› u,
        show (1 - ∫ t, Real.cos ((y - t) * u) ∂μ) - (1 - ∫ t, Real.cos ((y - t) * u) ∂ν)
          = -((∫ t, Real.cos ((y - t) * u) ∂μ) - ∫ t, Real.cos ((y - t) * u) ∂ν) by ring,
        abs_neg]
      exact abs_integral_cos_shift_sub_le μ ν y u
    gcongr
  have hsubint : IntegrableOn (fun u : ℝ => (∫ t, (1 - Real.cos ((y - t) * u)) ∂μ) / u ^ 2
      - (∫ t, (1 - Real.cos ((y - t) * u)) ∂ν) / u ^ 2) (Set.Ioi 0) := hIμ.sub hIν
  have habs : dirConst * |(∫ t, |y - t| ∂μ) - ∫ t, |y - t| ∂ν|
      ≤ ∫ u in Set.Ioi (0:ℝ), ‖charFun μ u - charFun ν u‖ / u ^ 2 := by
    rw [← abs_of_pos dirConst_pos, ← abs_mul, hdiff]
    calc |∫ u in Set.Ioi (0:ℝ), ((∫ t, (1 - Real.cos ((y - t) * u)) ∂μ) / u ^ 2
            - (∫ t, (1 - Real.cos ((y - t) * u)) ∂ν) / u ^ 2)|
        ≤ ∫ u in Set.Ioi (0:ℝ), |(∫ t, (1 - Real.cos ((y - t) * u)) ∂μ) / u ^ 2
            - (∫ t, (1 - Real.cos ((y - t) * u)) ∂ν) / u ^ 2| := abs_integral_le_integral_abs
      _ ≤ _ := setIntegral_mono_on hsubint.abs hcf measurableSet_Ioi hpt
  rw [le_div_iff₀ dirConst_pos]
  linarith [habs]


theorem max_eq_add_abs_div_two (s : ℝ) : max s 0 = (s + |s|) / 2 := by
  rcases le_total 0 s with h | h
  · rw [max_eq_left h, abs_of_nonneg h]; ring
  · rw [max_eq_right h, abs_of_nonpos h]; ring

/-- **Fourier inversion for the once-integrated CDF gap.**  For two probability laws with finite
first absolute moments whose characteristic-function gap is `u⁻²`-integrable on `(0, ∞)`,

  `|cdfGapInt μ ν a x| ≤ (∫₀^∞ ‖φ_μ(u) − φ_ν(u)‖/u² du) / dirConst`

for **every** pair `a`, `x`, with no support hypothesis and no relation between `a` and `x`.  The
route is the identity `(x − t)⁺ − (a − t)⁺ = (x − a)/2 + (|x − t| − |a − t|)/2`, in which the
`(x − a)/2` cancels between the two laws; the two mean absolute deviations are then read through
the kernel `∫₀^∞ (1 − cos(su))/u² du = dirConst·|s|`.

This is **not** a bound on `|cdf μ − cdf ν|`: nothing here controls the CDF gap itself, only its
integral.  A factor `2` is spent at `|G(x) − G(a)| ≤ |G(x)| + |G(a)|`.
`abs_cdfGapInt_le_of_charFun_sin` below keeps the cancellation between `x` and `a` that this step
discards and is never worse, but it does not improve the constant uniformly in `a` and `x`: the
factor `2` here is sharp for arbitrary pairs of laws, with the witness recorded in the §58a′
header.  This statement is what §58c and §58c′ use. -/
theorem abs_cdfGapInt_le_of_charFun (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (hμ : ∀ y : ℝ, Integrable (fun t : ℝ => |y - t|) μ)
    (hν : ∀ y : ℝ, Integrable (fun t : ℝ => |y - t|) ν)
    (hcf : IntegrableOn (fun u : ℝ => ‖charFun μ u - charFun ν u‖ / u ^ 2) (Set.Ioi 0))
    (a x : ℝ) :
    |Threshold.cdfGapInt μ ν a x|
      ≤ (∫ u in Set.Ioi (0:ℝ), ‖charFun μ u - charFun ν u‖ / u ^ 2) / dirConst := by
  set I : ℝ := ∫ u in Set.Ioi (0:ℝ), ‖charFun μ u - charFun ν u‖ / u ^ 2 with hIdef
  have hInn : 0 ≤ I / dirConst := by
    refine div_nonneg ?_ dirConst_pos.le
    exact setIntegral_nonneg measurableSet_Ioi fun u _ => by positivity
  rcases le_or_gt a x with hax | hax
  · have hclamp : ∀ ρ : Measure ℝ, ∀ _ : IsProbabilityMeasure ρ,
        Integrable (fun t : ℝ => |x - t|) ρ → Integrable (fun t : ℝ => |a - t|) ρ →
        (∫ t, (max (x - t) 0 - max (a - t) 0) ∂ρ)
          = ((x - a) + ((∫ t, |x - t| ∂ρ) - ∫ t, |a - t| ∂ρ)) / 2 := by
      intro ρ hρ hix hia
      haveI := hρ
      have hpt : ∀ t : ℝ, max (x - t) 0 - max (a - t) 0
          = ((x - a) + (|x - t| - |a - t|)) / 2 := by
        intro t
        rw [max_eq_add_abs_div_two (x - t), max_eq_add_abs_div_two (a - t)]; ring
      have hd : Integrable (fun t : ℝ => |x - t| - |a - t|) ρ := hix.sub hia
      simp_rw [hpt]
      rw [integral_div, integral_add (integrable_const (x - a)) hd, integral_sub hix hia]
      simp
    rw [Threshold.cdfGapInt_eq_integral_clamp_sub μ ν hax, hclamp μ ‹_› (hμ x) (hμ a),
      hclamp ν ‹_› (hν x) (hν a),
      show ((x - a) + ((∫ t, |x - t| ∂μ) - ∫ t, |a - t| ∂μ)) / 2
          - ((x - a) + ((∫ t, |x - t| ∂ν) - ∫ t, |a - t| ∂ν)) / 2
        = (((∫ t, |x - t| ∂μ) - ∫ t, |x - t| ∂ν)
            - ((∫ t, |a - t| ∂μ) - ∫ t, |a - t| ∂ν)) / 2 by ring,
      abs_div, abs_of_nonneg (by norm_num : (0:ℝ) ≤ 2)]
    have h1 := abs_integral_abs_sub_le μ ν x (hμ x) (hν x) hcf
    have h2 := abs_integral_abs_sub_le μ ν a (hμ a) (hν a) hcf
    have hsum : |((∫ t, |x - t| ∂μ) - ∫ t, |x - t| ∂ν)
        - ((∫ t, |a - t| ∂μ) - ∫ t, |a - t| ∂ν)| ≤ I / dirConst + I / dirConst :=
      (abs_sub _ _).trans (add_le_add h1 h2)
    linarith
  · have hempty : Set.Ioc a x = (∅ : Set ℝ) := Set.Ioc_eq_empty (by linarith)
    rw [Threshold.cdfGapInt, hempty, Measure.restrict_empty, integral_zero_measure, abs_zero]
    exact hInn


/-! ### §58a′  The same inversion with the cancellation between `x` and `a` kept

`abs_cdfGapInt_le_of_charFun` above bounds `|G(x) − G(a)|` by `|G(x)| + |G(a)|` and then bounds
each half by the same Fourier integral, which costs a factor `2`.  The Fourier representation in
fact delivers `G(x) − G(a)` directly: the kernel for the difference is the difference of the two
kernels, and by `cos α − cos β = 2 sin((α+β)/2) sin((β−α)/2)` that difference carries an extra
factor `2 sin((x − a)u/2)`.  Keeping it gives `abs_cdfGapInt_le_of_charFun_sin`, which is never
worse than `abs_cdfGapInt_le_of_charFun` (`integral_sinWeight_le`) and is strictly better at every
`u` where `|sin((x − a)u/2)| < 1`.

**The factor `2` is nevertheless sharp for the general statement, and this block says why.**  The
refined bound is `(∫₀^∞ |sin((x − a)u/2)|·‖φ_μ − φ_ν‖/u² du)/dirConst`, and taking the supremum
over `a` and `x` recovers the unrefined one: for any weight `w` concentrated near a single
frequency `u₀`, the choice `x − a = π/u₀` makes `|sin((x − a)u/2)| ≈ 1` across the whole band.
Such weights are realised: if `φ_μ − φ_ν` is a narrow bump near `±u₀`, then `G(y)` is
approximately a cosine in `y` of amplitude equal to the full Fourier integral, so `G` attains
`+B` and `−B` and `sup_{a,x}|G(x) − G(a)| = 2B` is attained in the limit.  The requirement that
`‖φ_μ − φ_ν‖/u²` be integrable at `0` forces the two means to agree and so rules out the cruder
witnesses, but it does not rule these out.  **So no constant below `1` is available in
`abs_cdfGapInt_le_of_charFun` uniformly in the pair of laws, and the factor `2` there is not slack
but sharpness.**

**What is still recoverable, and what it would take.**  At the particular weight §58b produces the
supremum over `x − a` is not attained: numerical evaluation at `d = 512` and a spread direction
gives `sup_h ∫|sin(hu/2)|w / ∫w = 0.72`, a factor `1.39` and not `2`.  Extracting it needs
`sup_{c>0} c·∫_c^∞ |sin w|/w² dw = 0.7787… < 1` on the high range, an oscillation estimate which
is **not** proved here; both numbers are numerical and neither is formalised.  The theorems below
are exactly the inequality, with no such estimate in them. -/

/-- The shifted cosine average in terms of the two real characteristic-function integrals.  This
is the `hsplit` of `abs_integral_cos_shift_sub_le` named, so that the two-point version can reuse
it; it is `cos((y − t)u) = cos(yu)cos(ut) + sin(yu)sin(ut)` integrated. -/
theorem integral_cos_shift_eq (ρ : Measure ℝ) [IsProbabilityMeasure ρ] (y u : ℝ) :
    (∫ t, Real.cos ((y - t) * u) ∂ρ)
      = Real.cos (y * u) * (∫ t, Real.cos (u * t) ∂ρ)
        + Real.sin (y * u) * ∫ t, Real.sin (u * t) ∂ρ := by
  have hpt : ∀ t : ℝ, Real.cos ((y - t) * u)
      = Real.cos (y * u) * Real.cos (u * t) + Real.sin (y * u) * Real.sin (u * t) := by
    intro t
    rw [show (y - t) * u = y * u - u * t by ring, Real.cos_sub]
  simp_rw [hpt]
  rw [integral_add ((integrable_cos ρ u).const_mul _) ((integrable_sin ρ u).const_mul _),
    integral_const_mul, integral_const_mul]

/-- **Cauchy-Schwarz in `ℝ²` against the characteristic-function gap.**  For any real `p`, `q`,
the combination `p·(cosine-average gap) + q·(sine-average gap)` is at most
`√(p² + q²)·‖φ_μ(u) − φ_ν(u)‖`.  `abs_integral_cos_shift_sub_le` is the case
`(p, q) = (cos(yu), sin(yu))`, where `√(p² + q²) = 1`; the point of stating it for general
`(p, q)` is that the two-point coefficient vector has norm `2|sin((x − a)u/2)|`, which is smaller
than `2` off a discrete set of `u`. -/
theorem abs_comb_charFun_gap_le (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (p q u : ℝ) :
    |p * ((∫ t, Real.cos (u * t) ∂μ) - ∫ t, Real.cos (u * t) ∂ν)
        + q * ((∫ t, Real.sin (u * t) ∂μ) - ∫ t, Real.sin (u * t) ∂ν)|
      ≤ Real.sqrt (p ^ 2 + q ^ 2) * ‖charFun μ u - charFun ν u‖ := by
  set A : ℝ := (∫ t, Real.cos (u * t) ∂μ) - ∫ t, Real.cos (u * t) ∂ν with hA
  set B : ℝ := (∫ t, Real.sin (u * t) ∂μ) - ∫ t, Real.sin (u * t) ∂ν with hB
  have hnorm : ‖charFun μ u - charFun ν u‖ = Real.sqrt (A ^ 2 + B ^ 2) := by
    rw [charFun_eq_cos_add_sin μ u, charFun_eq_cos_add_sin ν u]
    rw [show (((∫ t, Real.cos (u * t) ∂μ : ℝ) : ℂ)
          + ((∫ t, Real.sin (u * t) ∂μ : ℝ) : ℂ) * Complex.I)
        - (((∫ t, Real.cos (u * t) ∂ν : ℝ) : ℂ)
          + ((∫ t, Real.sin (u * t) ∂ν : ℝ) : ℂ) * Complex.I)
        = ((A : ℝ) : ℂ) + ((B : ℝ) : ℂ) * Complex.I by rw [hA, hB]; push_cast; ring]
    exact Complex.norm_add_mul_I A B
  rw [hnorm, ← Real.sqrt_mul (by positivity)]
  have hsq : (p * A + q * B) ^ 2 ≤ (p ^ 2 + q ^ 2) * (A ^ 2 + B ^ 2) := by
    nlinarith [sq_nonneg (q * A - p * B)]
  calc |p * A + q * B| = Real.sqrt ((p * A + q * B) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
    _ ≤ Real.sqrt ((p ^ 2 + q ^ 2) * (A ^ 2 + B ^ 2)) := Real.sqrt_le_sqrt hsq

/-- **The chord identity.**  The squared distance between the two unit vectors `(cos(xu), sin(xu))`
and `(cos(au), sin(au))` is `4 sin²((x − a)u/2)`.  This is the whole of the extra cancellation the
two-point kernel carries: its coefficient vector has norm `2|sin((x − a)u/2)|` rather than `2`. -/
theorem sq_add_sq_cos_sin_sub (a x u : ℝ) :
    (Real.cos (x * u) - Real.cos (a * u)) ^ 2 + (Real.sin (x * u) - Real.sin (a * u)) ^ 2
      = (2 * |Real.sin ((x - a) * u / 2)|) ^ 2 := by
  have h1 := Real.sin_sq_add_cos_sq (x * u)
  have h2 := Real.sin_sq_add_cos_sq (a * u)
  have h4 : Real.sin ((x - a) * u / 2) ^ 2 = 1 / 2 - Real.cos (2 * ((x - a) * u / 2)) / 2 :=
    Real.sin_sq_eq_half_sub _
  rw [show 2 * ((x - a) * u / 2) = x * u - a * u by ring, Real.cos_sub] at h4
  have habs : |Real.sin ((x - a) * u / 2)| ^ 2 = Real.sin ((x - a) * u / 2) ^ 2 := sq_abs _
  nlinarith [h1, h2, h4, habs]

/-- **The two-point kernel bound.**  The gap between the two laws of
`cos((x − t)u) − cos((a − t)u)` is at most `2|sin((x − a)u/2)|·‖φ_μ(u) − φ_ν(u)‖`.  Applying
`abs_integral_cos_shift_sub_le` at `x` and at `a` separately and adding gives `2·‖φ_μ − φ_ν‖`
instead, which is where the factor `2` of `abs_cdfGapInt_le_of_charFun` comes from; here the two
coefficient vectors are subtracted before the norm is taken. -/
theorem abs_integral_cos_shift_diff_sub_le (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a x u : ℝ) :
    |((∫ t, Real.cos ((x - t) * u) ∂μ) - ∫ t, Real.cos ((x - t) * u) ∂ν)
        - ((∫ t, Real.cos ((a - t) * u) ∂μ) - ∫ t, Real.cos ((a - t) * u) ∂ν)|
      ≤ 2 * |Real.sin ((x - a) * u / 2)| * ‖charFun μ u - charFun ν u‖ := by
  set p : ℝ := Real.cos (x * u) - Real.cos (a * u) with hp
  set q : ℝ := Real.sin (x * u) - Real.sin (a * u) with hq
  have hrw : ((∫ t, Real.cos ((x - t) * u) ∂μ) - ∫ t, Real.cos ((x - t) * u) ∂ν)
      - ((∫ t, Real.cos ((a - t) * u) ∂μ) - ∫ t, Real.cos ((a - t) * u) ∂ν)
      = p * ((∫ t, Real.cos (u * t) ∂μ) - ∫ t, Real.cos (u * t) ∂ν)
        + q * ((∫ t, Real.sin (u * t) ∂μ) - ∫ t, Real.sin (u * t) ∂ν) := by
    rw [integral_cos_shift_eq μ x u, integral_cos_shift_eq ν x u, integral_cos_shift_eq μ a u,
      integral_cos_shift_eq ν a u, hp, hq]
    ring
  rw [hrw]
  refine (abs_comb_charFun_gap_le μ ν p q u).trans ?_
  have hsq : p ^ 2 + q ^ 2 = (2 * |Real.sin ((x - a) * u / 2)|) ^ 2 :=
    sq_add_sq_cos_sin_sub a x u
  rw [hsq, Real.sqrt_sq (by positivity)]

/-- The sine-weighted integrand is integrable whenever the unweighted one is, since
`|sin| ≤ 1`. -/
theorem integrableOn_sinWeight (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (h : ℝ)
    (hcf : IntegrableOn (fun u : ℝ => ‖charFun μ u - charFun ν u‖ / u ^ 2) (Set.Ioi 0)) :
    IntegrableOn (fun u : ℝ => |Real.sin (h * u / 2)| * ‖charFun μ u - charFun ν u‖ / u ^ 2)
      (Set.Ioi 0) := by
  have hc : Continuous fun u : ℝ => charFun μ u - charFun ν u :=
    continuous_charFun.sub continuous_charFun
  have hmeas : Measurable fun u : ℝ =>
      |Real.sin (h * u / 2)| * ‖charFun μ u - charFun ν u‖ / u ^ 2 := by
    refine Measurable.div (Measurable.mul ?_ hc.norm.measurable) (by fun_prop)
    fun_prop
  refine hcf.mono' hmeas.aestronglyMeasurable ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
  have hu2 : (0:ℝ) < u ^ 2 := pow_pos hu 2
  rw [Real.norm_eq_abs, abs_of_nonneg (by positivity), div_le_div_iff_of_pos_right hu2]
  nlinarith [Real.abs_sin_le_one (h * u / 2), norm_nonneg (charFun μ u - charFun ν u)]

theorem integrableOn_two_sinWeight (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (h : ℝ)
    (hcf : IntegrableOn (fun u : ℝ => ‖charFun μ u - charFun ν u‖ / u ^ 2) (Set.Ioi 0)) :
    IntegrableOn (fun u : ℝ => 2 * |Real.sin (h * u / 2)| * ‖charFun μ u - charFun ν u‖ / u ^ 2)
      (Set.Ioi 0) := by
  have h2 : IntegrableOn (fun u : ℝ => 2 * (|Real.sin (h * u / 2)|
      * ‖charFun μ u - charFun ν u‖ / u ^ 2)) (Set.Ioi 0) :=
    (integrableOn_sinWeight μ ν h hcf).const_mul 2
  refine h2.congr_fun (fun u _ => ?_) measurableSet_Ioi
  ring

/-- **The refined bound is never worse than the unrefined one.**  `|sin| ≤ 1` pointwise, so the
sine-weighted Fourier integral is at most the plain one and `abs_cdfGapInt_le_of_charFun_sin`
implies `abs_cdfGapInt_le_of_charFun`.  It says nothing about how much smaller it is: at
`x − a` large the two are equal in the limit for a narrow-band weight, which is the sharpness
recorded in this block's header. -/
theorem integral_sinWeight_le (μ ν : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h : ℝ) (hcf : IntegrableOn (fun u : ℝ => ‖charFun μ u - charFun ν u‖ / u ^ 2) (Set.Ioi 0)) :
    (∫ u in Set.Ioi (0:ℝ), |Real.sin (h * u / 2)| * ‖charFun μ u - charFun ν u‖ / u ^ 2)
      ≤ ∫ u in Set.Ioi (0:ℝ), ‖charFun μ u - charFun ν u‖ / u ^ 2 := by
  refine setIntegral_mono_on (integrableOn_sinWeight μ ν h hcf) hcf measurableSet_Ioi ?_
  intro u hu
  have hu2 : (0:ℝ) < u ^ 2 := pow_pos hu 2
  rw [div_le_div_iff_of_pos_right hu2]
  nlinarith [Real.abs_sin_le_one (h * u / 2), norm_nonneg (charFun μ u - charFun ν u)]

/-- **The mean-absolute-deviation gap, two points at once.**  For two probability laws with finite
first absolute moments and `u⁻²`-integrable characteristic-function gap,

  `|(∫|x − t| dμ − ∫|x − t| dν) − (∫|a − t| dμ − ∫|a − t| dν)|
     ≤ (∫₀^∞ 2|sin((x − a)u/2)|·‖φ_μ − φ_ν‖/u² du)/dirConst`.

`abs_integral_abs_sub_le` applied at `x` and at `a` and added gives the same statement with
`2·‖φ_μ − φ_ν‖` in place of `2|sin((x − a)u/2)|·‖φ_μ − φ_ν‖`, so this is the version that keeps
the cancellation.  It does **not** improve the supremum over `a` and `x`: see the block header. -/
theorem abs_integral_abs_sub_diff_le (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a x : ℝ)
    (hμx : Integrable (fun t : ℝ => |x - t|) μ) (hνx : Integrable (fun t : ℝ => |x - t|) ν)
    (hμa : Integrable (fun t : ℝ => |a - t|) μ) (hνa : Integrable (fun t : ℝ => |a - t|) ν)
    (hcf : IntegrableOn (fun u : ℝ => ‖charFun μ u - charFun ν u‖ / u ^ 2) (Set.Ioi 0)) :
    |((∫ t, |x - t| ∂μ) - ∫ t, |x - t| ∂ν) - ((∫ t, |a - t| ∂μ) - ∫ t, |a - t| ∂ν)|
      ≤ (∫ u in Set.Ioi (0:ℝ),
          2 * |Real.sin ((x - a) * u / 2)| * ‖charFun μ u - charFun ν u‖ / u ^ 2) / dirConst := by
  obtain ⟨hIμx, hJμx⟩ := integrableOn_and_integral_kernel_measure μ x hμx
  obtain ⟨hIνx, hJνx⟩ := integrableOn_and_integral_kernel_measure ν x hνx
  obtain ⟨hIμa, hJμa⟩ := integrableOn_and_integral_kernel_measure μ a hμa
  obtain ⟨hIνa, hJνa⟩ := integrableOn_and_integral_kernel_measure ν a hνa
  have hone : ∀ ρ : Measure ℝ, ∀ _ : IsProbabilityMeasure ρ, ∀ y u : ℝ,
      (∫ t, (1 - Real.cos ((y - t) * u)) ∂ρ) = 1 - ∫ t, Real.cos ((y - t) * u) ∂ρ := by
    intro ρ hρ y u
    haveI := hρ
    rw [integral_sub (integrable_const 1) (integrable_cos_shift ρ y u)]
    simp
  have hX : (∫ u in Set.Ioi (0:ℝ), ((∫ t, (1 - Real.cos ((x - t) * u)) ∂μ) / u ^ 2
        - (∫ t, (1 - Real.cos ((x - t) * u)) ∂ν) / u ^ 2))
      = dirConst * ((∫ t, |x - t| ∂μ) - ∫ t, |x - t| ∂ν) := by
    rw [integral_sub hIμx hIνx, hJμx, hJνx]; ring
  have hA : (∫ u in Set.Ioi (0:ℝ), ((∫ t, (1 - Real.cos ((a - t) * u)) ∂μ) / u ^ 2
        - (∫ t, (1 - Real.cos ((a - t) * u)) ∂ν) / u ^ 2))
      = dirConst * ((∫ t, |a - t| ∂μ) - ∫ t, |a - t| ∂ν) := by
    rw [integral_sub hIμa hIνa, hJμa, hJνa]; ring
  have hOuter : (∫ u in Set.Ioi (0:ℝ), (((∫ t, (1 - Real.cos ((x - t) * u)) ∂μ) / u ^ 2
          - (∫ t, (1 - Real.cos ((x - t) * u)) ∂ν) / u ^ 2)
        - ((∫ t, (1 - Real.cos ((a - t) * u)) ∂μ) / u ^ 2
          - (∫ t, (1 - Real.cos ((a - t) * u)) ∂ν) / u ^ 2)))
      = (∫ u in Set.Ioi (0:ℝ), ((∫ t, (1 - Real.cos ((x - t) * u)) ∂μ) / u ^ 2
          - (∫ t, (1 - Real.cos ((x - t) * u)) ∂ν) / u ^ 2))
        - (∫ u in Set.Ioi (0:ℝ), ((∫ t, (1 - Real.cos ((a - t) * u)) ∂μ) / u ^ 2
          - (∫ t, (1 - Real.cos ((a - t) * u)) ∂ν) / u ^ 2)) :=
    integral_sub (hIμx.sub hIνx) (hIμa.sub hIνa)
  have hdiff : dirConst * (((∫ t, |x - t| ∂μ) - ∫ t, |x - t| ∂ν)
        - ((∫ t, |a - t| ∂μ) - ∫ t, |a - t| ∂ν))
      = ∫ u in Set.Ioi (0:ℝ), (((∫ t, (1 - Real.cos ((x - t) * u)) ∂μ) / u ^ 2
            - (∫ t, (1 - Real.cos ((x - t) * u)) ∂ν) / u ^ 2)
          - ((∫ t, (1 - Real.cos ((a - t) * u)) ∂μ) / u ^ 2
            - (∫ t, (1 - Real.cos ((a - t) * u)) ∂ν) / u ^ 2)) := by
    rw [hOuter, hX, hA]; ring
  have hpt : ∀ u ∈ Set.Ioi (0:ℝ),
      |((∫ t, (1 - Real.cos ((x - t) * u)) ∂μ) / u ^ 2
            - (∫ t, (1 - Real.cos ((x - t) * u)) ∂ν) / u ^ 2)
          - ((∫ t, (1 - Real.cos ((a - t) * u)) ∂μ) / u ^ 2
            - (∫ t, (1 - Real.cos ((a - t) * u)) ∂ν) / u ^ 2)|
        ≤ 2 * |Real.sin ((x - a) * u / 2)| * ‖charFun μ u - charFun ν u‖ / u ^ 2 := by
    intro u hu
    have hu2 : (0:ℝ) < u ^ 2 := pow_pos hu 2
    have hcomb : (((∫ t, (1 - Real.cos ((x - t) * u)) ∂μ) / u ^ 2
            - (∫ t, (1 - Real.cos ((x - t) * u)) ∂ν) / u ^ 2)
          - ((∫ t, (1 - Real.cos ((a - t) * u)) ∂μ) / u ^ 2
            - (∫ t, (1 - Real.cos ((a - t) * u)) ∂ν) / u ^ 2))
        = (-(((∫ t, Real.cos ((x - t) * u) ∂μ) - ∫ t, Real.cos ((x - t) * u) ∂ν)
            - ((∫ t, Real.cos ((a - t) * u) ∂μ) - ∫ t, Real.cos ((a - t) * u) ∂ν))) / u ^ 2 := by
      rw [hone μ ‹_› x u, hone ν ‹_› x u, hone μ ‹_› a u, hone ν ‹_› a u]
      ring
    rw [hcomb, abs_div, abs_of_pos hu2, abs_neg, div_le_div_iff_of_pos_right hu2]
    exact abs_integral_cos_shift_diff_sub_le μ ν a x u
  have hsubint : IntegrableOn (fun u : ℝ => (((∫ t, (1 - Real.cos ((x - t) * u)) ∂μ) / u ^ 2
        - (∫ t, (1 - Real.cos ((x - t) * u)) ∂ν) / u ^ 2)
      - ((∫ t, (1 - Real.cos ((a - t) * u)) ∂μ) / u ^ 2
        - (∫ t, (1 - Real.cos ((a - t) * u)) ∂ν) / u ^ 2))) (Set.Ioi 0) :=
    (hIμx.sub hIνx).sub (hIμa.sub hIνa)
  have hmaj := integrableOn_two_sinWeight μ ν (x - a) hcf
  have habs : dirConst * |((∫ t, |x - t| ∂μ) - ∫ t, |x - t| ∂ν)
        - ((∫ t, |a - t| ∂μ) - ∫ t, |a - t| ∂ν)|
      ≤ ∫ u in Set.Ioi (0:ℝ),
          2 * |Real.sin ((x - a) * u / 2)| * ‖charFun μ u - charFun ν u‖ / u ^ 2 := by
    rw [← abs_of_pos dirConst_pos, ← abs_mul, hdiff]
    calc |∫ u in Set.Ioi (0:ℝ), (((∫ t, (1 - Real.cos ((x - t) * u)) ∂μ) / u ^ 2
              - (∫ t, (1 - Real.cos ((x - t) * u)) ∂ν) / u ^ 2)
            - ((∫ t, (1 - Real.cos ((a - t) * u)) ∂μ) / u ^ 2
              - (∫ t, (1 - Real.cos ((a - t) * u)) ∂ν) / u ^ 2))|
        ≤ ∫ u in Set.Ioi (0:ℝ), |((∫ t, (1 - Real.cos ((x - t) * u)) ∂μ) / u ^ 2
              - (∫ t, (1 - Real.cos ((x - t) * u)) ∂ν) / u ^ 2)
            - ((∫ t, (1 - Real.cos ((a - t) * u)) ∂μ) / u ^ 2
              - (∫ t, (1 - Real.cos ((a - t) * u)) ∂ν) / u ^ 2)| :=
          abs_integral_le_integral_abs
      _ ≤ _ := setIntegral_mono_on hsubint.abs hmaj measurableSet_Ioi hpt
  rw [le_div_iff₀ dirConst_pos]
  linarith [habs]

/-- **Fourier inversion for the once-integrated CDF gap, with the `x`-against-`a` cancellation
kept.**  For two probability laws with finite first absolute moments whose characteristic-function
gap is `u⁻²`-integrable on `(0, ∞)`, and for **every** pair `a`, `x`,

  `|cdfGapInt μ ν a x| ≤ (∫₀^∞ |sin((x − a)u/2)|·‖φ_μ(u) − φ_ν(u)‖/u² du)/dirConst`.

Since `|sin| ≤ 1` this implies `abs_cdfGapInt_le_of_charFun` (`integral_sinWeight_le`), so it is
never worse, and it is strictly better at every `u` where `|sin((x − a)u/2)| < 1`.  Both
statements are kept; neither weakens the other.

**What it does not do.**  It is still not a bound on `|cdf μ − cdf ν|`.  And it does **not**
improve the constant uniformly in `a` and `x`: taking `x − a = π/u₀` when the weight
`‖φ_μ − φ_ν‖/u²` is concentrated near `u₀` makes the sine factor `≈ 1` across the band, and such
weights are realised by pairs whose characteristic-function gap is a narrow bump near `±u₀`, for
which `G(y) = ∫(y − t)⁺ dμ − ∫(y − t)⁺ dν` is approximately a cosine in `y` and so attains both
`+B` and `−B`.  The factor `2` of `abs_cdfGapInt_le_of_charFun` is therefore **sharp** as a
statement about arbitrary pairs of laws, and the gain here is available only against a specific
weight.  At the weight §58b produces the gain is a factor `1.39` at `d = 512` and a spread
direction, not `2`; that number is numerical, is not formalised, and is not used below.  The leg-1
bounds of §58c and §58c′ are proved from `abs_cdfGapInt_le_of_charFun`, not from this. -/
theorem abs_cdfGapInt_le_of_charFun_sin (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (hμ : ∀ y : ℝ, Integrable (fun t : ℝ => |y - t|) μ)
    (hν : ∀ y : ℝ, Integrable (fun t : ℝ => |y - t|) ν)
    (hcf : IntegrableOn (fun u : ℝ => ‖charFun μ u - charFun ν u‖ / u ^ 2) (Set.Ioi 0))
    (a x : ℝ) :
    |Threshold.cdfGapInt μ ν a x|
      ≤ (∫ u in Set.Ioi (0:ℝ),
          |Real.sin ((x - a) * u / 2)| * ‖charFun μ u - charFun ν u‖ / u ^ 2) / dirConst := by
  set J : ℝ := ∫ u in Set.Ioi (0:ℝ),
    |Real.sin ((x - a) * u / 2)| * ‖charFun μ u - charFun ν u‖ / u ^ 2 with hJdef
  have hJnn : 0 ≤ J := by
    rw [hJdef]
    exact setIntegral_nonneg measurableSet_Ioi fun u _ => by positivity
  have hInn : 0 ≤ J / dirConst := div_nonneg hJnn dirConst_pos.le
  have htwo : (∫ u in Set.Ioi (0:ℝ),
        2 * |Real.sin ((x - a) * u / 2)| * ‖charFun μ u - charFun ν u‖ / u ^ 2) = 2 * J := by
    rw [hJdef, ← integral_const_mul]
    exact integral_congr_ae (Filter.Eventually.of_forall fun u => by ring)
  rcases le_or_gt a x with hax | hax
  · have hclamp : ∀ ρ : Measure ℝ, ∀ _ : IsProbabilityMeasure ρ,
        Integrable (fun t : ℝ => |x - t|) ρ → Integrable (fun t : ℝ => |a - t|) ρ →
        (∫ t, (max (x - t) 0 - max (a - t) 0) ∂ρ)
          = ((x - a) + ((∫ t, |x - t| ∂ρ) - ∫ t, |a - t| ∂ρ)) / 2 := by
      intro ρ hρ hix hia
      haveI := hρ
      have hpt : ∀ t : ℝ, max (x - t) 0 - max (a - t) 0
          = ((x - a) + (|x - t| - |a - t|)) / 2 := by
        intro t
        rw [max_eq_add_abs_div_two (x - t), max_eq_add_abs_div_two (a - t)]; ring
      have hd : Integrable (fun t : ℝ => |x - t| - |a - t|) ρ := hix.sub hia
      simp_rw [hpt]
      rw [integral_div, integral_add (integrable_const (x - a)) hd, integral_sub hix hia]
      simp
    rw [Threshold.cdfGapInt_eq_integral_clamp_sub μ ν hax, hclamp μ ‹_› (hμ x) (hμ a),
      hclamp ν ‹_› (hν x) (hν a),
      show ((x - a) + ((∫ t, |x - t| ∂μ) - ∫ t, |a - t| ∂μ)) / 2
          - ((x - a) + ((∫ t, |x - t| ∂ν) - ∫ t, |a - t| ∂ν)) / 2
        = (((∫ t, |x - t| ∂μ) - ∫ t, |x - t| ∂ν)
            - ((∫ t, |a - t| ∂μ) - ∫ t, |a - t| ∂ν)) / 2 by ring,
      abs_div, abs_of_nonneg (by norm_num : (0:ℝ) ≤ 2)]
    have h := abs_integral_abs_sub_diff_le μ ν a x (hμ x) (hν x) (hμ a) (hν a) hcf
    rw [htwo] at h
    rw [div_le_iff₀ (by norm_num : (0:ℝ) < 2)]
    have h2 : 2 * J / dirConst = J / dirConst * 2 := by ring
    linarith [h]
  · have hempty : Set.Ioc a x = (∅ : Set ℝ) := Set.Ioc_eq_empty (by linarith)
    rw [Threshold.cdfGapInt, hempty, Measure.restrict_empty, integral_zero_measure, abs_zero]
    exact hInn


/-! ### Elementary integrals used by the two-range estimate -/

/-- `e^{−z} ≤ 2/z²` for `z > 0`, from `1 + z + z²/2 ≤ e^z`. -/
theorem exp_neg_le_two_div_sq {z : ℝ} (hz : 0 < z) : Real.exp (-z) ≤ 2 / z ^ 2 := by
  have hq := Real.quadratic_le_exp_of_nonneg hz.le
  have hge : z ^ 2 / 2 ≤ Real.exp z := by linarith
  have hpos : (0:ℝ) < z ^ 2 / 2 := by positivity
  rw [Real.exp_neg]
  calc (Real.exp z)⁻¹ ≤ (z ^ 2 / 2)⁻¹ := by
        exact inv_anti₀ hpos hge
    _ = 2 / z ^ 2 := by field_simp

/-- `z² ≤ 4·e^{z−2}` for `z ≥ 0`: square `z/2 ≤ e^{z/2 − 1}`, which is `1 + y ≤ e^y` at
`y = z/2 − 1`.  Equality holds at `z = 2` and nowhere else, so this is the sharp quadratic
comparison for the exponential and `4e^{−2}` below cannot be lowered. -/
theorem sq_le_four_mul_exp_sub_two {z : ℝ} (hz : 0 ≤ z) : z ^ 2 ≤ 4 * Real.exp (z - 2) := by
  have h : z / 2 ≤ Real.exp (z / 2 - 1) := by
    have h1 := Real.add_one_le_exp (z / 2 - 1)
    linarith
  have hexp : Real.exp (z - 2) = Real.exp (z / 2 - 1) * Real.exp (z / 2 - 1) := by
    rw [← Real.exp_add]; congr 1; ring
  rw [hexp]
  nlinarith [Real.exp_pos (z / 2 - 1)]

/-- `36/5 < e²`.  The true value is `7.389…`; `36/5 = 7.2` is all that the constant `5/9` of
`exp_neg_le_five_ninths_div_sq` needs, and nothing here evaluates `e²`. -/
theorem exp_two_gt : (36:ℝ) / 5 < Real.exp 2 := by
  have h : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
  have he : Real.exp 2 = Real.exp 1 * Real.exp 1 := by
    rw [← Real.exp_add]; norm_num
  rw [he]
  nlinarith [h]

/-- **The sharp middle-range damping estimate**, `e^{−z} ≤ (5/9)/z²` for `z > 0`.  The extremal
constant for a bound of this shape is `max_{z>0} z²e^{−z} = 4e^{−2} = 0.5413…`, attained at
`z = 2`; the rational `5/9 = 0.5555…` used here loses `2.6%` against it, and **no** `k < 4e^{−2}`
works in `e^{−z} ≤ k/z²`.  Against `exp_neg_le_two_div_sq`, the same inequality with `2` in place
of `5/9` and the version §58b used, this is a factor `18/5 = 3.6`.

What it does **not** do: it is still a pointwise bound on the Gaussian damping factor, so the
`∫₀^∞ u²e^{−u²/(4d)} du = √π·d^{3/2}` that the damped range really contributes is not reached.
The best any split of `(0, ∞)` into `e^{−z} ≤ 1` and `e^{−z} ≤ 4e^{−2}/z²` can give is
`(16/3)(4e^{−2})^{3/4}·∑ⱼvⱼ⁴/√d = 3.36·∑ⱼvⱼ⁴/√d`, against the true `√π·∑ⱼvⱼ⁴/√d = 1.77·…`;
the split at `√d` kept below gives `83/18 = 4.61`.  `exp_neg_le_two_div_sq` is untouched. -/
theorem exp_neg_le_five_ninths_div_sq {z : ℝ} (hz : 0 < z) : Real.exp (-z) ≤ 5 / 9 / z ^ 2 := by
  have hz2 : (0:ℝ) < z ^ 2 := by positivity
  have hez : (0:ℝ) < Real.exp z := Real.exp_pos z
  have hmul : z ^ 2 * Real.exp 2 ≤ 4 * Real.exp z := by
    have h := sq_le_four_mul_exp_sub_two hz.le
    have hsum : Real.exp (z - 2) * Real.exp 2 = Real.exp z := by
      rw [← Real.exp_add]; congr 1; ring
    calc z ^ 2 * Real.exp 2 ≤ 4 * Real.exp (z - 2) * Real.exp 2 :=
          mul_le_mul_of_nonneg_right h (Real.exp_pos 2).le
      _ = 4 * Real.exp z := by rw [mul_assoc, hsum]
  have hfin : z ^ 2 ≤ 5 / 9 * Real.exp z := by
    nlinarith [hmul, mul_pos hz2 (sub_pos.mpr exp_two_gt)]
  rw [le_div_iff₀ hz2, Real.exp_neg, inv_mul_eq_div, div_le_iff₀ hez]
  linarith

theorem integrableOn_const_div_sq {c : ℝ} (hc : 0 < c) (k : ℝ) :
    IntegrableOn (fun u : ℝ => k / u ^ 2) (Set.Ioi c) := by
  have hbase : IntegrableOn (fun u : ℝ => u ^ (-2 : ℝ)) (Set.Ioi c) :=
    integrableOn_Ioi_rpow_of_lt (by norm_num) hc
  have h2 : IntegrableOn (fun u : ℝ => k * u ^ (-2 : ℝ)) (Set.Ioi c) := hbase.const_mul k
  refine h2.congr_fun (fun u hu => ?_) measurableSet_Ioi
  show k * u ^ (-2 : ℝ) = k / u ^ 2
  rw [rpow_neg_two (lt_trans hc hu), mul_one_div]

theorem integral_Ioi_const_div_sq {c : ℝ} (hc : 0 < c) (k : ℝ) :
    ∫ u in Set.Ioi c, k / u ^ 2 = k / c := by
  have hcongr : ∀ u ∈ Set.Ioi c, k / u ^ 2 = k * u ^ (-2 : ℝ) := by
    intro u hu
    rw [rpow_neg_two (lt_trans hc hu)]; ring
  rw [setIntegral_congr_fun measurableSet_Ioi hcongr, integral_const_mul,
    integral_Ioi_rpow_of_lt (by norm_num) hc]
  rw [show (-2 : ℝ) + 1 = -1 by norm_num, Real.rpow_neg_one]
  field_simp

theorem integral_Ioc_sq {b : ℝ} (hb : 0 ≤ b) : ∫ u in Set.Ioc (0:ℝ) b, u ^ 2 = b ^ 3 / 3 := by
  rw [← intervalIntegral.integral_of_le hb, integral_pow]
  norm_num

theorem integrableOn_sq_Ioc (b : ℝ) : IntegrableOn (fun u : ℝ => u ^ 2) (Set.Ioc (0:ℝ) b) :=
  (continuous_pow 2).integrableOn_Ioc


/-! ### §58b  The two-range estimate for the deployed coordinate law

From here on the pair is fixed: `coordLaw H v i` against `N(0, 1/d)` at a **unit** direction `v`.
The only inputs are §44's damped characteristic-function bound and `‖φ‖ ≤ 1`. -/

section Deployed

open Orthogonal Hadamard

variable {d : ℕ}

/-- At a unit direction the Gaussian of §44 is `N(0, 1/d)`. -/
theorem gaussian_unit {v : Ed d} (hv : ‖v‖ = 1) :
    gaussianReal 0 (Real.toNNReal (‖v‖ ^ 2 / (d:ℝ)))
      = gaussianReal 0 (Real.toNNReal (1 / (d:ℝ))) := by
  rw [hv]; norm_num

/-- §44's damped bound at a unit direction, on the range `u·M ≤ √d` that a sup-norm bound `M` on
the coordinates of `v` certifies. -/
theorem norm_gap_le_damped {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d)
    {M u : ℝ} (hM : ∀ j, |v j| ≤ M) (hu : 0 < u) (huM : u * M ≤ Real.sqrt d) :
    ‖charFun (coordLaw H v i) u - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖
      ≤ u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * Real.exp (-(u ^ 2 / (d:ℝ)) / 4) := by
  have ht : ∀ j, |u * v j| ≤ Real.sqrt d := by
    intro j
    rw [abs_mul, abs_of_pos hu]
    exact le_trans (by gcongr; exact hM j) huM
  have h := norm_charFun_coordLaw_sub_gaussian_le_damped hH v i ht
  rw [hv] at h
  norm_num at h
  rw [one_div]
  exact h

theorem norm_charFun_sub_le_two (μ ν : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (u : ℝ) : ‖charFun μ u - charFun ν u‖ ≤ 2 := by
  calc ‖charFun μ u - charFun ν u‖ ≤ ‖charFun μ u‖ + ‖charFun ν u‖ := norm_sub_le _ _
    _ ≤ 1 + 1 := add_le_add (norm_charFun_le_one _) (norm_charFun_le_one _)
    _ = 2 := by norm_num


/-- The `u⁻²`-weighted characteristic-function gap of the pair leg 1 compares: the integrand of
the inversion bound. -/
def gapWeight (H : Od d) (v : Ed d) (i : Fin d) (u : ℝ) : ℝ :=
  ‖charFun (coordLaw H v i) u - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2

theorem gapWeight_nonneg (H : Od d) (v : Ed d) (i : Fin d) (u : ℝ) : 0 ≤ gapWeight H v i u :=
  div_nonneg (norm_nonneg _) (sq_nonneg u)

theorem measurable_gapWeight (H : Od d) (v : Ed d) (i : Fin d) :
    Measurable (gapWeight H v i) := by
  have hc : Continuous fun u : ℝ => charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u :=
    continuous_charFun.sub continuous_charFun
  exact hc.norm.measurable.div (by fun_prop)

/-- **Low range.**  For `0 < u ≤ √d` the damped bound of §44 gives `gapWeight ≤ (∑ⱼvⱼ⁴/(2d²))·u²`;
the damping factor is thrown away here. -/
theorem gapWeight_le_low {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d)
    {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM1 : M ≤ 1) {u : ℝ} (hu : 0 < u)
    (hud : u ≤ Real.sqrt d) :
    gapWeight H v i u ≤ (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * u ^ 2 := by
  have hM0 : 0 ≤ M := le_trans (abs_nonneg _) (hM i)
  have huM : u * M ≤ Real.sqrt d := le_trans (by nlinarith) hud
  have h := norm_gap_le_damped hH hv i hM hu huM
  have hS : 0 ≤ ∑ j, v j ^ 4 := Finset.sum_nonneg fun j _ => by positivity
  have hexp : Real.exp (-(u ^ 2 / (d:ℝ)) / 4) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    have : 0 ≤ u ^ 2 / (d:ℝ) := by positivity
    linarith
  have hA : (0:ℝ) ≤ u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) := by positivity
  rw [gapWeight, div_le_iff₀ (pow_pos hu 2)]
  calc ‖charFun (coordLaw H v i) u - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖
      ≤ u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * Real.exp (-(u ^ 2 / (d:ℝ)) / 4) := h
    _ ≤ u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * 1 := by gcongr
    _ = (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * u ^ 2 * u ^ 2 := by ring

/-- **Middle range.**  For `√d < u` with `u·M ≤ √d`, the Gaussian damping `e^{−u²/(4d)} ≤ 32d²/u⁴`
turns the quartic growth into the integrable `16·∑ⱼvⱼ⁴/u²`. -/
theorem gapWeight_le_mid {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d)
    {M : ℝ} (hM : ∀ j, |v j| ≤ M) {u : ℝ} (hd0 : (0:ℝ) < (d:ℝ)) (hu : 0 < u)
    (huM : u * M ≤ Real.sqrt d) :
    gapWeight H v i u ≤ 16 * (∑ j, v j ^ 4) / u ^ 2 := by
  have h := norm_gap_le_damped hH hv i hM hu huM
  have hS : 0 ≤ ∑ j, v j ^ 4 := Finset.sum_nonneg fun j _ => by positivity
  have hz : (0:ℝ) < u ^ 2 / (4 * (d:ℝ)) := by positivity
  have hexp : Real.exp (-(u ^ 2 / (d:ℝ)) / 4) ≤ 2 / (u ^ 2 / (4 * (d:ℝ))) ^ 2 := by
    have heq : -(u ^ 2 / (d:ℝ)) / 4 = -(u ^ 2 / (4 * (d:ℝ))) := by ring
    rw [heq]
    exact exp_neg_le_two_div_sq hz
  have hA : (0:ℝ) ≤ u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) := by positivity
  have hkey : u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * (2 / (u ^ 2 / (4 * (d:ℝ))) ^ 2)
      = 16 * (∑ j, v j ^ 4) := by
    field_simp
    ring
  rw [gapWeight, div_le_div_iff_of_pos_right (pow_pos hu 2)]
  calc ‖charFun (coordLaw H v i) u - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖
      ≤ u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * Real.exp (-(u ^ 2 / (d:ℝ)) / 4) := h
    _ ≤ u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * (2 / (u ^ 2 / (4 * (d:ℝ))) ^ 2) := by gcongr
    _ = 16 * (∑ j, v j ^ 4) := hkey

/-- **Middle range, sharpened.**  `gapWeight_le_mid` with the extremal damping constant of
`exp_neg_le_five_ninths_div_sq` in place of `exp_neg_le_two_div_sq`: for `√d < u` with
`u·M ≤ √d`, `e^{−u²/(4d)} ≤ (5/9)(4d/u²)²` replaces `16·∑ⱼvⱼ⁴/u²` by `(40/9)·∑ⱼvⱼ⁴/u²`, a
factor `18/5 = 3.6`.

The floor for this argument is `32e^{−2}·∑ⱼvⱼ⁴/u² = 4.33·∑ⱼvⱼ⁴/u²`, since `4e^{−2}` is the
largest value of `z²e^{−z}`; `40/9 = 4.44` is within `2.6%` of it.  Reaching the `√π·∑ⱼvⱼ⁴/√d`
that the Gaussian tail really contributes needs the damping factor integrated rather than bounded
pointwise, which is not done here.  `gapWeight_le_mid` is kept and is not weakened. -/
theorem gapWeight_le_mid_sharp {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d)
    {M : ℝ} (hM : ∀ j, |v j| ≤ M) {u : ℝ} (hd0 : (0:ℝ) < (d:ℝ)) (hu : 0 < u)
    (huM : u * M ≤ Real.sqrt d) :
    gapWeight H v i u ≤ 40 / 9 * (∑ j, v j ^ 4) / u ^ 2 := by
  have h := norm_gap_le_damped hH hv i hM hu huM
  have hS : 0 ≤ ∑ j, v j ^ 4 := Finset.sum_nonneg fun j _ => by positivity
  have hz : (0:ℝ) < u ^ 2 / (4 * (d:ℝ)) := by positivity
  have hexp : Real.exp (-(u ^ 2 / (d:ℝ)) / 4) ≤ 5 / 9 / (u ^ 2 / (4 * (d:ℝ))) ^ 2 := by
    have heq : -(u ^ 2 / (d:ℝ)) / 4 = -(u ^ 2 / (4 * (d:ℝ))) := by ring
    rw [heq]
    exact exp_neg_le_five_ninths_div_sq hz
  have hA : (0:ℝ) ≤ u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) := by positivity
  have hkey : u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * (5 / 9 / (u ^ 2 / (4 * (d:ℝ))) ^ 2)
      = 40 / 9 * (∑ j, v j ^ 4) := by
    field_simp
    ring
  rw [gapWeight, div_le_div_iff_of_pos_right (pow_pos hu 2)]
  calc ‖charFun (coordLaw H v i) u - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖
      ≤ u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * Real.exp (-(u ^ 2 / (d:ℝ)) / 4) := h
    _ ≤ u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * (5 / 9 / (u ^ 2 / (4 * (d:ℝ))) ^ 2) := by
        gcongr
    _ = 40 / 9 * (∑ j, v j ^ 4) := hkey

/-- **High range.**  Beyond `√d/M` nothing is known about the atomic law's characteristic function
and the trivial bound `‖φ_μ − φ_ν‖ ≤ 2` is used; this term is what stops the estimate short of
`Θ(d^{-3/2})`. -/
theorem gapWeight_le_high (H : Od d) (v : Ed d) (i : Fin d) {u : ℝ} (hu : 0 < u) :
    gapWeight H v i u ≤ 2 / u ^ 2 := by
  rw [gapWeight, div_le_div_iff_of_pos_right (pow_pos hu 2)]
  exact norm_charFun_sub_le_two _ _ u


/-! ### Integrability and the two-range integral estimate -/

theorem integrableOn_gapWeight_Ioc {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1)
    (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM1 : M ≤ 1) :
    IntegrableOn (gapWeight H v i) (Set.Ioc 0 (Real.sqrt d)) := by
  have hg : IntegrableOn (fun u : ℝ => (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * u ^ 2)
      (Set.Ioc (0:ℝ) (Real.sqrt d)) := (integrableOn_sq_Ioc _).const_mul _
  refine hg.mono' (measurable_gapWeight H v i).aestronglyMeasurable ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
  rw [Real.norm_eq_abs, abs_of_nonneg (gapWeight_nonneg H v i u)]
  exact gapWeight_le_low hH hv i hM hM1 hu.1 hu.2

theorem integrableOn_gapWeight_Ioi (H : Od d) (v : Ed d) (i : Fin d) (hd0 : (0:ℝ) < (d:ℝ)) :
    IntegrableOn (gapWeight H v i) (Set.Ioi (Real.sqrt d)) := by
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  refine (integrableOn_const_div_sq hsd 2).mono'
    (measurable_gapWeight H v i).aestronglyMeasurable ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
  rw [Real.norm_eq_abs, abs_of_nonneg (gapWeight_nonneg H v i u)]
  exact gapWeight_le_high H v i (lt_trans hsd hu)

theorem Ioi_zero_split (b : ℝ) (hb : 0 < b) : Set.Ioi (0:ℝ) = Set.Ioc 0 b ∪ Set.Ioi b := by
  ext u
  simp only [Set.mem_Ioi, Set.mem_union, Set.mem_Ioc]
  constructor
  · intro h
    rcases le_or_gt u b with h1 | h1
    · exact Or.inl ⟨h, h1⟩
    · exact Or.inr h1
  · rintro (⟨h, -⟩ | h)
    · exact h
    · linarith

theorem Ioi_split {b c : ℝ} (hbc : b ≤ c) : Set.Ioi b = Set.Ioc b c ∪ Set.Ioi c := by
  ext u
  simp only [Set.mem_Ioi, Set.mem_union, Set.mem_Ioc]
  constructor
  · intro h
    rcases le_or_gt u c with h1 | h1
    · exact Or.inl ⟨h, h1⟩
    · exact Or.inr h1
  · rintro (⟨h, -⟩ | h)
    · exact h
    · linarith

theorem integrableOn_gapWeight {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d)
    {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM1 : M ≤ 1) (hd0 : (0:ℝ) < (d:ℝ)) :
    IntegrableOn (gapWeight H v i) (Set.Ioi 0) := by
  rw [Ioi_zero_split _ (Real.sqrt_pos.mpr hd0)]
  exact (integrableOn_gapWeight_Ioc hH hv i hM hM1).union
    (integrableOn_gapWeight_Ioi H v i hd0)

theorem arith_three {S M r : ℝ} (hr : 0 < r) :
    S / (2 * (r ^ 2) ^ 2) * (r ^ 3 / 3) + (16 * S / r + 2 * M / r)
      = (97 / 6 * S + 2 * M) / r := by
  field_simp
  ring

/-- **The two-range estimate.**  For a unit direction `v` whose coordinates are bounded by
`M ≤ 1`, the `u⁻²`-weighted characteristic-function gap of leg 1 integrates to at most
`(97/6·∑ⱼvⱼ⁴ + 2M)/√d`.  The first term is the damped Gaussian comparison of §44 on
`|u| ≤ √d/M`; the second is the trivial bound `‖φ_μ − φ_ν‖ ≤ 2` beyond that range, and it is the
term that dominates. -/
theorem integral_gapWeight_le {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d)
    {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M) (hM1 : M ≤ 1) (hd0 : (0:ℝ) < (d:ℝ)) :
    (∫ u in Set.Ioi (0:ℝ), gapWeight H v i u)
      ≤ (97 / 6 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d := by
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  set S : ℝ := ∑ j, v j ^ 4 with hSdef
  have hS : 0 ≤ S := Finset.sum_nonneg fun j _ => by positivity
  set T : ℝ := Real.sqrt d / M with hTdef
  have hTpos : 0 < T := div_pos hsd hM0
  have hTge : Real.sqrt d ≤ T := by
    rw [hTdef, le_div_iff₀ hM0]
    nlinarith
  have hintA : IntegrableOn (gapWeight H v i) (Set.Ioc 0 (Real.sqrt d)) :=
    integrableOn_gapWeight_Ioc hH hv i hM hM1
  have hintBC : IntegrableOn (gapWeight H v i) (Set.Ioi (Real.sqrt d)) :=
    integrableOn_gapWeight_Ioi H v i hd0
  have hintB : IntegrableOn (gapWeight H v i) (Set.Ioc (Real.sqrt d) T) :=
    hintBC.mono_set fun u hu => hu.1
  have hintC : IntegrableOn (gapWeight H v i) (Set.Ioi T) :=
    hintBC.mono_set fun u hu => lt_of_le_of_lt hTge hu
  have hdisj1 : Disjoint (Set.Ioc (0:ℝ) (Real.sqrt d)) (Set.Ioi (Real.sqrt d)) := by
    rw [Set.disjoint_left]
    intro u hu hu'
    exact absurd hu' (not_lt.mpr hu.2)
  have hdisj2 : Disjoint (Set.Ioc (Real.sqrt d) T) (Set.Ioi T) := by
    rw [Set.disjoint_left]
    intro u hu hu'
    exact absurd hu' (not_lt.mpr hu.2)
  have hE : (∫ u in Set.Ioi (0:ℝ), gapWeight H v i u)
      = (∫ u in Set.Ioc (0:ℝ) (Real.sqrt d), gapWeight H v i u)
        + ((∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u)
          + ∫ u in Set.Ioi T, gapWeight H v i u) := by
    rw [Ioi_zero_split _ hsd, setIntegral_union hdisj1 measurableSet_Ioi hintA hintBC]
    congr 1
    rw [Ioi_split hTge, setIntegral_union hdisj2 measurableSet_Ioi hintB hintC]
  have hA : (∫ u in Set.Ioc (0:ℝ) (Real.sqrt d), gapWeight H v i u)
      ≤ S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3) := by
    have hmaj : IntegrableOn (fun u : ℝ => S / (2 * (d:ℝ) ^ 2) * u ^ 2)
        (Set.Ioc (0:ℝ) (Real.sqrt d)) := (integrableOn_sq_Ioc _).const_mul _
    calc (∫ u in Set.Ioc (0:ℝ) (Real.sqrt d), gapWeight H v i u)
        ≤ ∫ u in Set.Ioc (0:ℝ) (Real.sqrt d), S / (2 * (d:ℝ) ^ 2) * u ^ 2 :=
          setIntegral_mono_on hintA hmaj measurableSet_Ioc fun u hu =>
            gapWeight_le_low hH hv i hM hM1 hu.1 hu.2
      _ = S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3) := by
          rw [integral_const_mul, integral_Ioc_sq hsd.le]
  have hB : (∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u) ≤ 16 * S / Real.sqrt d := by
    have hmaj : IntegrableOn (fun u : ℝ => 16 * S / u ^ 2) (Set.Ioi (Real.sqrt d)) :=
      integrableOn_const_div_sq hsd (16 * S)
    have hmajB : IntegrableOn (fun u : ℝ => 16 * S / u ^ 2) (Set.Ioc (Real.sqrt d) T) :=
      hmaj.mono_set fun u hu => hu.1
    calc (∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u)
        ≤ ∫ u in Set.Ioc (Real.sqrt d) T, 16 * S / u ^ 2 :=
          setIntegral_mono_on hintB hmajB measurableSet_Ioc fun u hu =>
            gapWeight_le_mid hH hv i hM hd0 (lt_trans hsd hu.1) ((le_div_iff₀ hM0).mp hu.2)
      _ ≤ ∫ u in Set.Ioi (Real.sqrt d), 16 * S / u ^ 2 := by
          refine setIntegral_mono_set hmaj ?_ (Filter.Eventually.of_forall fun u hu => hu.1)
          filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
          have : 0 < u := lt_trans hsd hu
          positivity
      _ = 16 * S / Real.sqrt d := integral_Ioi_const_div_sq hsd _
  have hC : (∫ u in Set.Ioi T, gapWeight H v i u) ≤ 2 * M / Real.sqrt d := by
    have hmaj : IntegrableOn (fun u : ℝ => 2 / u ^ 2) (Set.Ioi T) :=
      integrableOn_const_div_sq hTpos 2
    calc (∫ u in Set.Ioi T, gapWeight H v i u)
        ≤ ∫ u in Set.Ioi T, 2 / u ^ 2 :=
          setIntegral_mono_on hintC hmaj measurableSet_Ioi fun u hu =>
            gapWeight_le_high H v i (lt_trans hTpos hu)
      _ = 2 / T := integral_Ioi_const_div_sq hTpos 2
      _ = 2 * M / Real.sqrt d := by rw [hTdef]; field_simp
  have harith := arith_three (S := S) (M := M) hsd
  rw [Real.sq_sqrt hd0.le] at harith
  calc (∫ u in Set.Ioi (0:ℝ), gapWeight H v i u)
      = (∫ u in Set.Ioc (0:ℝ) (Real.sqrt d), gapWeight H v i u)
        + ((∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u)
          + ∫ u in Set.Ioi T, gapWeight H v i u) := hE
    _ ≤ S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3) + (16 * S / Real.sqrt d
          + 2 * M / Real.sqrt d) := add_le_add hA (add_le_add hB hC)
    _ = (97 / 6 * S + 2 * M) / Real.sqrt d := harith


theorem arith_three_sharp {S M r : ℝ} (hr : 0 < r) :
    S / (2 * (r ^ 2) ^ 2) * (r ^ 3 / 3) + (40 / 9 * S / r + 2 * M / r)
      = (83 / 18 * S + 2 * M) / r := by
  field_simp
  ring

/-- **The two-range estimate, sharpened.**  `integral_gapWeight_le` with `gapWeight_le_mid_sharp`
in place of `gapWeight_le_mid`: for a unit direction `v` whose coordinates are bounded by
`M ≤ 1`,

  `∫₀^∞ ‖φ_μ − φ_ν‖/u² du ≤ (83/18·∑ⱼvⱼ⁴ + 2M)/√d`,

against the `(97/6·∑ⱼvⱼ⁴ + 2M)/√d` of `integral_gapWeight_le`.  The spikiness coefficient
improves by `97/6 ÷ 83/18 = 3.51`; `97/6 = 16.17` was `16` from the middle range plus `1/6` from
the low range, and `83/18 = 4.61` is `40/9` plus the same `1/6`.

**The `2M` is untouched.**  It comes from `gapWeight_le_high`, the range `u > √d/M` on which this
development knows nothing about the atomic law's characteristic function beyond `‖φ‖ ≤ 1`, and it
is the term that dominates at a spread direction.  Nothing here improves it, and it is the
largest single piece of slack left on this route.  `integral_gapWeight_le` is kept as it was. -/
theorem integral_gapWeight_le_sharp {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1)
    (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M) (hM1 : M ≤ 1)
    (hd0 : (0:ℝ) < (d:ℝ)) :
    (∫ u in Set.Ioi (0:ℝ), gapWeight H v i u)
      ≤ (83 / 18 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d := by
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  set S : ℝ := ∑ j, v j ^ 4 with hSdef
  have hS : 0 ≤ S := Finset.sum_nonneg fun j _ => by positivity
  set T : ℝ := Real.sqrt d / M with hTdef
  have hTpos : 0 < T := div_pos hsd hM0
  have hTge : Real.sqrt d ≤ T := by
    rw [hTdef, le_div_iff₀ hM0]
    nlinarith
  have hintA : IntegrableOn (gapWeight H v i) (Set.Ioc 0 (Real.sqrt d)) :=
    integrableOn_gapWeight_Ioc hH hv i hM hM1
  have hintBC : IntegrableOn (gapWeight H v i) (Set.Ioi (Real.sqrt d)) :=
    integrableOn_gapWeight_Ioi H v i hd0
  have hintB : IntegrableOn (gapWeight H v i) (Set.Ioc (Real.sqrt d) T) :=
    hintBC.mono_set fun u hu => hu.1
  have hintC : IntegrableOn (gapWeight H v i) (Set.Ioi T) :=
    hintBC.mono_set fun u hu => lt_of_le_of_lt hTge hu
  have hdisj1 : Disjoint (Set.Ioc (0:ℝ) (Real.sqrt d)) (Set.Ioi (Real.sqrt d)) := by
    rw [Set.disjoint_left]
    intro u hu hu'
    exact absurd hu' (not_lt.mpr hu.2)
  have hdisj2 : Disjoint (Set.Ioc (Real.sqrt d) T) (Set.Ioi T) := by
    rw [Set.disjoint_left]
    intro u hu hu'
    exact absurd hu' (not_lt.mpr hu.2)
  have hE : (∫ u in Set.Ioi (0:ℝ), gapWeight H v i u)
      = (∫ u in Set.Ioc (0:ℝ) (Real.sqrt d), gapWeight H v i u)
        + ((∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u)
          + ∫ u in Set.Ioi T, gapWeight H v i u) := by
    rw [Ioi_zero_split _ hsd, setIntegral_union hdisj1 measurableSet_Ioi hintA hintBC]
    congr 1
    rw [Ioi_split hTge, setIntegral_union hdisj2 measurableSet_Ioi hintB hintC]
  have hA : (∫ u in Set.Ioc (0:ℝ) (Real.sqrt d), gapWeight H v i u)
      ≤ S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3) := by
    have hmaj : IntegrableOn (fun u : ℝ => S / (2 * (d:ℝ) ^ 2) * u ^ 2)
        (Set.Ioc (0:ℝ) (Real.sqrt d)) := (integrableOn_sq_Ioc _).const_mul _
    calc (∫ u in Set.Ioc (0:ℝ) (Real.sqrt d), gapWeight H v i u)
        ≤ ∫ u in Set.Ioc (0:ℝ) (Real.sqrt d), S / (2 * (d:ℝ) ^ 2) * u ^ 2 :=
          setIntegral_mono_on hintA hmaj measurableSet_Ioc fun u hu =>
            gapWeight_le_low hH hv i hM hM1 hu.1 hu.2
      _ = S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3) := by
          rw [integral_const_mul, integral_Ioc_sq hsd.le]
  have hB : (∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u) ≤ 40 / 9 * S / Real.sqrt d := by
    have hmaj : IntegrableOn (fun u : ℝ => 40 / 9 * S / u ^ 2) (Set.Ioi (Real.sqrt d)) :=
      integrableOn_const_div_sq hsd (40 / 9 * S)
    have hmajB : IntegrableOn (fun u : ℝ => 40 / 9 * S / u ^ 2) (Set.Ioc (Real.sqrt d) T) :=
      hmaj.mono_set fun u hu => hu.1
    calc (∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u)
        ≤ ∫ u in Set.Ioc (Real.sqrt d) T, 40 / 9 * S / u ^ 2 :=
          setIntegral_mono_on hintB hmajB measurableSet_Ioc fun u hu =>
            gapWeight_le_mid_sharp hH hv i hM hd0 (lt_trans hsd hu.1) ((le_div_iff₀ hM0).mp hu.2)
      _ ≤ ∫ u in Set.Ioi (Real.sqrt d), 40 / 9 * S / u ^ 2 := by
          refine setIntegral_mono_set hmaj ?_ (Filter.Eventually.of_forall fun u hu => hu.1)
          filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
          have : 0 < u := lt_trans hsd hu
          positivity
      _ = 40 / 9 * S / Real.sqrt d := integral_Ioi_const_div_sq hsd _
  have hC : (∫ u in Set.Ioi T, gapWeight H v i u) ≤ 2 * M / Real.sqrt d := by
    have hmaj : IntegrableOn (fun u : ℝ => 2 / u ^ 2) (Set.Ioi T) :=
      integrableOn_const_div_sq hTpos 2
    calc (∫ u in Set.Ioi T, gapWeight H v i u)
        ≤ ∫ u in Set.Ioi T, 2 / u ^ 2 :=
          setIntegral_mono_on hintC hmaj measurableSet_Ioi fun u hu =>
            gapWeight_le_high H v i (lt_trans hTpos hu)
      _ = 2 / T := integral_Ioi_const_div_sq hTpos 2
      _ = 2 * M / Real.sqrt d := by rw [hTdef]; field_simp
  have harith := arith_three_sharp (S := S) (M := M) hsd
  rw [Real.sq_sqrt hd0.le] at harith
  calc (∫ u in Set.Ioi (0:ℝ), gapWeight H v i u)
      = (∫ u in Set.Ioc (0:ℝ) (Real.sqrt d), gapWeight H v i u)
        + ((∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u)
          + ∫ u in Set.Ioi T, gapWeight H v i u) := hE
    _ ≤ S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3) + (40 / 9 * S / Real.sqrt d
          + 2 * M / Real.sqrt d) := add_le_add hA (add_le_add hB hC)
    _ = (83 / 18 * S + 2 * M) / Real.sqrt d := harith


/-! ### Spikiness bookkeeping -/

theorem sum_pow_four_nonneg (v : Ed d) : 0 ≤ ∑ j, v j ^ 4 :=
  Finset.sum_nonneg fun j _ => by positivity

/-- `∑ⱼ vⱼ⁴ ≤ 1` at a unit direction, since `∑ aⱼ² ≤ (∑ aⱼ)²` for nonnegative `aⱼ`. -/
theorem sum_pow_four_le_one {v : Ed d} (hv : ‖v‖ = 1) : (∑ j, v j ^ 4) ≤ 1 := by
  have hsum : ∑ j, v j ^ 2 = 1 := by
    have h := norm_sq_eq_sum_sq v
    rw [hv] at h
    simpa using h.symm
  calc ∑ j, v j ^ 4 = ∑ j, (v j ^ 2) ^ 2 := Finset.sum_congr rfl fun j _ => by ring
    _ ≤ (∑ j, v j ^ 2) ^ 2 := sum_sq_le_sq_sum_of_nonneg fun j _ => sq_nonneg _
    _ = 1 := by rw [hsum]; norm_num

/-- `0 < ∑ⱼ vⱼ⁴` at a unit direction: a vanishing fourth-power sum forces `v = 0`. -/
theorem sum_pow_four_pos {v : Ed d} (hv : ‖v‖ = 1) : 0 < ∑ j, v j ^ 4 := by
  rcases (sum_pow_four_nonneg v).lt_or_eq with h | h
  · exact h
  · exfalso
    have hz : ∀ j ∈ Finset.univ, v j ^ 4 = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg fun j _ => by positivity).mp h.symm
    have hsum : ∑ j, v j ^ 2 = 0 :=
      Finset.sum_eq_zero fun j hj => by have := hz j hj; nlinarith [sq_nonneg (v j)]
    have h2 := norm_sq_eq_sum_sq v
    rw [hv, hsum] at h2
    norm_num at h2

/-- Every coordinate is bounded by `(∑ⱼ vⱼ⁴)^{1/4}`, which is the sup-norm bound the estimate
below consumes when none is supplied. -/
theorem abs_coord_le_sqrt_sqrt (v : Ed d) (j : Fin d) :
    |v j| ≤ Real.sqrt (Real.sqrt (∑ k, v k ^ 4)) := by
  have hS : 0 ≤ ∑ k, v k ^ 4 := sum_pow_four_nonneg v
  have h1 : v j ^ 4 ≤ ∑ k, v k ^ 4 :=
    Finset.single_le_sum (f := fun k => v k ^ 4) (fun k _ => by positivity) (Finset.mem_univ j)
  have h2 : |v j| ^ 2 ≤ Real.sqrt (∑ k, v k ^ 4) := by
    rw [Real.le_sqrt (by positivity) hS]
    calc (|v j| ^ 2) ^ 2 = v j ^ 4 := by rw [sq_abs]; ring
      _ ≤ _ := h1
  rw [Real.le_sqrt (abs_nonneg _) (Real.sqrt_nonneg _)]
  exact h2


/-! ### The two first-moment facts leg 1 needs -/

theorem integrable_abs_sub_coordLaw (H : Od d) (v : Ed d) (i : Fin d) (y : ℝ) :
    Integrable (fun t : ℝ => |y - t|) (coordLaw H v i) :=
  Integrable.abs (Integrable.sub (integrable_const y) (Threshold.integrable_id_coordLaw H v i))

theorem integrable_abs_sub_gaussianReal (σ : NNReal) (y : ℝ) :
    Integrable (fun t : ℝ => |y - t|) (gaussianReal 0 σ) :=
  Integrable.abs (Integrable.sub (integrable_const y)
    (IsGaussian.integrable_fun_id : Integrable (fun t : ℝ => t) (gaussianReal 0 σ)))

end Deployed

end FourierGap

/-! ### §58c  Leg 1

The estimate of §58b, fed through the inversion bound of §58a.  Only the two statements below
leave the `FourierGap` namespace. -/

section Leg1

open Orthogonal Hadamard FourierGap

variable {d : ℕ}

/-- **Leg 1, in `cdfGapInt` form, with an explicit sup-norm bound on the direction.**  For `d ≥ 2`,
a flat `H`, a **unit** direction `v` whose coordinates satisfy `|vⱼ| ≤ M` with `0 < M ≤ 1`, any
coordinate `i` and **every** pair `a`, `x`,

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ (15·∑ⱼvⱼ⁴ + 2M)/√d`.

The route is Fourier inversion of the once-integrated CDF gap
(`abs_cdfGapInt_le_of_charFun`) fed with §44's damped characteristic-function bound on
`|u| ≤ √d/M` and the trivial bound `‖φ_μ − φ_ν‖ ≤ 2` beyond it.

**What this is not.**  It is not a Kolmogorov-Smirnov bound: `|cdf μ − cdf ν|` is not bounded
here, so §55 and §56 in their sup-norm form are not discharged by it.  It is one leg only: the
second law is the Gaussian `N(0, 1/d)`, not `sphericalCoordLaw d`, and nothing here composes it
with `abs_cdfGapInt_sphericalCoordLaw_gaussian_le`.  The constant is not claimed to be optimal;
in particular the kernel constant is used here only through the lower bound `9/8` where its exact
value `π/2` is available as `dirConst_eq` and is used in §58d's primed form of this statement,
and the `2M/√d` term comes from a range beyond which nothing is known about the atomic law's
characteristic function, so the bound decays like `M/√d` and not like the `d^{-3/2}` that the
leading term `15·∑ⱼvⱼ⁴/√d` alone would give at a spread direction. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_of_bound (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M)
    (hM1 : M ≤ 1) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (15 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hS : 0 ≤ ∑ j, v j ^ 4 := sum_pow_four_nonneg v
  have hcf : IntegrableOn (fun u : ℝ => ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2) (Set.Ioi 0) :=
    integrableOn_gapWeight hH hv i hM hM1 hd0
  have hI : (∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2)
      ≤ (97 / 6 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d :=
    integral_gapWeight_le hH hv i hM hM0 hM1 hd0
  have hInn : 0 ≤ ∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2 :=
    setIntegral_nonneg measurableSet_Ioi fun u _ => by positivity
  refine (abs_cdfGapInt_le_of_charFun (coordLaw H v i)
    (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ))))
    (integrable_abs_sub_coordLaw H v i) (integrable_abs_sub_gaussianReal _) hcf a x).trans ?_
  have hstep : (∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2) / dirConst
      ≤ 8 / 9 * ∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
          - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2 := by
    rw [div_le_iff₀ dirConst_pos]
    nlinarith [dirConst_ge, hInn]
  refine hstep.trans ?_
  have hnum : 8 / 9 * (97 / 6 * (∑ j, v j ^ 4) + 2 * M) ≤ 15 * (∑ j, v j ^ 4) + 2 * M := by
    nlinarith [hS, hM0.le]
  calc 8 / 9 * ∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
          - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2
      ≤ 8 / 9 * ((97 / 6 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d) := by gcongr
    _ = (8 / 9 * (97 / 6 * (∑ j, v j ^ 4) + 2 * M)) / Real.sqrt d := by ring
    _ ≤ (15 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d := by gcongr

/-- **Leg 1, in `cdfGapInt` form, in the spikiness alone.**  Taking `M = (∑ⱼ vⱼ⁴)^{1/4}`, which is
a valid coordinate bound at any direction and is `≤ 1` at a unit one,

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ (15·∑ⱼvⱼ⁴ + 2·(∑ⱼvⱼ⁴)^{1/4})/√d`

for every `a` and `x`.  At a maximally spread direction, where `|vⱼ| = d^{-1/2}` and so
`∑ⱼ vⱼ⁴ = 1/d`, the right-hand side is `15·d^{-3/2} + 2·d^{-3/4}`; at a coordinate direction,
where `∑ⱼ vⱼ⁴ = 1`, it is `17/√d`.  Both tend to zero in `d`, which is the point of measuring the
gap by `cdfGapInt` rather than by a sup norm; but the sup-norm hypothesis `η` of §55 and §56 is a
different quantity and is not supplied by this or any statement here.

The caveats of `abs_cdfGapInt_coordLaw_gaussian_le_of_bound` apply verbatim: this is not a
Kolmogorov-Smirnov bound, it is one leg only, it is not composed with §57's leg anywhere, and the
constants are not claimed optimal. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H) {v : Ed d}
    (hv : ‖v‖ = 1) (i : Fin d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (15 * (∑ j, v j ^ 4) + 2 * Real.sqrt (Real.sqrt (∑ j, v j ^ 4))) / Real.sqrt d := by
  have hpos : 0 < ∑ j, v j ^ 4 := sum_pow_four_pos hv
  have hle : (∑ j, v j ^ 4) ≤ 1 := sum_pow_four_le_one hv
  refine abs_cdfGapInt_coordLaw_gaussian_le_of_bound hd hH hv i (abs_coord_le_sqrt_sqrt v)
    (Real.sqrt_pos.mpr (Real.sqrt_pos.mpr hpos)) ?_ a x
  calc Real.sqrt (Real.sqrt (∑ j, v j ^ 4)) ≤ Real.sqrt (Real.sqrt 1) := by
        gcongr
    _ = 1 := by simp

/-- **Leg 1 at a spread direction.**  If every coordinate of the unit direction `v` satisfies
`|vⱼ| ≤ d^{-1/2}`, which is the flat case and the smallest such bound a unit vector admits, then

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ 15·∑ⱼvⱼ⁴/√d + 2/d`

for every `a` and `x`; and `∑ⱼvⱼ⁴ ≤ 1/d` under that hypothesis, so the right-hand side is at most
`15·d^{-3/2} + 2·d^{-1}`.  This is the instance of `abs_cdfGapInt_coordLaw_gaussian_le_of_bound`
at `M = d^{-1/2}` and nothing more; the second term still comes from the range on which the atomic
law's characteristic function is not controlled, and the caveats there apply verbatim. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_of_spread (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (15 * (∑ j, v j ^ 4) + 2 * (1 / Real.sqrt d)) / Real.sqrt d := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have h1 : (1:ℝ) ≤ Real.sqrt d := by
    rw [show (1:ℝ) = Real.sqrt 1 by simp]
    exact Real.sqrt_le_sqrt (by exact_mod_cast (by omega : (1:ℕ) ≤ d))
  exact abs_cdfGapInt_coordLaw_gaussian_le_of_bound hd hH hv i hspread (by positivity)
    (by rw [div_le_one hsd]; exact h1) a x

/-! ### §58d  The same leg with the Dirichlet constant evaluated

`dirConst_eq` replaces the lower bound `9/8` by the exact value `π/2`, so the division by
`dirConst` costs `2/π ≈ 0.6366` rather than `8/9 ≈ 0.8889`.  Each statement below is its unprimed
counterpart with that substitution and nothing else; the unprimed ones are retained unchanged.
The rational constants are rounded up from `(2/π)·(97/6) ≈ 10.292` to `21/2` and from
`(2/π)·2 ≈ 1.273` to `13/10`, which loses a further `2%` in each.  Every caveat recorded on the
unprimed statements applies verbatim: these are not Kolmogorov-Smirnov bounds, they are one leg
only, and the slack listed at `dirConst_eq` is untouched. -/

/-- **Leg 1 with the Dirichlet constant evaluated.**  For `d ≥ 2`, a flat `H`, a unit direction `v`
with `|vⱼ| ≤ M` and `0 < M ≤ 1`, any coordinate `i` and every pair `a`, `x`,

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ ((21/2)·∑ⱼvⱼ⁴ + (13/10)·M)/√d`.

This is `abs_cdfGapInt_coordLaw_gaussian_le_of_bound` with `dirConst_eq` in place of
`dirConst_ge`, improving `15` to `21/2` and `2` to `13/10`.  It is still not a Kolmogorov-Smirnov
bound, still one leg only, and the `M/√d` term still comes from a range beyond which nothing is
known about the atomic law's characteristic function. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_of_bound' (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M)
    (hM1 : M ≤ 1) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (21 / 2 * (∑ j, v j ^ 4) + 13 / 10 * M) / Real.sqrt d := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hS : 0 ≤ ∑ j, v j ^ 4 := sum_pow_four_nonneg v
  have hcf : IntegrableOn (fun u : ℝ => ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2) (Set.Ioi 0) :=
    integrableOn_gapWeight hH hv i hM hM1 hd0
  have hI : (∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2)
      ≤ (97 / 6 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d :=
    integral_gapWeight_le hH hv i hM hM0 hM1 hd0
  refine (abs_cdfGapInt_le_of_charFun (coordLaw H v i)
    (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ))))
    (integrable_abs_sub_coordLaw H v i) (integrable_abs_sub_gaussianReal _) hcf a x).trans ?_
  have hstep : (∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2) / dirConst
      = 2 / π * ∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
          - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2 := by
    rw [dirConst_eq]
    field_simp
  have hkey : 2 / π * (97 / 6 * (∑ j, v j ^ 4) + 2 * M)
      ≤ 21 / 2 * (∑ j, v j ^ 4) + 13 / 10 * M := by
    rw [div_mul_eq_mul_div, div_le_iff₀ Real.pi_pos]
    nlinarith [Real.pi_gt_d2, hS, hM0.le,
      mul_nonneg (by linarith : (0:ℝ) ≤ 21 / 2 * (∑ j, v j ^ 4) + 13 / 10 * M)
        (by linarith [Real.pi_gt_d2] : (0:ℝ) ≤ π - 3.14)]
  rw [hstep]
  calc 2 / π * ∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
          - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2
      ≤ 2 / π * ((97 / 6 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d) :=
        mul_le_mul_of_nonneg_left hI (by positivity)
    _ = (2 / π * (97 / 6 * (∑ j, v j ^ 4) + 2 * M)) / Real.sqrt d := by ring
    _ ≤ (21 / 2 * (∑ j, v j ^ 4) + 13 / 10 * M) / Real.sqrt d := by gcongr

/-- **Leg 1 in the spikiness alone, with the Dirichlet constant evaluated:**

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ ((21/2)·∑ⱼvⱼ⁴ + (13/10)·(∑ⱼvⱼ⁴)^{1/4})/√d`.

The sharpened form of `abs_cdfGapInt_coordLaw_gaussian_le`, whose caveats apply verbatim. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le' (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H) {v : Ed d}
    (hv : ‖v‖ = 1) (i : Fin d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (21 / 2 * (∑ j, v j ^ 4) + 13 / 10 * Real.sqrt (Real.sqrt (∑ j, v j ^ 4)))
          / Real.sqrt d := by
  have hpos : 0 < ∑ j, v j ^ 4 := sum_pow_four_pos hv
  refine abs_cdfGapInt_coordLaw_gaussian_le_of_bound' hd hH hv i (abs_coord_le_sqrt_sqrt v)
    (Real.sqrt_pos.mpr (Real.sqrt_pos.mpr hpos)) ?_ a x
  calc Real.sqrt (Real.sqrt (∑ j, v j ^ 4)) ≤ Real.sqrt (Real.sqrt 1) := by
        gcongr
        exact sum_pow_four_le_one hv
    _ = 1 := by simp

/-- **Leg 1 at a spread direction, with the Dirichlet constant evaluated:**

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ ((21/2)·∑ⱼvⱼ⁴ + (13/10)/√d)/√d`.

The sharpened form of `abs_cdfGapInt_coordLaw_gaussian_le_of_spread`, whose caveats apply
verbatim. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_of_spread' (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (21 / 2 * (∑ j, v j ^ 4) + 13 / 10 * (1 / Real.sqrt d)) / Real.sqrt d := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have h1 : (1:ℝ) ≤ Real.sqrt d := by
    rw [show (1:ℝ) = Real.sqrt 1 by simp]
    exact Real.sqrt_le_sqrt (by exact_mod_cast (by omega : (1:ℕ) ≤ d))
  exact abs_cdfGapInt_coordLaw_gaussian_le_of_bound' hd hH hv i hspread (by positivity)
    (by rw [div_le_one hsd]; exact h1) a x

/-! ### §58c′  Leg 1 with the sharp damping constant

The same three statements with `integral_gapWeight_le_sharp` in place of `integral_gapWeight_le`.
Only the spikiness coefficient moves, `15 → 41/10`; the `2M` term, which dominates at a spread
direction, is identical.  The originals above are kept and are not weakened by these. -/

/-- **Leg 1 with an explicit sup-norm bound, sharpened.**  For `d ≥ 2`, a flat `H`, a unit
direction `v` with `|vⱼ| ≤ M` and `0 < M ≤ 1`, any `i` and **every** pair `a`, `x`,

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ (41/10·∑ⱼvⱼ⁴ + 2M)/√d`,

against `15·∑ⱼvⱼ⁴ + 2M` in `abs_cdfGapInt_coordLaw_gaussian_le_of_bound`: a factor `3.66` on the
spikiness term and nothing at all on `2M`.

**What has not changed.**  It is not a Kolmogorov-Smirnov bound, it is one leg only, and the
route, the hypotheses and the caveats of `abs_cdfGapInt_coordLaw_gaussian_le_of_bound` are those
of this statement too.  `dirConst` still enters only through `9/8 ≤ dirConst` where the true
value is `π/2`, so `41/10` is still about `1.40` times what the sharp kernel constant would give;
`abs_cdfGapInt_coordLaw_gaussian_le_of_bound_sharp'` in §58d′ takes `dirConst_eq` as well and
reaches `3`.  The `2` in `2M` is the trivial `‖φ_μ − φ_ν‖ ≤ 2` on a range where the atomic law's
characteristic function is not controlled at all, and nothing here touches it. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_of_bound_sharp (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M)
    (hM1 : M ≤ 1) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (41 / 10 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hS : 0 ≤ ∑ j, v j ^ 4 := sum_pow_four_nonneg v
  have hcf : IntegrableOn (fun u : ℝ => ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2) (Set.Ioi 0) :=
    integrableOn_gapWeight hH hv i hM hM1 hd0
  have hI : (∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2)
      ≤ (83 / 18 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d :=
    integral_gapWeight_le_sharp hH hv i hM hM0 hM1 hd0
  have hInn : 0 ≤ ∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2 :=
    setIntegral_nonneg measurableSet_Ioi fun u _ => by positivity
  refine (abs_cdfGapInt_le_of_charFun (coordLaw H v i)
    (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ))))
    (integrable_abs_sub_coordLaw H v i) (integrable_abs_sub_gaussianReal _) hcf a x).trans ?_
  have hstep : (∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2) / dirConst
      ≤ 8 / 9 * ∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
          - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2 := by
    rw [div_le_iff₀ dirConst_pos]
    nlinarith [dirConst_ge, hInn]
  refine hstep.trans ?_
  have hnum : 8 / 9 * (83 / 18 * (∑ j, v j ^ 4) + 2 * M) ≤ 41 / 10 * (∑ j, v j ^ 4) + 2 * M := by
    nlinarith [hS, hM0.le]
  calc 8 / 9 * ∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
          - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2
      ≤ 8 / 9 * ((83 / 18 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d) := by gcongr
    _ = (8 / 9 * (83 / 18 * (∑ j, v j ^ 4) + 2 * M)) / Real.sqrt d := by ring
    _ ≤ (41 / 10 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d := by gcongr

/-- **Leg 1 in the spikiness alone, sharpened.**  `M = (∑ⱼvⱼ⁴)^{1/4}` in the statement above:

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ (41/10·∑ⱼvⱼ⁴ + 2·(∑ⱼvⱼ⁴)^{1/4})/√d`

for every `a` and `x`.  At a coordinate direction this reads `61/10/√d` where
`abs_cdfGapInt_coordLaw_gaussian_le` reads `17/√d`; at a maximally spread direction both read
`2·d^{-3/4}` to leading order, since the improvement is entirely in the subdominant term.  The
caveats of `abs_cdfGapInt_coordLaw_gaussian_le` apply verbatim. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_sharp (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (41 / 10 * (∑ j, v j ^ 4) + 2 * Real.sqrt (Real.sqrt (∑ j, v j ^ 4))) / Real.sqrt d := by
  have hpos : 0 < ∑ j, v j ^ 4 := sum_pow_four_pos hv
  have hle : (∑ j, v j ^ 4) ≤ 1 := sum_pow_four_le_one hv
  refine abs_cdfGapInt_coordLaw_gaussian_le_of_bound_sharp hd hH hv i (abs_coord_le_sqrt_sqrt v)
    (Real.sqrt_pos.mpr (Real.sqrt_pos.mpr hpos)) ?_ a x
  calc Real.sqrt (Real.sqrt (∑ j, v j ^ 4)) ≤ Real.sqrt (Real.sqrt 1) := by
        gcongr
    _ = 1 := by simp

/-- **Leg 1 at a spread direction, sharpened.**  If `|vⱼ| ≤ d^{-1/2}` for every `j`, then

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ (41/10·∑ⱼvⱼ⁴ + 2/√d)/√d`

for every `a` and `x`, and with `∑ⱼvⱼ⁴ ≤ 1/d` this is `(41/10/√d + 2)/d`, against
`(15/√d + 2)/d` from `abs_cdfGapInt_coordLaw_gaussian_le_of_spread`.  The `2` is unchanged and is
what the bound tends to: the improvement is `1.09/d` at `d = 512` out of `2.66/d`, so the
dominant `2/d` is where the remaining slack on this route lives.  It comes from the range
`|u| > √d/M`, on which the atomic law's characteristic function is bounded only by `1`. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_of_spread_sharp (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (41 / 10 * (∑ j, v j ^ 4) + 2 * (1 / Real.sqrt d)) / Real.sqrt d := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have h1 : (1:ℝ) ≤ Real.sqrt d := by
    rw [show (1:ℝ) = Real.sqrt 1 by simp]
    exact Real.sqrt_le_sqrt (by exact_mod_cast (by omega : (1:ℕ) ≤ d))
  exact abs_cdfGapInt_coordLaw_gaussian_le_of_bound_sharp hd hH hv i hspread (by positivity)
    (by rw [div_le_one hsd]; exact h1) a x

/-! ### §58d′  Leg 1 with both constants sharpened at once

`dirConst_eq` (§58d) and `exp_neg_le_five_ninths_div_sq` (§58b) are independent improvements, one
on the kernel constant and one on the Gaussian damping, and nothing stops them being taken
together: `integral_gapWeight_le_sharp` fed through `2/π` rather than `8/9`.  Against the
unprimed §58c this is `15 → 3` on the spikiness term and `2 → 13/10` on the term that dominates
at a spread direction.  The rational constants are rounded up from `(2/π)·(83/18) ≈ 2.936` and
`(2/π)·2 ≈ 1.273`, losing a further `2%` each.

**What is still lost.**  The factor `2` at `|G(x) − G(a)| ≤ |G(x)| + |G(a)|`, which §58a′ shows is
sharp for arbitrary pairs of laws and worth at most `1.39` at this particular weight; the gap
between `40/9` and the `√π` the damped range really contributes; and the whole of the `13/10·M`
term, which comes from the range `|u| > √d/M` on which nothing is known about the atomic law's
characteristic function.  These remain not Kolmogorov-Smirnov bounds, and they remain one leg. -/

/-- **Leg 1 with an explicit sup-norm bound, both constants sharpened.**  For `d ≥ 2`, a flat `H`,
a unit direction `v` with `|vⱼ| ≤ M` and `0 < M ≤ 1`, any `i` and **every** pair `a`, `x`,

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ (3·∑ⱼvⱼ⁴ + (13/10)·M)/√d`.

This is `abs_cdfGapInt_coordLaw_gaussian_le_of_bound` with `integral_gapWeight_le_sharp` for
`integral_gapWeight_le` and `dirConst_eq` for `dirConst_ge`, which is
`abs_cdfGapInt_coordLaw_gaussian_le_of_bound_sharp` and
`abs_cdfGapInt_coordLaw_gaussian_le_of_bound'` at once.  All four statements are kept and none
weakens another.  It is still not a Kolmogorov-Smirnov bound, still one leg only, and the
`(13/10)·M` term is still the trivial `‖φ_μ − φ_ν‖ ≤ 2` on the range `|u| > √d/M`. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_of_bound_sharp' (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M)
    (hM1 : M ≤ 1) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (3 * (∑ j, v j ^ 4) + 13 / 10 * M) / Real.sqrt d := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hS : 0 ≤ ∑ j, v j ^ 4 := sum_pow_four_nonneg v
  have hcf : IntegrableOn (fun u : ℝ => ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2) (Set.Ioi 0) :=
    integrableOn_gapWeight hH hv i hM hM1 hd0
  have hI : (∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2)
      ≤ (83 / 18 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d :=
    integral_gapWeight_le_sharp hH hv i hM hM0 hM1 hd0
  refine (abs_cdfGapInt_le_of_charFun (coordLaw H v i)
    (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ))))
    (integrable_abs_sub_coordLaw H v i) (integrable_abs_sub_gaussianReal _) hcf a x).trans ?_
  have hstep : (∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2) / dirConst
      = 2 / π * ∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
          - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2 := by
    rw [dirConst_eq]
    field_simp
  have hkey : 2 / π * (83 / 18 * (∑ j, v j ^ 4) + 2 * M)
      ≤ 3 * (∑ j, v j ^ 4) + 13 / 10 * M := by
    rw [div_mul_eq_mul_div, div_le_iff₀ Real.pi_pos]
    nlinarith [Real.pi_gt_d2, hS, hM0.le,
      mul_nonneg (by linarith : (0:ℝ) ≤ 3 * (∑ j, v j ^ 4) + 13 / 10 * M)
        (by linarith [Real.pi_gt_d2] : (0:ℝ) ≤ π - 3.14)]
  rw [hstep]
  calc 2 / π * ∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
          - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2
      ≤ 2 / π * ((83 / 18 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d) :=
        mul_le_mul_of_nonneg_left hI (by positivity)
    _ = (2 / π * (83 / 18 * (∑ j, v j ^ 4) + 2 * M)) / Real.sqrt d := by ring
    _ ≤ (3 * (∑ j, v j ^ 4) + 13 / 10 * M) / Real.sqrt d := by gcongr

/-- **Leg 1 in the spikiness alone, both constants sharpened:**

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ (3·∑ⱼvⱼ⁴ + (13/10)·(∑ⱼvⱼ⁴)^{1/4})/√d`

for every `a` and `x`, against `15·∑ⱼvⱼ⁴ + 2·(∑ⱼvⱼ⁴)^{1/4}` in
`abs_cdfGapInt_coordLaw_gaussian_le`.  Its caveats apply verbatim. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_sharp' (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (3 * (∑ j, v j ^ 4) + 13 / 10 * Real.sqrt (Real.sqrt (∑ j, v j ^ 4)))
          / Real.sqrt d := by
  have hpos : 0 < ∑ j, v j ^ 4 := sum_pow_four_pos hv
  refine abs_cdfGapInt_coordLaw_gaussian_le_of_bound_sharp' hd hH hv i (abs_coord_le_sqrt_sqrt v)
    (Real.sqrt_pos.mpr (Real.sqrt_pos.mpr hpos)) ?_ a x
  calc Real.sqrt (Real.sqrt (∑ j, v j ^ 4)) ≤ Real.sqrt (Real.sqrt 1) := by
        gcongr
        exact sum_pow_four_le_one hv
    _ = 1 := by simp

/-- **Leg 1 at a spread direction, both constants sharpened:**

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ (3·∑ⱼvⱼ⁴ + (13/10)/√d)/√d`

for every `a` and `x`.  With `∑ⱼvⱼ⁴ ≤ 1/d` this reads `(3/√d + 13/10)/d`, against
`(15/√d + 2)/d` from `abs_cdfGapInt_coordLaw_gaussian_le_of_spread`: at `d = 512` that is
`1.433/d` against `2.663/d`, a factor `1.86`.  The limit as `d → ∞` is `13/10` and not `0`, so
the leg is still `Θ(1/d)` and no better, for the reason recorded at
`abs_cdfGapInt_coordLaw_gaussian_le_of_spread`. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_of_spread_sharp' (hd : 2 ≤ d) {H : Od d}
    (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d)
    (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (3 * (∑ j, v j ^ 4) + 13 / 10 * (1 / Real.sqrt d)) / Real.sqrt d := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have h1 : (1:ℝ) ≤ Real.sqrt d := by
    rw [show (1:ℝ) = Real.sqrt 1 by simp]
    exact Real.sqrt_le_sqrt (by exact_mod_cast (by omega : (1:ℕ) ≤ d))
  exact abs_cdfGapInt_coordLaw_gaussian_le_of_bound_sharp' hd hH hv i hspread (by positivity)
    (by rw [div_le_one hsd]; exact h1) a x

end Leg1

end FourierLeg
