
/-! ## §56  Reading the threshold bound at the pair of laws the paper compares

§55 turns the threshold identity into a bound: for an ordered scalar codebook, any
nearest-codeword map of it, and two probability laws on a common bounded interval with equal
means and equal second moments, a sup-norm bound `η` on `cdf μ − cdf ν` gives

  `|D(q, μ) − D(q, ν)| ≤ 2 · η · spread cbk · (b − a)`.

This section does two things with that bound and nothing else.

* It repackages it for the laws the development actually has, which come with **closed** supports
  rather than the half-open ones §55 wants, and which need not share the interval the codebook's
  thresholds live in.  `abs_distortion_sub_distortion_le_of_Icc` is that repackaging; it is the
  §55 bound plus §55's `compl_Ioc_eq_zero_of_Icc`, and it proves nothing new.
* It discharges every hypothesis of the bound for the pair `(coordLaw H v i, sphericalCoordLaw d)`
  at a **unit** direction `v`, *except one*, and carries the result through §11c's
  `ScalarGap.hadRate_sub_rhoSq` to the deployed per-pass rate.  The hypotheses discharged here
  are the two supports (§52's `coordLaw_compl_Icc_eq_zero` and `sphericalCoordLaw_compl_Icc_eq_zero`),
  the equality of means (both laws are symmetric: `integral_id_coordLaw_eq_zero` and
  `integral_id_sphericalCoordLaw`) and the equality of second moments (`‖v‖²/d = 1/d` when
  `‖v‖ = 1`, by §36's `signAvg_coord_sq` and §4's `integral_sq_sphericalCoordLaw`).

**The hypothesis that is not discharged, and why.**  The one input left standing is the
Kolmogorov-Smirnov bound `η` between the deployed coordinate law and the spherical coordinate
law.  The development does not contain it.  §44's `abs_cdfR_coordLaw_sub_gaussian_le` compares
the coordinate law with a **Gaussian**, `N(0, ‖v‖²/d)`, not with `sphericalCoordLaw d`, and it
does so in the implicit form `|F − Φ| · (1 − 8M/(π√d a)) ≤ …` rather than as a bare sup-norm
bound.  Nothing *here* composes the two legs, and no Kolmogorov-Smirnov bound between
`sphericalCoordLaw d` and that Gaussian exists anywhere in this development, so the sup-norm route
below is not completed at any point.  The route that is completed is the other one: §57 and §58
bound the once-integrated gap `cdfGapInt` for the two legs, and §59 adds them and discharges `hC`
in the `cdfGapInt`-native theorems of this section.  The `η` of the sup-norm statements below
stays a free variable.  Note also that the Gaussian is not compactly supported,
so it does not itself meet the support hypothesis of the bound below.  Accordingly `η` appears as
an explicit hypothesis in every statement of this section, in the `cdfR` form §42 and §44 speak
in, and `cdf_eq_cdfR_coordLaw` and `cdf_eq_cdfR_sphericalCoordLaw` are the only bridge used.

So `abs_hadRate_sub_rhoSq_le` is a **conditional** statement.  It says exactly what a
Kolmogorov-Smirnov bound between the two coordinate laws would buy, namely that the gap between
the deployed per-pass rate of a coordinatewise nearest-codeword stage and the paper's `ρ²` is at
most `2 · η · spread cbk · (b − a) · d`; it does not say that any such `η` is small, and it does
not supply one.  In particular nothing here shows the deployed randomised Hadamard rotation
attains the Haar rate `ρ²`.  The `d` on the right is the one §11c's identity already carries, so
at a fixed `η` the right-hand side grows linearly in `d`; whether it decays at all is a question
about `η`, and no decay of `η` is proved here.  Nothing here is asymptotic, no codebook the paper
ships is named, and `K` is not claimed to be the number of levels of any deployed quantiser. -/

namespace Threshold

open Hadamard Orthogonal CoordLaw

variable {K : ℕ}

/-! ### The bound, for laws given with a closed support

§55's support hypothesis is `ρ (Ioc a b)ᶜ = 0`, half-open on the left; §52 supplies closed
supports.  §55's `compl_Ioc_eq_zero_of_Icc` is the one lemma that converts, and it also widens the
interval on both sides so that the codebook's thresholds and the two laws need not live in exactly
the same interval. -/

/-- **The §55 bound, for two laws with a common closed support.**  Nothing new is proved: the
half-open support is manufactured from the closed one by §55's `compl_Ioc_eq_zero_of_Icc`, and
the conclusion is §55's verbatim.  The interval `[a, b]` that carries the thresholds is only
required to contain `[a', b']` strictly on the left. -/
theorem abs_distortion_sub_distortion_le_of_Icc (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] {a a' b' b η : ℝ} (haa : a < a') (hbb : b' ≤ b)
    (hsupp_mu : μ (Set.Icc a' b')ᶜ = 0) (hsupp_nu : ν (Set.Icc a' b')ᶜ = 0)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν)
    (hks : ∀ x, |(cdf μ) x - (cdf ν) x| ≤ η) :
    |distortion q μ - distortion q ν| ≤ 2 * η * spread cbk * (b - a) :=
  abs_distortion_sub_distortion_le cbk hq μ ν a b hthr
    (compl_Ioc_eq_zero_of_Icc haa hbb hsupp_mu)
    (compl_Ioc_eq_zero_of_Icc haa hbb hsupp_nu) hmean hsq hks

/-- **The §55 uniform-`cdfGapInt` bound, for two laws with a common closed support.**  Nothing new
is proved: the half-open support is manufactured from the closed one exactly as in
`abs_distortion_sub_distortion_le_of_Icc`, and the conclusion is
`abs_distortion_sub_distortion_le_of_cdfGapInt`'s verbatim, with no `(b − a)` factor. -/
theorem abs_distortion_sub_distortion_le_of_Icc_cdfGapInt (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] {a a' b' b C : ℝ} (haa : a < a') (hbb : b' ≤ b)
    (hsupp_mu : μ (Set.Icc a' b')ᶜ = 0) (hsupp_nu : ν (Set.Icc a' b')ᶜ = 0)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν)
    (hC : ∀ x, |cdfGapInt μ ν a x| ≤ C) :
    |distortion q μ - distortion q ν| ≤ 2 * C * spread cbk :=
  abs_distortion_sub_distortion_le_of_cdfGapInt cbk hq μ ν a b hthr
    (compl_Ioc_eq_zero_of_Icc haa hbb hsupp_mu)
    (compl_Ioc_eq_zero_of_Icc haa hbb hsupp_nu) hmean hsq hC

/-! ### Mathlib's `cdf` is §42's `cdfR` on both laws

§42 and §44 state their distribution-function bounds in terms of `cdfR`, the `Iic` mass; §55's
bound is stated in terms of Mathlib's `cdf`.  On a probability measure the two agree by
`cdf_eq_real`, and that is the whole bridge. -/

/-- On the deployed coordinate law, Mathlib's `cdf` is §42's `cdfR`. -/
theorem cdf_eq_cdfR_coordLaw {d : ℕ} (H : Od d) (v : Ed d) (i : Fin d) (x : ℝ) :
    (cdf (coordLaw H v i)) x = cdfR (coordLaw H v i) x := cdf_eq_real _ x

/-- On the spherical coordinate law, Mathlib's `cdf` is §42's `cdfR`.  The probability-measure
instance is a theorem in §4b, not an instance, so `2 ≤ d` has to be supplied. -/
theorem cdf_eq_cdfR_sphericalCoordLaw {d : ℕ} (hd : 2 ≤ d) (x : ℝ) :
    (cdf (sphericalCoordLaw d)) x = cdfR (sphericalCoordLaw d) x := by
  haveI := isProbabilityMeasure_sphericalCoordLaw hd
  exact cdf_eq_real _ x

/-! ### The first two moments of the two laws

Equal means and equal second moments are hypotheses of the threshold identity, and for this pair
they are theorems.  Both laws are symmetric, so both means are zero; the second moments are
`‖v‖²/d` and `1/d`, which agree exactly when `‖v‖ = 1`. -/

/-- **The deployed coordinate law has mean zero.**  Flipping every sign negates the coordinate
and preserves the uniform law on the sign cube, which is §40's `signAvg_odd`.  No flatness and no
hypothesis on `d` is used. -/
theorem integral_id_coordLaw_eq_zero {d : ℕ} (H : Od d) (v : Ed d) (i : Fin d) :
    ∫ t, t ∂(coordLaw H v i) = 0 := by
  rw [integral_coordLaw H v i (g := fun t : ℝ => t) measurable_id]
  simp_rw [hadRot_apply_eq_radSum]
  exact signAvg_odd (fun j => (H : Matrix (Fin d) (Fin d) ℝ) i j * v j) (F := fun x : ℝ => x)
    (fun _ => rfl)

/-- **The second moment of the deployed coordinate law is `‖v‖²/d`**, for a flat `H`.  This is
§36's `signAvg_coord_sq` read through `integral_coordLaw`. -/
theorem integral_sq_coordLaw_eq {d : ℕ} {H : Od d} (hH : IsFlat H) (v : Ed d) (i : Fin d) :
    ∫ t, t ^ 2 ∂(coordLaw H v i) = ‖v‖ ^ 2 / d := by
  rw [integral_coordLaw H v i (g := fun t : ℝ => t ^ 2) (measurable_id.pow_const 2)]
  exact signAvg_coord_sq hH v i

/-- **The spherical coordinate law has mean zero**, its density being even.  This is §4's
`sphericalPdf_first_moment` pushed from an interval integral to an integral against the measure,
exactly as `integral_sq_sphericalCoordLaw` does for the second moment. -/
theorem integral_id_sphericalCoordLaw {d : ℕ} (hd : 2 ≤ d) :
    ∫ t, t ∂(sphericalCoordLaw d) = 0 := by
  have hmeas : Measurable (sphericalPdf d) := by
    unfold sphericalPdf
    fun_prop
  have hrestr : (volume : Measure ℝ).restrict (Set.Icc (-1 : ℝ) 1)
      = (volume : Measure ℝ).restrict (Set.Ioc (-1 : ℝ) 1) :=
    (Measure.restrict_congr_set Ioc_ae_eq_Icc).symm
  rw [sphericalCoordLaw,
    integral_withDensity_eq_integral_toReal_smul
      (f := fun t => ENNReal.ofReal (sphericalPdf d t))
      (ENNReal.measurable_ofReal.comp hmeas)
      (Filter.Eventually.of_forall fun _ => ENNReal.ofReal_lt_top)]
  rw [show (∫ t, (ENNReal.ofReal (sphericalPdf d t)).toReal • t
        ∂(volume.restrict (Set.Icc (-1 : ℝ) 1)))
      = ∫ t, t * sphericalPdf d t ∂(volume.restrict (Set.Icc (-1 : ℝ) 1)) from
    setIntegral_congr_fun measurableSet_Icc fun t ht => by
      rw [ENNReal.toReal_ofReal (sphericalPdf_nonneg ht hd), smul_eq_mul, mul_comm]]
  rw [hrestr, ← intervalIntegral.integral_of_le (by norm_num : (-1 : ℝ) ≤ 1)]
  exact sphericalPdf_first_moment hd

/-! ### The conditional bound at the deployed pair -/

/-- **The distortion gap between the deployed coordinate law and the spherical coordinate law,
given a Kolmogorov-Smirnov bound between them.**  For a flat `H`, a **unit** direction `v`, any
nearest-codeword map of an ordered codebook whose thresholds lie in `[a, b]` with
`a < −‖v‖` and `‖v‖ ≤ b`, and any `η` bounding `|cdfR (coordLaw H v i) − cdfR (sphericalCoordLaw d)|`
everywhere,

  `|D(q, coordLaw H v i) − D(q, sphericalCoordLaw d)| ≤ 2 · η · spread cbk · (b − a)`.

Every other hypothesis of §55 is discharged here: both supports, equal means and equal second
moments.  The hypothesis `hks` is **not** supplied by this development; §44 bounds the distance
from the coordinate law to a Gaussian, which is a different pair, so this statement is
conditional on a bound the development does not have.  `‖v‖ = 1` is used twice, for the common
support and to make the two second moments both `1/d`. -/
theorem abs_distortion_coordLaw_sub_sphericalCoordLaw_le {d : ℕ} (hd : 2 ≤ d) {H : Od d}
    (hH : IsFlat H) (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) {a b η : ℝ} (ha : a < -‖v‖) (hb : ‖v‖ ≤ b)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hks : ∀ x, |cdfR (coordLaw H v i) x - cdfR (sphericalCoordLaw d) x| ≤ η) :
    |distortion q (coordLaw H v i) - distortion q (sphericalCoordLaw d)|
      ≤ 2 * η * spread cbk * (b - a) := by
  haveI := isProbabilityMeasure_sphericalCoordLaw hd
  have hsupp_mu : (coordLaw H v i) (Set.Icc (-1 : ℝ) 1)ᶜ = 0 := by
    have h := coordLaw_compl_Icc_eq_zero H v i
    rwa [hv] at h
  have hsupp_nu : (sphericalCoordLaw d) (Set.Icc (-1 : ℝ) 1)ᶜ = 0 :=
    sphericalCoordLaw_compl_Icc_eq_zero d
  have hmean : ∫ t, t ∂(coordLaw H v i) = ∫ t, t ∂(sphericalCoordLaw d) := by
    rw [integral_id_coordLaw_eq_zero H v i, integral_id_sphericalCoordLaw hd]
  have hsq : ∫ t, t ^ 2 ∂(coordLaw H v i) = ∫ t, t ^ 2 ∂(sphericalCoordLaw d) := by
    rw [integral_sq_coordLaw_eq hH v i, integral_sq_sphericalCoordLaw hd, hv]
    norm_num
  have hcdf : ∀ x, |(cdf (coordLaw H v i)) x - (cdf (sphericalCoordLaw d)) x| ≤ η := by
    intro x
    rw [cdf_eq_cdfR_coordLaw H v i x, cdf_eq_cdfR_sphericalCoordLaw hd x]
    exact hks x
  refine abs_distortion_sub_distortion_le_of_Icc cbk hq _ _ ?_ ?_ hsupp_mu hsupp_nu hthr
    hmean hsq hcdf
  · rwa [hv] at ha
  · rwa [hv] at hb

/-- **The distortion gap between the deployed coordinate law and the spherical coordinate law,
given a uniform bound on `cdfGapInt`.**  Same hypotheses as
`abs_distortion_coordLaw_sub_sphericalCoordLaw_le` except that `hks`, a sup-norm bound on
`cdfR (coordLaw H v i) − cdfR (sphericalCoordLaw d)`, is replaced by `hC`, a bound on
`|cdfGapInt (coordLaw H v i) (sphericalCoordLaw d) a x|` uniform in `x`, and the conclusion drops
the `(b − a)` factor:

  `|D(q, coordLaw H v i) − D(q, sphericalCoordLaw d)| ≤ 2 · C · spread cbk`.

Every support and moment hypothesis is discharged exactly as before; only the estimate changes.
`hC` is **not** supplied by this development. -/
theorem abs_distortion_coordLaw_sub_sphericalCoordLaw_le_of_cdfGapInt {d : ℕ} (hd : 2 ≤ d)
    {H : Od d} (hH : IsFlat H) (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) (cbk : Codebook K)
    {q : ℝ → ℝ} (hq : IsNearest (Set.range cbk.c) q) {a b C : ℝ} (ha : a < -‖v‖) (hb : ‖v‖ ≤ b)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hC : ∀ x, |cdfGapInt (coordLaw H v i) (sphericalCoordLaw d) a x| ≤ C) :
    |distortion q (coordLaw H v i) - distortion q (sphericalCoordLaw d)|
      ≤ 2 * C * spread cbk := by
  haveI := isProbabilityMeasure_sphericalCoordLaw hd
  have hsupp_mu : (coordLaw H v i) (Set.Icc (-1 : ℝ) 1)ᶜ = 0 := by
    have h := coordLaw_compl_Icc_eq_zero H v i
    rwa [hv] at h
  have hsupp_nu : (sphericalCoordLaw d) (Set.Icc (-1 : ℝ) 1)ᶜ = 0 :=
    sphericalCoordLaw_compl_Icc_eq_zero d
  have hmean : ∫ t, t ∂(coordLaw H v i) = ∫ t, t ∂(sphericalCoordLaw d) := by
    rw [integral_id_coordLaw_eq_zero H v i, integral_id_sphericalCoordLaw hd]
  have hsq : ∫ t, t ^ 2 ∂(coordLaw H v i) = ∫ t, t ^ 2 ∂(sphericalCoordLaw d) := by
    rw [integral_sq_coordLaw_eq hH v i, integral_sq_sphericalCoordLaw hd, hv]
    norm_num
  refine abs_distortion_sub_distortion_le_of_Icc_cdfGapInt cbk hq _ _ ?_ ?_ hsupp_mu hsupp_nu hthr
    hmean hsq hC
  · rwa [hv] at ha
  · rwa [hv] at hb

/-- **The deployed per-pass rate against `ρ²`, given a Kolmogorov-Smirnov bound.**  §11c's
`ScalarGap.hadRate_sub_rhoSq` makes the gap between the deployed rate of a coordinatewise stage
and `ρ² = d · D(q, sphericalCoordLaw d)` exactly `d` times a scalar distortion gap; the bound
above bounds that gap.  Hence

  `|hadRate H (blockStage q) v − ρ²| ≤ 2 · η · spread cbk · (b − a) · d`.

This is conditional on `hks` exactly as the statement above is, and the development does not
supply `hks`.  The factor `d` is the one §11c's identity already carries, so at a fixed `η` the
right-hand side grows linearly in `d`; no decay of `η` is proved here, and no claim is made that
the deployed rotation attains `ρ²`.  Flatness of `H`, measurability of `q` and `‖v‖ = 1` are all
used. -/
theorem abs_hadRate_sub_rhoSq_le {d : ℕ} (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H) (v : Ed d)
    (hv : ‖v‖ = 1) (i : Fin d) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b η : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hks : ∀ x, |cdfR (coordLaw H v i) x - cdfR (sphericalCoordLaw d) x| ≤ η) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ 2 * η * spread cbk * (b - a) * (d : ℝ) := by
  have hgap := abs_distortion_coordLaw_sub_sphericalCoordLaw_le hd hH v hv i cbk hq ha hb hthr hks
  have hid : distortion q (radLaw d v) = distortion q (coordLaw H v i) := by
    rw [coordLaw_eq_radLaw hH v i]
  have hd0 : (0 : ℝ) ≤ (d : ℝ) := Nat.cast_nonneg d
  rw [ScalarGap.hadRate_sub_rhoSq hH hqm v, hid, abs_mul, abs_of_nonneg hd0]
  calc (d : ℝ) * |distortion q (coordLaw H v i) - distortion q (sphericalCoordLaw d)|
      ≤ (d : ℝ) * (2 * η * spread cbk * (b - a)) := mul_le_mul_of_nonneg_left hgap hd0
    _ = 2 * η * spread cbk * (b - a) * (d : ℝ) := by ring

/-- **The deployed per-pass rate against `ρ²`, given a uniform bound on `cdfGapInt`.**  Same
hypotheses as `abs_hadRate_sub_rhoSq_le` except that `hks`, a sup-norm bound, is replaced by `hC`,
a bound on `|cdfGapInt (coordLaw H v i) (sphericalCoordLaw d) a x|` uniform in `x`.  §11c's
`ScalarGap.hadRate_sub_rhoSq` still supplies the factor `d`, but there is no factor `(b − a)`:

  `|hadRate H (blockStage q) v − ρ²| ≤ 2 · C · spread cbk · d`.

At a fixed `C` the right side still grows linearly in `d` through the explicit factor, but unlike
the `η · (b − a)` shape, a `C` that itself decays like `1/d` makes the whole right side
`Θ(spread cbk)` rather than growing.  `hC` is **not** discharged in this section, and the theorem
is stated conditionally here.  It is discharged in §59, by adding §57's coupling leg
(`sphericalCoordLaw d` against `N(0, 1/d)`) to §58's Fourier leg (`coordLaw H v i` against the
same Gaussian); `Threshold.abs_hadRate_sub_rhoSq_le_unconditional` and
`Threshold.abs_hadRate_sub_rhoSq_le_of_spread` are the results, the second of which is
`Θ(spread cbk)`.  Nothing in this section proves any of that, and even there no claim is made
that the deployed randomised Hadamard rotation attains the Haar rate `ρ²`. -/
theorem abs_hadRate_sub_rhoSq_le_of_cdfGapInt {d : ℕ} (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b C : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hC : ∀ x, |cdfGapInt (coordLaw H v i) (sphericalCoordLaw d) a x| ≤ C) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ 2 * C * spread cbk * (d : ℝ) := by
  have hgap := abs_distortion_coordLaw_sub_sphericalCoordLaw_le_of_cdfGapInt hd hH v hv i cbk hq
    ha hb hthr hC
  have hid : distortion q (radLaw d v) = distortion q (coordLaw H v i) := by
    rw [coordLaw_eq_radLaw hH v i]
  have hd0 : (0 : ℝ) ≤ (d : ℝ) := Nat.cast_nonneg d
  rw [ScalarGap.hadRate_sub_rhoSq hH hqm v, hid, abs_mul, abs_of_nonneg hd0]
  calc (d : ℝ) * |distortion q (coordLaw H v i) - distortion q (sphericalCoordLaw d)|
      ≤ (d : ℝ) * (2 * C * spread cbk) := mul_le_mul_of_nonneg_left hgap hd0
    _ = 2 * C * spread cbk * (d : ℝ) := by ring

/-- The same, for the explicit step quantiser of the codebook: a check that the hypotheses of
`abs_hadRate_sub_rhoSq_le` are met by something, since `stepQuant` is a nearest-codeword map
(§52's `isNearest_stepQuant`) and is measurable (§52's `measurable_stepQuant`).  It is still
conditional on `hks`. -/
theorem abs_hadRate_stepQuant_sub_rhoSq_le {d : ℕ} (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) (cbk : Codebook K) {a b η : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hks : ∀ x, |cdfR (coordLaw H v i) x - cdfR (sphericalCoordLaw d) x| ≤ η) :
    |hadRate H (blockStage (m := d) (F := ℝ) (stepQuant cbk)) v
        - rhoSq d (stepQuant cbk) (sphericalCoordLaw d)|
      ≤ 2 * η * spread cbk * (b - a) * (d : ℝ) :=
  abs_hadRate_sub_rhoSq_le hd hH v hv i cbk (isNearest_stepQuant cbk) (measurable_stepQuant cbk)
    ha hb hthr hks

end Threshold
