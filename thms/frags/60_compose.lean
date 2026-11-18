
/-! ## §59  Composing the two legs, and the deployed rate without a Kolmogorov-Smirnov hypothesis

§57 and §58 bound the once-integrated CDF gap `cdfGapInt` between each of the two laws the paper
compares and one and the same reference Gaussian `N(0, 1/d)`:

* §57, by coupling `u = Z/‖Z‖` against `Z/√d`:
  `|cdfGapInt (f_d) (N(0,1/d)) a x| ≤ √(1 + 1/(4(d−1)))/d`, which is `Θ(1/d)`;
* §58, by Fourier inversion of the integrated gap:
  `|cdfGapInt (coordLaw H v i) (N(0,1/d)) a x| ≤ (15·∑ⱼvⱼ⁴ + 2·(∑ⱼvⱼ⁴)^{1/4})/√d`, and in §58d's
  primed form, with the Dirichlet constant evaluated, `((21/2)·∑ⱼvⱼ⁴ + (13/10)·(∑ⱼvⱼ⁴)^{1/4})/√d`.

Both are uniform in `a` and `x`.  This section does the one step that was deliberately left out of
each: it adds them.  `cdfGapInt μ ν a x` is a difference of two once-integrated CDFs
(`cdfGapInt_eq_sub`), so it telescopes through any third law with no hypothesis whatever
(`cdfGapInt_middle`), and the Gaussian, which is the middle law here, drops out.  That the
Gaussian is not compactly supported and does not meet §55's support hypothesis costs nothing: it
never appears in the composed statement.

The composed bound is then fed to §56's `abs_hadRate_sub_rhoSq_le_of_cdfGapInt`, whose hypothesis
`hC` was the last undischarged input on that route.  The result,
`abs_hadRate_sub_rhoSq_le_unconditional`, has **no** `η` and no `C`: every quantity on the right
is an explicit function of `d`, of the direction's spikiness `∑ⱼvⱼ⁴`, and of `spread cbk`.

**What the rate is, and when it vanishes.**  Two statements, because they behave differently.

* In general, with `S = ∑ⱼvⱼ⁴ ∈ (0, 1]`, the bound is
  `2·spread cbk·(√d·(15S + 2S^{1/4}) + √(1 + 1/(4(d−1))))`.
  The factor `d` supplied by §11c's identity has cancelled one of the two `1/d`'s, but not both:
  at a maximally spiky direction (`v` a coordinate vector, `S = 1`) this is `Θ(√d·spread cbk)`,
  which does **not** tend to zero for a codebook at the coordinate scale.  Leg 1's rate is
  `Θ(S/√d + S^{1/4}/√d)`, not `Θ(1/d)`, and at `S = 1` that is only `Θ(d^{-1/2})`.
* At a spread direction, `|vⱼ| ≤ d^{-1/2}` for every `j`, which is the flat case and the smallest
  sup norm a unit vector admits, `S ≤ 1/d` and the composed `cdfGapInt` bound is
  `(15/√d + 7/2)/d`, which is `Θ(1/d)`: the same order as §57's leg, so leg 1 no longer dominates.
  The deployed-rate bound is then `(30/√d + 7)·spread cbk`, which is `Θ(spread cbk)`; the
  constant does not decay, being `8.33` at `d = 512` and never below `7`.  With §58d's evaluated
  Dirichlet constant the primed form is `((21/2)/√d + 14/5)/d` and `(21/√d + 28/5)·spread cbk`,
  whose constant is `6.53` at `d = 512` and never below `5.6`.  The order in `d` is unchanged.

**`spread cbk` is left abstract, so the vanishing is conditional on the codebook.**  Nothing here
constrains `cbk`.  `spread cbk = cbk.c (Fin.last K) − cbk.c 0` is the width of the codeword range,
and the statements below are uniform in it.  If, and only if, the codebook is scaled to the
coordinate scale so that `spread cbk → 0`, does `(30/√d + 7)·spread cbk → 0`, and likewise
`(21/√d + 28/5)·spread cbk`; a codebook of fixed width gives a bound that tends to `7·spread cbk`,
or to `(28/5)·spread cbk` in the primed form, and not to zero.  The theorems below prove the
inequality, not any scaling of `cbk`, and no codebook the paper ships is named or analysed here.

**What is still not claimed.**  The deployed randomised Hadamard rotation is not shown to attain
the Haar rate `ρ²`; a bound on `|hadRate − ρ²|` that is `Θ(spread cbk)` is not the statement that
the difference is zero.  The constants are not sharp.  The unprimed ones use §58's `dirConst`
only through `9/8`, where its exact value `π/2` is now available (`dirConst_eq`), so `15` and `2`
are each at most about `1.40` times what that step allows; the primed statements below substitute
the exact value and reach `21/2` and `13/10`.  That is the only slack removed.  The `3/2` used
for `√(1 + 1/(4(d−1)))` below, whose true value is under `1.001` at `d = 512`, is untouched and is
the larger remaining loss in the spread constant; so are the factor `2` in §58a's inversion and
the crude `e^{−z} ≤ 2/z²` inside §58b.  The
hypotheses `a < −‖v‖`, `‖v‖ ≤ b` and `hthr` survive, since the threshold identity needs a window
containing both supports and every threshold; but `(b − a)` does not appear in any conclusion, and
no sup-norm bound on `cdf μ − cdf ν` is assumed anywhere on this route. -/

noncomputable section ComposeLegs

namespace Threshold

/-! ### `cdfGapInt` telescopes through a third law

§54's `cdfGapInt_add` is additivity in the *interval*, `(a, x] ∪ (x, y]`.  What the triangle
inequality needs is additivity in the middle *measure*, which is a different statement and is
proved here.  It is immediate from `cdfGapInt_eq_sub` and needs no integrability, no support and
no probability hypothesis: both sides are differences of the same three once-integrated CDFs. -/

/-- **`cdfGapInt` telescopes through any third law.**  For all measures `μ`, `σ`, `ν` on `ℝ` and
all reals `a`, `x`,

  `cdfGapInt μ ν a x = cdfGapInt μ σ a x + cdfGapInt σ ν a x`.

No hypothesis on `σ` is needed, in particular no support hypothesis: `σ` cancels. -/
theorem cdfGapInt_middle (μ σ ν : Measure ℝ) (a x : ℝ) :
    cdfGapInt μ ν a x = cdfGapInt μ σ a x + cdfGapInt σ ν a x := by
  simp only [cdfGapInt_eq_sub]; ring

/-- `cdfGapInt` is antisymmetric in its two laws. -/
theorem cdfGapInt_swap (μ ν : Measure ℝ) (a x : ℝ) :
    cdfGapInt ν μ a x = -cdfGapInt μ ν a x := by
  simp only [cdfGapInt_eq_sub]; ring

/-- **The triangle inequality for `cdfGapInt`, against a common third law.**  If `μ` and `ν` are
each within `C₁` and `C₂` of the same `σ` in the `cdfGapInt` sense at one pair `(a, x)`, then `μ`
and `ν` are within `C₁ + C₂` of each other at that pair.  The two legs of §57 and §58 are both
stated with the Gaussian in second position, which is why the hypotheses are too.  This is
`cdfGapInt_middle` plus `cdfGapInt_swap` and nothing else; no property of `σ` is used, so the fact
that the Gaussian below is not compactly supported is irrelevant. -/
theorem abs_cdfGapInt_le_add_of_common {μ ν σ : Measure ℝ} {a x C₁ C₂ : ℝ}
    (h₁ : |cdfGapInt μ σ a x| ≤ C₁) (h₂ : |cdfGapInt ν σ a x| ≤ C₂) :
    |cdfGapInt μ ν a x| ≤ C₁ + C₂ := by
  have h : cdfGapInt μ ν a x = cdfGapInt μ σ a x - cdfGapInt ν σ a x := by
    simp only [cdfGapInt_eq_sub]; ring
  obtain ⟨h₁a, h₁b⟩ := abs_le.mp h₁
  obtain ⟨h₂a, h₂b⟩ := abs_le.mp h₂
  rw [h, abs_le]
  constructor <;> linarith

end Threshold

/-! ### The two arithmetic facts the spread case needs -/

open Hadamard Orthogonal CoordLaw

variable {d : ℕ}

/-- **A spread unit direction has spikiness at most `1/d`.**  If `|vⱼ| ≤ d^{-1/2}` for every `j`
and `‖v‖ = 1`, then `∑ⱼ vⱼ⁴ ≤ 1/d`.  The hypothesis does more than it looks: with `‖v‖ = 1` it
forces `|vⱼ| = d^{-1/2}` for every `j`, since `d` terms each at most `1/d` summing to `1` must
each equal `1/d`.  So it selects the `2^d` sign patterns of the flat direction rather than a
neighbourhood of them, and the inequality is an equality wherever it applies.  Only the
inequality is proved, since that is all the estimate consumes.  This is the one place the
spread hypothesis is converted into a bound on the quantity §58's leg is stated in. -/
theorem sum_pow_four_le_inv_of_spread (hd : 2 ≤ d) {v : Ed d} (hv : ‖v‖ = 1)
    (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d) : (∑ j, v j ^ 4) ≤ 1 / (d : ℝ) := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsum : ∑ j, v j ^ 2 = 1 := by
    have h := norm_sq_eq_sum_sq v
    rw [hv] at h
    simpa using h.symm
  have hcoord : ∀ j, v j ^ 2 ≤ 1 / (d:ℝ) := by
    intro j
    have h := hspread j
    have hnn := abs_nonneg (v j)
    have hsq : |v j| ^ 2 ≤ (1 / Real.sqrt d) ^ 2 := by nlinarith
    calc v j ^ 2 = |v j| ^ 2 := (sq_abs _).symm
      _ ≤ (1 / Real.sqrt d) ^ 2 := hsq
      _ = 1 / (d:ℝ) := by rw [div_pow, one_pow, Real.sq_sqrt hd0.le]
  calc ∑ j, v j ^ 4 = ∑ j, v j ^ 2 * v j ^ 2 := Finset.sum_congr rfl fun j _ => by ring
    _ ≤ ∑ j, (1 / (d:ℝ)) * v j ^ 2 :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_right (hcoord j) (sq_nonneg _)
    _ = (1 / (d:ℝ)) * ∑ j, v j ^ 2 := by rw [← Finset.mul_sum]
    _ = 1 / (d:ℝ) := by rw [hsum, mul_one]

/-- A crude bound on §57's coupling constant: `√(1 + 1/(4(d−1))) ≤ 3/2` for `d ≥ 2`.  The sharp
value at `d = 2` is `√(5/4) ≈ 1.118` and the constant decreases to `1`, so `3/2` loses about 34%
at the worst dimension and about 50% in the limit.  It is used only to turn the composed bound
into a single readable fraction. -/
theorem sqrt_one_add_quarter_le (hd : 2 ≤ d) :
    Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) ≤ 3 / 2 := by
  have hd1 : (1:ℝ) ≤ (d:ℝ) - 1 := by
    have : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
    linarith
  have hquarter : 1 / (4 * ((d:ℝ) - 1)) ≤ 1 / 4 :=
    div_le_div_of_nonneg_left (by norm_num) (by norm_num) (by linarith)
  have hle : 1 + 1 / (4 * ((d:ℝ) - 1)) ≤ (3/2 : ℝ) ^ 2 := by
    have hnine : ((3:ℝ)/2) ^ 2 = 9 / 4 := by norm_num
    rw [hnine]
    linarith
  have := Real.sqrt_le_sqrt hle
  rwa [Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 3/2)] at this

/-! ### The composed leg: deployed coordinate law against spherical coordinate law -/

/-- **The two legs, composed.**  For `d ≥ 2`, a flat `H`, a unit direction `v` and any coordinate
`i`,

  `|cdfGapInt (coordLaw H v i) (f_d) a x| ≤ (15·S + 2·S^{1/4})/√d + √(1 + 1/(4(d−1)))/d`,

where `S = ∑ⱼ vⱼ⁴`, uniformly in `a` and `x`.  This is §58's leg plus §57's leg, added by
`abs_cdfGapInt_le_add_of_common`; the Gaussian `N(0, 1/d)` that both legs run through does not
appear.  It is the bound §56's `abs_hadRate_sub_rhoSq_le_of_cdfGapInt` asks for.

It is **not** a Kolmogorov-Smirnov bound: `|cdf μ − cdf ν|` is not bounded here, and no `η` for
the sup-norm route of §55 or §56 is produced by it.  The dominant term is `2·S^{1/4}/√d`, which at
a spiky direction is only `Θ(d^{-1/2})`; see the section header for why leg 1 does not reach
`Θ(1/d)` in general and what the spread hypothesis buys. -/
theorem abs_cdfGapInt_coordLaw_sphericalCoordLaw_le (hd : 2 ≤ d) {H : Od d}
    (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (sphericalCoordLaw d) a x|
      ≤ (15 * (∑ j, v j ^ 4) + 2 * Real.sqrt (Real.sqrt (∑ j, v j ^ 4))) / Real.sqrt d
          + Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) :=
  Threshold.abs_cdfGapInt_le_add_of_common (abs_cdfGapInt_coordLaw_gaussian_le hd hH hv i a x)
    (abs_cdfGapInt_sphericalCoordLaw_gaussian_le hd a x)

/-- **The two legs, composed at a spread direction.**  Same statement with §58's spread instance
in place of its general one, so that the second term of leg 1 carries `d^{-1/2}` rather than
`S^{1/4}`:

  `|cdfGapInt (coordLaw H v i) (f_d) a x| ≤ (15·S + 2/√d)/√d + √(1 + 1/(4(d−1)))/d`.

Still uniform in `a` and `x`, and still not a Kolmogorov-Smirnov bound. -/
theorem abs_cdfGapInt_coordLaw_sphericalCoordLaw_le_of_spread (hd : 2 ≤ d) {H : Od d}
    (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d)
    (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (sphericalCoordLaw d) a x|
      ≤ (15 * (∑ j, v j ^ 4) + 2 * (1 / Real.sqrt d)) / Real.sqrt d
          + Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) :=
  Threshold.abs_cdfGapInt_le_add_of_common
    (abs_cdfGapInt_coordLaw_gaussian_le_of_spread hd hH hv i hspread a x)
    (abs_cdfGapInt_sphericalCoordLaw_gaussian_le hd a x)

/-- **The composed leg at a spread direction, as one fraction:**

  `|cdfGapInt (coordLaw H v i) (f_d) a x| ≤ (15/√d + 7/2)/d`,

uniformly in `a` and `x`.  This is `Θ(1/d)` in the dimension, the same order as §57's coupling
leg, so at a spread direction neither leg dominates the other by a power of `d`.  It is the
previous statement with `S ≤ 1/d` (`sum_pow_four_le_inv_of_spread`) and
`√(1 + 1/(4(d−1))) ≤ 3/2` (`sqrt_one_add_quarter_le`) substituted, so the `7/2` is loose by the
amount recorded at the latter; nothing new is proved. -/
theorem abs_cdfGapInt_coordLaw_sphericalCoordLaw_le_spread (hd : 2 ≤ d) {H : Od d}
    (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d)
    (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (sphericalCoordLaw d) a x|
      ≤ (15 / Real.sqrt d + 7 / 2) / (d : ℝ) := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hS := sum_pow_four_le_inv_of_spread hd hv hspread
  have hT := sqrt_one_add_quarter_le (d := d) hd
  refine (abs_cdfGapInt_coordLaw_sphericalCoordLaw_le_of_spread hd hH hv i hspread a x).trans ?_
  have step1 : (15 * (∑ j, v j ^ 4) + 2 * (1 / Real.sqrt d)) / Real.sqrt d
      ≤ (15 * (1 / (d:ℝ)) + 2 * (1 / Real.sqrt d)) / Real.sqrt d := by gcongr
  have step2 : Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) ≤ (3/2 : ℝ) / (d:ℝ) := by gcongr
  have heq : (15 * (1 / (d:ℝ)) + 2 * (1 / Real.sqrt d)) / Real.sqrt d + (3/2 : ℝ) / (d:ℝ)
      = (15 / Real.sqrt d + 7 / 2) / (d : ℝ) := by
    have hs2 : Real.sqrt (d:ℝ) * Real.sqrt (d:ℝ) = (d:ℝ) := Real.mul_self_sqrt hd0.le
    set s := Real.sqrt (d:ℝ) with hs
    rw [← hs2]
    field_simp
    ring
  linarith

/-! ### The composed leg with the Dirichlet constant evaluated

§58a's `dirConst_eq` gives the kernel constant its exact value `π/2`, improving §58d's leg-1
constants from `15` and `2` to `21/2` and `13/10`.  §57's leg and the `3/2` for
`√(1 + 1/(4(d−1)))` are unaffected, so only part of the composed constant improves:
`7/2 = 2 + 3/2` becomes
`14/5 = 13/10 + 3/2`, not `2·(2/π)·... + 3/2` scaled throughout.  The unprimed statements are
retained unchanged. -/

/-- **The two legs composed, with the Dirichlet constant evaluated:**

  `|cdfGapInt (coordLaw H v i) (f_d) a x| ≤ ((21/2)S + (13/10)S^{1/4})/√d + √(1 + 1/(4(d−1)))/d`.

The sharpened form of `abs_cdfGapInt_coordLaw_sphericalCoordLaw_le`.  It is still not a
Kolmogorov-Smirnov bound, and the dominant term is still `Θ(d^{-1/2})` at a spiky direction. -/
theorem abs_cdfGapInt_coordLaw_sphericalCoordLaw_le' (hd : 2 ≤ d) {H : Od d}
    (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (sphericalCoordLaw d) a x|
      ≤ (21 / 2 * (∑ j, v j ^ 4) + 13 / 10 * Real.sqrt (Real.sqrt (∑ j, v j ^ 4)))
            / Real.sqrt d
          + Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) :=
  Threshold.abs_cdfGapInt_le_add_of_common (abs_cdfGapInt_coordLaw_gaussian_le' hd hH hv i a x)
    (abs_cdfGapInt_sphericalCoordLaw_gaussian_le hd a x)

/-- **The two legs composed at a spread direction, with the Dirichlet constant evaluated:**

  `|cdfGapInt (coordLaw H v i) (f_d) a x| ≤ ((21/2)S + (13/10)/√d)/√d + √(1 + 1/(4(d−1)))/d`.

The sharpened form of `abs_cdfGapInt_coordLaw_sphericalCoordLaw_le_of_spread`. -/
theorem abs_cdfGapInt_coordLaw_sphericalCoordLaw_le_of_spread' (hd : 2 ≤ d) {H : Od d}
    (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d)
    (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (sphericalCoordLaw d) a x|
      ≤ (21 / 2 * (∑ j, v j ^ 4) + 13 / 10 * (1 / Real.sqrt d)) / Real.sqrt d
          + Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) :=
  Threshold.abs_cdfGapInt_le_add_of_common
    (abs_cdfGapInt_coordLaw_gaussian_le_of_spread' hd hH hv i hspread a x)
    (abs_cdfGapInt_sphericalCoordLaw_gaussian_le hd a x)

/-- **The composed leg at a spread direction, as one fraction, with the Dirichlet constant
evaluated:**

  `|cdfGapInt (coordLaw H v i) (f_d) a x| ≤ ((21/2)/√d + 14/5)/d`,

uniformly in `a` and `x`, improving `abs_cdfGapInt_coordLaw_sphericalCoordLaw_le_spread`'s
`(15/√d + 7/2)/d`.  Still `Θ(1/d)`; the order does not change, only the constant.  The `14/5`
is `13/10 + 3/2`, and the `3/2` half of it is §59's crude bound on `√(1 + 1/(4(d−1)))`, whose
true value is below `1.001` at `d = 512`; that slack is untouched here and is the larger of the
two remaining losses in this constant. -/
theorem abs_cdfGapInt_coordLaw_sphericalCoordLaw_le_spread' (hd : 2 ≤ d) {H : Od d}
    (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d)
    (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (sphericalCoordLaw d) a x|
      ≤ (21 / 2 / Real.sqrt d + 14 / 5) / (d : ℝ) := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hS := sum_pow_four_le_inv_of_spread hd hv hspread
  have hT := sqrt_one_add_quarter_le (d := d) hd
  refine (abs_cdfGapInt_coordLaw_sphericalCoordLaw_le_of_spread' hd hH hv i hspread a x).trans ?_
  have step1 : (21 / 2 * (∑ j, v j ^ 4) + 13 / 10 * (1 / Real.sqrt d)) / Real.sqrt d
      ≤ (21 / 2 * (1 / (d:ℝ)) + 13 / 10 * (1 / Real.sqrt d)) / Real.sqrt d := by gcongr
  have step2 : Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) ≤ (3/2 : ℝ) / (d:ℝ) := by gcongr
  have heq : (21 / 2 * (1 / (d:ℝ)) + 13 / 10 * (1 / Real.sqrt d)) / Real.sqrt d
        + (3/2 : ℝ) / (d:ℝ)
      = (21 / 2 / Real.sqrt d + 14 / 5) / (d : ℝ) := by
    have hs2 : Real.sqrt (d:ℝ) * Real.sqrt (d:ℝ) = (d:ℝ) := Real.mul_self_sqrt hd0.le
    set s := Real.sqrt (d:ℝ) with hs
    rw [← hs2]
    field_simp
    ring
  linarith

/-! ### The deployed per-pass rate, with no Kolmogorov-Smirnov hypothesis -/

namespace Threshold

open Hadamard Orthogonal CoordLaw

variable {K : ℕ}

/-- **The deployed per-pass rate against `ρ²`, unconditionally.**  For `d ≥ 2`, a flat `H`, a unit
direction `v`, an ordered codebook `cbk` whose thresholds all lie in a window `[a, b]` that
strictly contains the common support, and any measurable nearest-codeword map `q` of `cbk`,

  `|hadRate H (blockStage q) v − ρ²| ≤ 2·((15S + 2S^{1/4})/√d + √(1 + 1/(4(d−1)))/d)·spread cbk·d`

with `S = ∑ⱼ vⱼ⁴`.  This is `abs_hadRate_sub_rhoSq_le_of_cdfGapInt` with its hypothesis `hC`
discharged by `abs_cdfGapInt_coordLaw_sphericalCoordLaw_le`.  Every quantity on the right is
explicit: there is no `η`, no `C`, and no `(b − a)`.  `a` and `b` survive only in the hypotheses,
where they place the codebook's thresholds in a window around the support.

**The rate, stated honestly.**  The right-hand side equals
`2·spread cbk·(√d·(15S + 2S^{1/4}) + √(1 + 1/(4(d−1))))`.  The factor `d` that §11c's identity
carries is cancelled by one of the two `1/d`'s, but not by both, because §58's leg is
`Θ((S + S^{1/4})/√d)` rather than `Θ(1/d)`; see §58's own account of why.  So at a spiky
direction, `S` of order `1`, this bound is `Θ(√d·spread cbk)` and does **not** tend to zero for a
codebook at the coordinate scale.  For the case where it does, use
`abs_hadRate_sub_rhoSq_le_of_spread` below.  Nothing here says the deployed randomised Hadamard
rotation attains `ρ²`; it bounds the gap, and the bound is not zero. -/
theorem abs_hadRate_sub_rhoSq_le_unconditional (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ 2 * ((15 * (∑ j, v j ^ 4) + 2 * Real.sqrt (Real.sqrt (∑ j, v j ^ 4))) / Real.sqrt d
              + Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ)) * spread cbk * (d : ℝ) :=
  abs_hadRate_sub_rhoSq_le_of_cdfGapInt hd hH v hv i cbk hq hqm ha hb hthr
    (fun x => abs_cdfGapInt_coordLaw_sphericalCoordLaw_le hd hH hv i a x)

/-- **The deployed per-pass rate against `ρ²` at a spread direction, unconditionally:**

  `|hadRate H (blockStage q) v − ρ²| ≤ (30/√d + 7)·spread cbk`,

for `d ≥ 2`, a flat `H`, a unit direction with `|vⱼ| ≤ d^{-1/2}` for every `j`, and the same
codebook and window hypotheses as above.  No `η`, no `C`, no `(b − a)`, and the explicit factor
`d` of §11c's identity has cancelled completely against the `Θ(1/d)` of the composed leg.

**Does it tend to zero?**  Only as fast as `spread cbk` does, and `spread cbk` is left abstract
here.  The dimension-dependent factor `30/√d + 7` decreases to `7`, so the bound is
`Θ(spread cbk)` and no better: a codebook of fixed codeword range gives a bound tending to
`7·spread cbk`, not to zero.  If the codebook is placed at the coordinate scale, so that
`spread cbk = cbk.c (Fin.last K) − cbk.c 0` is `Θ(d^{-1/2})`, then the right-hand side is
`Θ(d^{-1/2})` and does tend to zero.  The constant decides where that begins to say anything: the
factor `30/√d + 7` is `8.33` at `d = 512` and never falls below `7`, so with the shipped `2`-bit
codebook, whose range is `3.02` coordinate units, the bound reads `1.11` at `d = 512` against the
free bound `|hadRate − ρ²| ≤ 1` that holds whenever `0` is a codeword, and it first falls below
`ρ² = 0.1171` near `d ≈ 3.4·10⁴`.  At `d = 512` this is not a numerically useful estimate.
That scaling is a hypothesis about `cbk` which this theorem
does not make and does not prove; the statement is uniform in `cbk`, and the vanishing is exactly
as conditional as the scaling is.

No analytic hypothesis is left undischarged on this route: the only inputs are `d ≥ 2`, flatness
of `H`, the unit and spread conditions on `v`, and the codebook window.  It is still an
inequality, not an identity: the deployed rotation is not shown to attain the Haar rate, and the
constants `30` and `7` are not claimed sharp (see the section header for where the slack is). -/
theorem abs_hadRate_sub_rhoSq_le_of_spread (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d)
    (cbk : Codebook K) {q : ℝ → ℝ} (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q)
    {a b : ℝ} (ha : a < -‖v‖) (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ (30 / Real.sqrt d + 7) * spread cbk := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  refine (abs_hadRate_sub_rhoSq_le_of_cdfGapInt hd hH v hv i cbk hq hqm ha hb hthr
    (fun x => abs_cdfGapInt_coordLaw_sphericalCoordLaw_le_spread hd hH hv i hspread a x)).trans_eq
    ?_
  field_simp
  ring

/-- The same at a spread direction for the explicit step quantiser of the codebook, which §52
shows is a nearest-codeword map (`isNearest_stepQuant`) and is measurable
(`measurable_stepQuant`).  This is a check that the hypotheses of
`abs_hadRate_sub_rhoSq_le_of_spread` are met by something concrete; it proves nothing new, and the
caveats on `spread cbk` recorded there apply verbatim. -/
theorem abs_hadRate_stepQuant_sub_rhoSq_le_of_spread (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d)
    (cbk : Codebook K) {a b : ℝ} (ha : a < -‖v‖) (hb : ‖v‖ ≤ b)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) (stepQuant cbk)) v
        - rhoSq d (stepQuant cbk) (sphericalCoordLaw d)|
      ≤ (30 / Real.sqrt d + 7) * spread cbk :=
  abs_hadRate_sub_rhoSq_le_of_spread hd hH v hv i hspread cbk (isNearest_stepQuant cbk)
    (measurable_stepQuant cbk) ha hb hthr

/-! ### The deployed rate with the Dirichlet constant evaluated -/

/-- **The deployed per-pass rate against `ρ²`, unconditionally, with the Dirichlet constant
evaluated.**  Same hypotheses as `abs_hadRate_sub_rhoSq_le_unconditional`, with leg 1's constants
improved from `15` and `2` to `21/2` and `13/10` by §58a's `dirConst_eq`.  The caveats there apply
verbatim: at a spiky direction this is still `Θ(√d·spread cbk)` and does not tend to zero for a
codebook at the coordinate scale. -/
theorem abs_hadRate_sub_rhoSq_le_unconditional' (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ 2 * ((21 / 2 * (∑ j, v j ^ 4)
                + 13 / 10 * Real.sqrt (Real.sqrt (∑ j, v j ^ 4))) / Real.sqrt d
              + Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ)) * spread cbk * (d : ℝ) :=
  abs_hadRate_sub_rhoSq_le_of_cdfGapInt hd hH v hv i cbk hq hqm ha hb hthr
    (fun x => abs_cdfGapInt_coordLaw_sphericalCoordLaw_le' hd hH hv i a x)

/-- **The deployed per-pass rate against `ρ²` at a spread direction, with the Dirichlet constant
evaluated:**

  `|hadRate H (blockStage q) v − ρ²| ≤ (21/√d + 28/5)·spread cbk`,

under exactly the hypotheses of `abs_hadRate_sub_rhoSq_le_of_spread`, whose `(30/√d + 7)` this
improves.  The gain comes entirely from `dirConst_eq`: `30` was `2·(8/9)·(97/6)` rounded up and is
now `2·(2/π)·(97/6) ≈ 20.58` rounded up to `21`, while `7 = 2·2 + 3` becomes
`28/5 = 2·(13/10) + 3`.  The `3` does **not** improve, since it is §57's coupling leg through
§59's `3/2`, which `dirConst` does not enter.

**What this buys, numerically, and what it does not.**  The factor is `6.53` at `d = 512` against
the unprimed `8.33`, a reduction of `21.6%`, and it never falls below `28/5 = 5.6`.  With the
shipped `2`-bit codebook, whose codeword range is `3.02` coordinate units so that
`spread cbk = 3.02/√d`, the bound reads `0.872` at `d = 512`, which is the first time this route
returns something smaller than the bound `|hadRate − ρ²| ≤ 1` that holds for free whenever `0` is
a codeword.  It is still far above `ρ² = 0.1171`: this is a bound of the same order as the
quantity it bounds, not a small relative error, and it first falls below `ρ²` near `d ≈ 2.2·10⁴`
against `d ≈ 3.4·10⁴` unprimed.
`spread cbk` is left abstract by the statement, so the vanishing remains exactly as conditional on
the codebook's scaling as it was before, and the theorem proves no such scaling.

The remaining slack is itemised at `dirConst_eq` and is not addressed here: §59's `3/2` for
`√(1 + 1/(4(d−1)))`, whose true value at `d = 512` is `1.00024`, alone accounts for `0.40` of the
`0.872`.  Neither `21` nor `28/5` is claimed sharp. -/
theorem abs_hadRate_sub_rhoSq_le_of_spread' (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d)
    (cbk : Codebook K) {q : ℝ → ℝ} (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q)
    {a b : ℝ} (ha : a < -‖v‖) (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ (21 / Real.sqrt d + 28 / 5) * spread cbk := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  refine (abs_hadRate_sub_rhoSq_le_of_cdfGapInt hd hH v hv i cbk hq hqm ha hb hthr
    (fun x =>
      abs_cdfGapInt_coordLaw_sphericalCoordLaw_le_spread' hd hH hv i hspread a x)).trans_eq ?_
  field_simp
  ring

/-- The same for the explicit step quantiser, as `abs_hadRate_stepQuant_sub_rhoSq_le_of_spread` is
for the unprimed bound.  It proves nothing new, and the caveats on `spread cbk` recorded at
`abs_hadRate_sub_rhoSq_le_of_spread'` apply verbatim. -/
theorem abs_hadRate_stepQuant_sub_rhoSq_le_of_spread' (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d)
    (cbk : Codebook K) {a b : ℝ} (ha : a < -‖v‖) (hb : ‖v‖ ≤ b)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) (stepQuant cbk)) v
        - rhoSq d (stepQuant cbk) (sphericalCoordLaw d)|
      ≤ (21 / Real.sqrt d + 28 / 5) * spread cbk :=
  abs_hadRate_sub_rhoSq_le_of_spread' hd hH v hv i hspread cbk (isNearest_stepQuant cbk)
    (measurable_stepQuant cbk) ha hb hthr

end Threshold

end ComposeLegs
