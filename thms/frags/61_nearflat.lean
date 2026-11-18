
/-! ## §60  Near-flat directions: the vanishing bound on a neighbourhood, not on `2^d` points

§59's `abs_hadRate_sub_rhoSq_le_of_spread` is the only statement here whose bound on
`|hadRate − ρ²|` carries no growing power of `d`, and its hypothesis is `|vⱼ| ≤ d^{-1/2}` for
every `j`.  At a **unit** direction that hypothesis is an equality constraint: `d` numbers whose
squares are each at most `1/d` and sum to `1` are each exactly `1/d`.  So that theorem speaks
about the `2^d` sign patterns of the flat vector and about no other direction at all.

This section relaxes the hypothesis to `|vⱼ| ≤ c·d^{-1/2}` for a real `c`, and carries `c`
through to the conclusion.  `one_le_of_nearFlat` shows `‖v‖ = 1` forces `c ≥ 1`, so the
hypothesis is never vacuous below `c = 1` and `c = 1` is exactly §59's flat case; for `c > 1` it
describes an open set of directions, those flat to within a factor `c` in sup norm, rather than a
finite set of points.

**The bound, as a function of `c`.**  With `S = ∑ⱼ vⱼ⁴` and `A`, `B` the leg-1 constants,
`abs_hadRate_sub_rhoSq_le_of_nearFlat_gen` reads

  `|hadRate H (blockStage q) v − ρ²| ≤ 2·(A·c²/√d + B·c + 1 + 1/(4(d−1)))·spread cbk`,

and `abs_hadRate_sub_rhoSq_le_of_nearFlat` instantiates it at `A = 3`, `B = 13/10`, which is
§58's two-range estimate with the sharp damping constant and with `dirConst` evaluated.  Two
things are visible in that shape and are the point of writing `c` explicitly.

* The `c`-dependence is **linear in the dominant term**, not quadratic: `B·c` comes from the
  range `|u| > √d/M` on which nothing is known about the atomic law's characteristic function,
  and `A·c²/√d` from the damped range, which is smaller by `√d`.  So the price of leaving the
  flat point is `13/10` per unit of `c`, against a total budget of `1/spread cbk`.
* Nothing decays.  As `d → ∞` the factor tends to `2 + 13c/5`, so the bound is `Θ(spread cbk)`
  exactly as §59's is, and vanishing is still conditional on the codebook being placed at the
  coordinate scale.  This section improves the constant and widens the hypothesis; it does not
  change the rate in `d`.

**What this does not do, and it is the main caveat.**  `c` bounds the **largest** coordinate.
A Haar-random unit direction in dimension `d` has `maxⱼ|vⱼ|·√d` concentrated near `√(2 log d)`,
which is about `3.5` at `d = 512`, so the near-flat class at any `c` of order `1` is **not** of
full measure on the sphere and this section does **not** discharge §36's
`integral_norm_resid_sq_le_had`, which integrates over all unit directions.  The set has been
enlarged from `2^d` points to a sup-norm neighbourhood of them, and that is all.

Nor does the fourth-moment statistic control `c`: `S ≥ maxⱼ vⱼ⁴` gives only
`c ≤ 3^{1/4}·d^{1/4}` from `d·S = 3`, which grows with `d`.  For that reason
`abs_hadRate_sub_rhoSq_le_of_nearFlat_spikiness` is stated with `S` free beside `c`, so a
measured spikiness can be substituted where it is available, but the `B·c` term survives it: the
sup-norm hypothesis is not removable by any argument in this file.

**Numerically, at the dimension the paper ships.**  At `d = 512` the factor
`2·(3c²/√d + 13c/10 + 1 + 1/(4(d−1)))` is `4.867` at `c = 1`, `6.499` at `c = 3/2` and `7.301` at
`c = √3`, against `8.326` for §59's `(30/√d + 7)`.  `abs_hadRate_sub_rhoSq_lt_one_of_nearFlat_512`
records the consequence in checkable form: for `c ≤ 7/4` and any codebook of spread at most
`3.0208/√512`, the bound is **below `1`**, which is the free bound that holds whenever `0` is a
codeword.  At `c = 1` it is `0.650`.  That is a statement about codebooks of that spread, given
as a hypothesis; no codebook is defined or analysed here, and nothing is claimed about any
codebook the paper ships.  It is still far from `ρ² = 0.1171`: at `c = 1` and a codebook of
spread `3.0208/√d` the bound first falls below `ρ²` near `d ≈ 1.8·10⁴`, against `d ≈ 3.4·10⁴`
for §59's constant. -/

noncomputable section NearFlatLeg

open Hadamard Orthogonal CoordLaw

variable {d : ℕ}

namespace NearFlat

/-! ### What `|vⱼ| ≤ c/√d` gives, and what it forces -/

/-- **A near-flat unit direction has spikiness at most `c²/d`.**  If `|vⱼ| ≤ c·d^{-1/2}` for
every `j` and `‖v‖ = 1`, then `∑ⱼ vⱼ⁴ ≤ c²/d`.  At `c = 1` this is §59's
`sum_pow_four_le_inv_of_spread`, where the inequality is forced to be an equality; for `c > 1` it
is a genuine inequality and is usually far from tight, since the hypothesis constrains only the
largest coordinate.  Only the inequality is proved, which is all the estimate consumes. -/
theorem sum_pow_four_le_sq_div_of_nearFlat (hd : 2 ≤ d) {v : Ed d} (hv : ‖v‖ = 1) {c : ℝ}
    (hnf : ∀ j, |v j| ≤ c / Real.sqrt d) : (∑ j, v j ^ 4) ≤ c ^ 2 / (d : ℝ) := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsum : ∑ j, v j ^ 2 = 1 := by
    have h := norm_sq_eq_sum_sq v
    rw [hv] at h
    simpa using h.symm
  have hcoord : ∀ j, v j ^ 2 ≤ c ^ 2 / (d:ℝ) := by
    intro j
    have h := hnf j
    have hnn := abs_nonneg (v j)
    have hsq : |v j| ^ 2 ≤ (c / Real.sqrt d) ^ 2 := by nlinarith
    calc v j ^ 2 = |v j| ^ 2 := (sq_abs _).symm
      _ ≤ (c / Real.sqrt d) ^ 2 := hsq
      _ = c ^ 2 / (d:ℝ) := by rw [div_pow, Real.sq_sqrt hd0.le]
  calc ∑ j, v j ^ 4 = ∑ j, v j ^ 2 * v j ^ 2 := Finset.sum_congr rfl fun j _ => by ring
    _ ≤ ∑ j, (c ^ 2 / (d:ℝ)) * v j ^ 2 :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_right (hcoord j) (sq_nonneg _)
    _ = (c ^ 2 / (d:ℝ)) * ∑ j, v j ^ 2 := by rw [← Finset.mul_sum]
    _ = c ^ 2 / (d:ℝ) := by rw [hsum, mul_one]

/-- **`c ≥ 1` is forced.**  A unit direction satisfying `|vⱼ| ≤ c·d^{-1/2}` for every `j` has
`c ≥ 1`, since `1 = ∑ⱼ vⱼ² ≤ d·(c²/d) = c²`.  So the near-flat hypothesis is never satisfiable
below `c = 1`, `c = 1` reproduces §59's flat case exactly, and the `c` in every bound below is at
least `1`.  This is recorded because it fixes the scale against which the growth in `c` should be
read; it is used nowhere. -/
theorem one_le_of_nearFlat (hd : 2 ≤ d) {v : Ed d} (hv : ‖v‖ = 1) {c : ℝ}
    (hnf : ∀ j, |v j| ≤ c / Real.sqrt d) : 1 ≤ c := by
  have hdpos : (0:ℕ) < d := by omega
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast hdpos
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hsum : ∑ j, v j ^ 2 = 1 := by
    have h := norm_sq_eq_sum_sq v
    rw [hv] at h
    simpa using h.symm
  have hcoord : ∀ j, v j ^ 2 ≤ c ^ 2 / (d:ℝ) := by
    intro j
    have h := hnf j
    have hnn := abs_nonneg (v j)
    have hsq : |v j| ^ 2 ≤ (c / Real.sqrt d) ^ 2 := by nlinarith
    calc v j ^ 2 = |v j| ^ 2 := (sq_abs _).symm
      _ ≤ (c / Real.sqrt d) ^ 2 := hsq
      _ = c ^ 2 / (d:ℝ) := by rw [div_pow, Real.sq_sqrt hd0.le]
  have hc0 : 0 ≤ c := by
    have h := le_trans (abs_nonneg (v ⟨0, hdpos⟩)) (hnf ⟨0, hdpos⟩)
    have h2 : 0 ≤ c / Real.sqrt d * Real.sqrt d := mul_nonneg h hsd.le
    rwa [div_mul_cancel₀ _ (ne_of_gt hsd)] at h2
  have hsq : (1:ℝ) ≤ c ^ 2 := by
    calc (1:ℝ) = ∑ j, v j ^ 2 := hsum.symm
      _ ≤ ∑ _j : Fin d, c ^ 2 / (d:ℝ) := Finset.sum_le_sum fun j _ => hcoord j
      _ = (d:ℝ) * (c ^ 2 / (d:ℝ)) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      _ = c ^ 2 := by field_simp
  nlinarith

/-- `√(1 + x) ≤ 1 + x` for `x ≥ 0`, applied to §57's coupling constant.  §59's
`sqrt_one_add_quarter_le` bounds the same quantity by `3/2`, which is loose by a factor `1.4998`
at `d = 512`; this bound is loose by a factor below `1.00025` there, and the whole difference
between the two is what removes `1/2` from the composed constant.  The exact value is
`√(1 + 1/(4(d−1)))` and neither bound is it. -/
theorem sqrt_one_add_quarter_le_self (hd : 2 ≤ d) :
    Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) ≤ 1 + 1 / (4 * ((d : ℝ) - 1)) := by
  have hd1 : (1:ℝ) ≤ (d:ℝ) - 1 := by
    have : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
    linarith
  have hq : (0:ℝ) ≤ 1 / (4 * ((d:ℝ) - 1)) := by positivity
  have hle : 1 + 1 / (4 * ((d:ℝ) - 1)) ≤ (1 + 1 / (4 * ((d:ℝ) - 1))) ^ 2 := by nlinarith
  have := Real.sqrt_le_sqrt hle
  rwa [Real.sqrt_sq (by linarith)] at this

/-! ### Leg 1 with both available improvements applied at once

§58c′ sharpens the damping constant (`83/18` in place of `97/6` inside the integral) and §58d
evaluates `dirConst` (`2/π` in place of `8/9` outside it), but each is stated against the other's
unimproved input.  Composing them gives `(2/π)·(83/18) = 2.9360` and `(2/π)·2 = 1.2732`, rounded
up to `3` and `13/10`, a further `2.2%` and `2.1%` of slack.  Nothing new is proved here: this is
`abs_cdfGapInt_le_of_charFun` fed with `integral_gapWeight_le_sharp` and `dirConst_eq`. -/

/-- **Leg 1 with an explicit sup-norm bound, sharp damping and the Dirichlet constant.**  For
`d ≥ 2`, a flat `H`, a unit direction `v` with `|vⱼ| ≤ M` and `0 < M ≤ 1`, any coordinate `i` and
every pair `a`, `x`,

  `|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x| ≤ (3·∑ⱼvⱼ⁴ + (13/10)·M)/√d`,

against `15·∑ⱼvⱼ⁴ + 2M` in `abs_cdfGapInt_coordLaw_gaussian_le_of_bound`.

**What has not changed.**  It is not a Kolmogorov-Smirnov bound, it is one leg only, the second
law is the Gaussian and not `sphericalCoordLaw d`, and the `M/√d` term still comes from the range
`|u| > √d/M` on which this development knows nothing about the atomic law's characteristic
function and uses `‖φ_μ − φ_ν‖ ≤ 2`.  That term dominates at a near-flat direction and is the
reason the rate in `d` is unchanged. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_of_bound_best (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M)
    (hM1 : M ≤ 1) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (3 * (∑ j, v j ^ 4) + 13 / 10 * M) / Real.sqrt d := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hS : 0 ≤ ∑ j, v j ^ 4 := FourierGap.sum_pow_four_nonneg v
  have hpi : (3.141592:ℝ) < π := Real.pi_gt_d6
  have hpi0 : (0:ℝ) < π := Real.pi_pos
  have hcf : IntegrableOn (fun u : ℝ => ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2) (Set.Ioi 0) :=
    FourierGap.integrableOn_gapWeight hH hv i hM hM1 hd0
  have hI : (∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2)
      ≤ (83 / 18 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d :=
    FourierGap.integral_gapWeight_le_sharp hH hv i hM hM0 hM1 hd0
  refine (FourierGap.abs_cdfGapInt_le_of_charFun (coordLaw H v i)
    (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ))))
    (FourierGap.integrable_abs_sub_coordLaw H v i)
    (FourierGap.integrable_abs_sub_gaussianReal _) hcf a x).trans ?_
  rw [FourierGap.dirConst_eq]
  set X := ∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2 with hX
  have hXnn : 0 ≤ X := by
    rw [hX]; exact setIntegral_nonneg measurableSet_Ioi fun u _ => by positivity
  have hdiv : X / (π / 2) = 2 / π * X := by field_simp
  rw [hdiv]
  have hcoef1 : 2 / π * (83 / 18) ≤ 3 := by
    rw [div_mul_eq_mul_div, div_le_iff₀ hpi0]; nlinarith [hpi]
  have hcoef2 : 2 / π * 2 ≤ 13 / 10 := by
    rw [div_mul_eq_mul_div, div_le_iff₀ hpi0]; nlinarith [hpi]
  calc 2 / π * X ≤ 2 / π * ((83 / 18 * (∑ j, v j ^ 4) + 2 * M) / Real.sqrt d) := by
        gcongr
    _ = (2 / π * (83 / 18 * (∑ j, v j ^ 4) + 2 * M)) / Real.sqrt d := by ring
    _ ≤ (3 * (∑ j, v j ^ 4) + 13 / 10 * M) / Real.sqrt d := by
        gcongr
        have hsplit : 2 / π * (83 / 18 * (∑ j, v j ^ 4) + 2 * M)
            = (2 / π * (83 / 18)) * (∑ j, v j ^ 4) + (2 / π * 2) * M := by ring
        rw [hsplit]
        nlinarith [hS, hM0.le, hcoef1, hcoef2]

/-! ### Composing the two legs at a near-flat direction, with the leg-1 constants abstract

The composition is stated once, with the leg-1 bound as a hypothesis in the form
`(A·S + B·M)/√d` at `M = c/√d`, so that no constant is baked into it and an improvement to
either leg-1 constant can be substituted without reproving anything below. -/

/-- **The composed leg at a near-flat direction, constants abstract.**  Given any leg-1 bound of
the shape `(A·∑ⱼvⱼ⁴ + B·(c/√d))/√d` against the Gaussian `N(0, 1/d)`, uniform in `x`, the
triangle inequality against §57's coupling leg gives

  `|cdfGapInt (coordLaw H v i) (f_d) a x| ≤ (A·S·√d + B·c + 1 + 1/(4(d−1)))/d`

with `S = ∑ⱼvⱼ⁴`, uniformly in `x`.  The Gaussian that both legs run through does not appear,
and the near-flat hypothesis is not used: this step is algebra plus
`sqrt_one_add_quarter_le_self`.  It is not a Kolmogorov-Smirnov bound and produces no `η`. -/
theorem abs_cdfGapInt_le_of_legBound (hd : 2 ≤ d) {H : Od d} {v : Ed d} (i : Fin d)
    {c A B a : ℝ}
    (hleg : ∀ x, |Threshold.cdfGapInt (coordLaw H v i)
        (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (A * (∑ j, v j ^ 4) + B * (c / Real.sqrt d)) / Real.sqrt d) (x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (sphericalCoordLaw d) a x|
      ≤ (A * (∑ j, v j ^ 4) * Real.sqrt d + B * c + 1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hcomp := Threshold.abs_cdfGapInt_le_add_of_common (hleg x)
    (abs_cdfGapInt_sphericalCoordLaw_gaussian_le hd a x)
  have step2 : Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ)
      ≤ (1 + 1 / (4 * ((d:ℝ) - 1))) / (d:ℝ) := by
    gcongr
    exact sqrt_one_add_quarter_le_self (d := d) hd
  have heq : (A * (∑ j, v j ^ 4) + B * (c / Real.sqrt d)) / Real.sqrt d
        + (1 + 1 / (4 * ((d:ℝ) - 1))) / (d:ℝ)
      = (A * (∑ j, v j ^ 4) * Real.sqrt d + B * c + 1 + 1 / (4 * ((d : ℝ) - 1)))
          / (d : ℝ) := by
    have hs2 : Real.sqrt (d:ℝ) * Real.sqrt (d:ℝ) = (d:ℝ) := Real.mul_self_sqrt hd0.le
    set s := Real.sqrt (d:ℝ) with hs
    rw [← hs2]
    have hs0 : s ≠ 0 := ne_of_gt hsd
    have hs1 : s * s - 1 ≠ 0 := by
      rw [hs2]
      have : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
      intro h; linarith
    field_simp
    ring
  linarith

/-- `22.6 ≤ √512`, for the numerical corollary at the dimension the paper ships.  The true value
is `22.6274…`. -/
theorem sqrt_512_ge : (22.6:ℝ) ≤ Real.sqrt 512 := by
  have h : (22.6:ℝ) = Real.sqrt (22.6 ^ 2) := (Real.sqrt_sq (by norm_num)).symm
  rw [h]
  exact Real.sqrt_le_sqrt (by norm_num)

end NearFlat

/-! ### The deployed per-pass rate at a near-flat direction -/

namespace Threshold

open Hadamard Orthogonal CoordLaw NearFlat

variable {K : ℕ}

/-- **The deployed rate from any leg-1 bound, in the spikiness and in `c`.**  For `d ≥ 2`, a flat
`H`, a unit direction `v`, an ordered codebook whose thresholds lie in a window `[a, b]` around
the support, and any measurable nearest-codeword map,

  `|hadRate H (blockStage q) v − ρ²| ≤ 2·(A·S·√d + B·c + 1 + 1/(4(d−1)))·spread cbk`

whenever leg 1 holds in the form `(A·S + B·(c/√d))/√d`, with `S = ∑ⱼvⱼ⁴`.  This is §56's
`abs_hadRate_sub_rhoSq_le_of_cdfGapInt` with `hC` discharged by `abs_cdfGapInt_le_of_legBound`;
the explicit factor `d` of §11c's identity cancels the `1/d` of the composed leg exactly.

The near-flat hypothesis is **not** used here; it enters only through the shape of `hleg`.  No
`η`, no `C` and no `(b − a)` appear in the conclusion.  Nothing is claimed to be sharp, and the
deployed rotation is not shown to attain `ρ²`. -/
theorem abs_hadRate_sub_rhoSq_le_of_legBound (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) {c A B : ℝ} (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hleg : ∀ x, |cdfGapInt (coordLaw H v i)
        (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (A * (∑ j, v j ^ 4) + B * (c / Real.sqrt d)) / Real.sqrt d) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ 2 * (A * (∑ j, v j ^ 4) * Real.sqrt d + B * c + 1 + 1 / (4 * ((d:ℝ) - 1)))
          * spread cbk := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  refine (abs_hadRate_sub_rhoSq_le_of_cdfGapInt hd hH v hv i cbk hq hqm ha hb hthr
    (fun x => abs_cdfGapInt_le_of_legBound hd i hleg x)).trans_eq ?_
  field_simp

/-- **The deployed rate at a near-flat direction, in the spikiness.**  For `d ≥ 2`, a flat `H`, a
unit direction with `|vⱼ| ≤ c·d^{-1/2}` for every `j` and `0 < c ≤ √d`, and the codebook window
hypotheses of §56,

  `|hadRate H (blockStage q) v − ρ²| ≤ 2·(3·S·√d + (13/10)·c + 1 + 1/(4(d−1)))·spread cbk`

with `S = ∑ⱼvⱼ⁴`.  `c ≥ 1` is forced (`one_le_of_nearFlat`) and `c ≤ √d` is no restriction, being
implied by `‖v‖ = 1` whenever the hypothesis is non-vacuous.

`S` is left free here, so a measured spikiness may be substituted; but the `(13/10)·c` term does
not see `S` and is the dominant one, so a small `S` does not by itself make the bound small.  The
bound does not tend to zero in `d`: it tends to `(2 + 13c/5)·spread cbk`. -/
theorem abs_hadRate_sub_rhoSq_le_of_nearFlat_spikiness (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) {c : ℝ} (hc0 : 0 < c) (hcd : c ≤ Real.sqrt d)
    (hnf : ∀ j, |v j| ≤ c / Real.sqrt d) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ 2 * (3 * (∑ j, v j ^ 4) * Real.sqrt d + 13 / 10 * c + 1 + 1 / (4 * ((d:ℝ) - 1)))
          * spread cbk := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  refine abs_hadRate_sub_rhoSq_le_of_legBound hd hH v hv i cbk hq hqm ha hb hthr (fun x => ?_)
  exact abs_cdfGapInt_coordLaw_gaussian_le_of_bound_best hd hH hv i hnf (by positivity)
    (by rw [div_le_one hsd]; exact hcd) a x

/-- **The deployed rate at a near-flat direction, in `c` alone.**  Same hypotheses, with
`S ≤ c²/d` substituted:

  `|hadRate H (blockStage q) v − ρ²| ≤ 2·(3c²/√d + (13/10)·c + 1 + 1/(4(d−1)))·spread cbk`.

At `c = 1`, which is §59's flat case, the factor is `4.867` at `d = 512` against §59's `8.326`,
a reduction of `1.71`; at `c = 3/2` it is `6.499` and at `c = √3` it is `7.301`.  The factor
decreases in `d` to `2 + 13c/5`, so as in §59 the bound is `Θ(spread cbk)` and vanishes only if
the codebook is placed at the coordinate scale, which is a hypothesis about `cbk` that nothing
here proves.  `spread cbk` is left abstract and no codebook is named. -/
theorem abs_hadRate_sub_rhoSq_le_of_nearFlat (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) {c : ℝ} (hc0 : 0 < c) (hcd : c ≤ Real.sqrt d)
    (hnf : ∀ j, |v j| ≤ c / Real.sqrt d) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ 2 * (3 * c ^ 2 / Real.sqrt d + 13 / 10 * c + 1 + 1 / (4 * ((d:ℝ) - 1)))
          * spread cbk := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hS := sum_pow_four_le_sq_div_of_nearFlat hd hv hnf
  refine (abs_hadRate_sub_rhoSq_le_of_nearFlat_spikiness hd hH v hv i hc0 hcd hnf cbk hq hqm ha hb
    hthr).trans ?_
  have hkey : 3 * (∑ j, v j ^ 4) * Real.sqrt d ≤ 3 * c ^ 2 / Real.sqrt d := by
    have h1 : 3 * (∑ j, v j ^ 4) * Real.sqrt d ≤ 3 * (c ^ 2 / (d:ℝ)) * Real.sqrt d := by gcongr
    refine h1.trans (le_of_eq ?_)
    have hs2 : Real.sqrt (d:ℝ) * Real.sqrt (d:ℝ) = (d:ℝ) := Real.mul_self_sqrt hd0.le
    set s := Real.sqrt (d:ℝ) with hs
    rw [← hs2]
    field_simp
  exact mul_le_mul_of_nonneg_right (by linarith) (spread_nonneg cbk)

/-- **§59's spread case, with the constant reduced.**  At `c = 1`, which by
`one_le_of_nearFlat` is the smallest admissible value and is §59's exactly-flat hypothesis,

  `|hadRate H (blockStage q) v − ρ²| ≤ (6/√d + 23/5 + 1/(2(d−1)))·spread cbk`,

against §59's `(30/√d + 7)·spread cbk`.  The factor is `4.867` at `d = 512` where §59's is
`8.326`, and it decreases to `23/5` where §59's decreases to `7`.
`abs_hadRate_sub_rhoSq_le_of_spread` is untouched and not weakened by this; both are proved.

The caveats there apply verbatim.  This is `Θ(spread cbk)` and not `o(1)`; the deployed rotation
is not shown to attain `ρ²`; and the constants `6` and `23/5` are not claimed sharp, `13/10`
being the only one traceable to an evaluated integral. -/
theorem abs_hadRate_sub_rhoSq_le_of_spread_sharp (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) (hspread : ∀ j, |v j| ≤ 1 / Real.sqrt d)
    (cbk : Codebook K) {q : ℝ → ℝ} (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q)
    {a b : ℝ} (ha : a < -‖v‖) (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ (6 / Real.sqrt d + 23 / 5 + 1 / (2 * ((d:ℝ) - 1))) * spread cbk := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have h1 : (1:ℝ) ≤ Real.sqrt d := by
    rw [show (1:ℝ) = Real.sqrt 1 by simp]
    exact Real.sqrt_le_sqrt (by exact_mod_cast (by omega : (1:ℕ) ≤ d))
  have hnf : ∀ j, |v j| ≤ (1:ℝ) / Real.sqrt d := hspread
  have hd1 : (d:ℝ) - 1 ≠ 0 := by
    have h2 : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
    intro h; linarith
  have hsne : Real.sqrt (d:ℝ) ≠ 0 := ne_of_gt hsd
  refine (abs_hadRate_sub_rhoSq_le_of_nearFlat hd hH v hv i (c := 1) one_pos h1 hnf cbk hq hqm ha
    hb hthr).trans_eq ?_
  congr 1
  field_simp
  ring

/-- The same at a near-flat direction for the explicit step quantiser, which §52 shows is a
nearest-codeword map and is measurable.  It checks that the hypotheses are met by something
concrete and proves nothing new; every caveat of `abs_hadRate_sub_rhoSq_le_of_nearFlat` applies
verbatim. -/
theorem abs_hadRate_stepQuant_sub_rhoSq_le_of_nearFlat (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) {c : ℝ} (hc0 : 0 < c) (hcd : c ≤ Real.sqrt d)
    (hnf : ∀ j, |v j| ≤ c / Real.sqrt d) (cbk : Codebook K) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) (stepQuant cbk)) v
        - rhoSq d (stepQuant cbk) (sphericalCoordLaw d)|
      ≤ 2 * (3 * c ^ 2 / Real.sqrt d + 13 / 10 * c + 1 + 1 / (4 * ((d:ℝ) - 1)))
          * spread cbk :=
  abs_hadRate_sub_rhoSq_le_of_nearFlat hd hH v hv i hc0 hcd hnf cbk (isNearest_stepQuant cbk)
    (measurable_stepQuant cbk) ha hb hthr

/-- **Below the free bound at the dimension the paper ships.**  At `d = 512`, for a flat `H`, a
unit direction near-flat with `c ≤ 7/4`, any ordered codebook whose thresholds lie in a window
around the support and whose **spread is at most `3.0208/√512`**, and any measurable
nearest-codeword map,

  `|hadRate H (blockStage q) v − ρ²| < 1`.

`1` is the bound that holds for free whenever `0` is a codeword, since then `|t − q t| ≤ |t|`
pointwise; so this is the first statement in this development that beats it at `d = 512`, where
§59's `(30/√d + 7)·spread cbk` gives `1.112` at the same spread.  The margin is not large: the
factor is `7.365` against the `7.490` that `spread cbk = 3.0208/√512` allows, and at `c = 1` the
bound is `0.650`.

**The codebook width is a hypothesis, not a fact proved here.**  `3.0208/√512` is supplied by
`hw`; no codebook is constructed, named or analysed anywhere in this file, and nothing is claimed
about any codebook the paper ships.  Nor is `1` the target: `ρ² = 0.1171` is roughly six times
smaller, and at `c = 1` this route reaches it only near `d ≈ 1.8·10⁴`.  `c` bounds the largest
coordinate of `v`, which a Haar-random direction does not satisfy at `c = 7/4` in dimension
`512`. -/
theorem abs_hadRate_sub_rhoSq_lt_one_of_nearFlat_512 {H : Od 512} (hH : IsFlat H)
    (v : Ed 512) (hv : ‖v‖ = 1) (i : Fin 512) {c : ℝ} (hc0 : 0 < c) (hc : c ≤ 7 / 4)
    (hnf : ∀ j, |v j| ≤ c / Real.sqrt 512) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hw : spread cbk ≤ 3.0208 / Real.sqrt 512) :
    |hadRate H (blockStage (m := 512) (F := ℝ) q) v
        - rhoSq 512 q (sphericalCoordLaw 512)| < 1 := by
  have hs := sqrt_512_ge
  have hsp : (0:ℝ) < Real.sqrt 512 := by linarith
  have hcast : Real.sqrt ((512:ℕ):ℝ) = Real.sqrt 512 := by norm_num
  have hnf' : ∀ j, |v j| ≤ c / Real.sqrt ((512:ℕ):ℝ) := by simpa [hcast] using hnf
  have hmain := abs_hadRate_sub_rhoSq_le_of_nearFlat (d := 512) (by norm_num) hH v hv i hc0
    (by rw [hcast]; linarith) hnf' cbk hq hqm ha hb hthr
  rw [hcast] at hmain
  push_cast at hmain
  have hsp0 : 0 ≤ spread cbk := spread_nonneg cbk
  have hfac : 2 * (3 * c ^ 2 / Real.sqrt 512 + 13 / 10 * c + 1 + 1 / (4 * ((512:ℝ) - 1)))
      ≤ 37 / 5 := by
    have hc2 : c ^ 2 ≤ 49 / 16 := by nlinarith
    have h1 : 3 * c ^ 2 / Real.sqrt 512 ≤ 3 * (49 / 16) / 22.6 := by
      rw [div_le_div_iff₀ (by positivity) (by norm_num)]
      nlinarith [hs, hc2]
    norm_num at h1 ⊢
    linarith
  have hstep : |hadRate H (blockStage (m := 512) (F := ℝ) q) v
      - rhoSq 512 q (sphericalCoordLaw 512)| ≤ 37 / 5 * (3.0208 / Real.sqrt 512) := by
    refine hmain.trans ((mul_le_mul_of_nonneg_right hfac hsp0).trans ?_)
    exact mul_le_mul_of_nonneg_left hw (by norm_num)
  refine lt_of_le_of_lt hstep ?_
  have heq : (37:ℝ) / 5 * (3.0208 / Real.sqrt 512) = (37 * 3.0208) / (5 * Real.sqrt 512) := by
    ring
  rw [heq, div_lt_one (by positivity)]
  nlinarith [hs]

end Threshold

end NearFlatLeg
