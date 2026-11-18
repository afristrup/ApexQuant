
/-! ## §66  The middle range: the damping factor integrated rather than bounded pointwise

`FourierGap.gapWeight_le_mid_sharp` charges `gapWeight ≤ (40/9)·∑ⱼvⱼ⁴/u²` on `√d < u ≤ √d/M`, and
`40/9` is the **supremum** of `z²e^{−z}/(5/9)` at `z = u²/(4d)`, not its integral: the true damped
estimate `norm_gap_le_damped` reads `gapWeight ≤ (∑ⱼvⱼ⁴/(2d²))·u²·e^{−u²/(4d)}`, and replacing
`e^{−u²/(4d)}` by its pointwise cover `(5/9)(4d/u²)²` throws the Gaussian away.  Its own docstring
says the fix is "the damping factor integrated rather than bounded pointwise, which is not done
here".  That is what §66 does.

**What is replaced, and by what.**  Writing `r = √d`, integration by parts on `(r, T]` against the
explicit antiderivative `−2r²u·e^{−u²/(4r²)}` gives

  `∫_r^T u²e^{−u²/(4r²)} du ≤ 2r³e^{−1/4} + 2r²·∫_r^∞ e^{−u²/(4r²)} du`,

and the remaining Gaussian mass is *not* estimated by a Mills ratio but taken from the exact
half-line integral: `∫_0^∞ e^{−u²/(4r²)} du = √π·r` by `integral_gaussian_Ioi`, minus the
elementary lower bound `∫_0^r e^{−u²/(4r²)} du ≥ ∫_0^r (1 − u²/(4r²)) du = (11/12)·r` that
`1 + x ≤ eˣ` supplies.  So `∫_r^∞ e^{−u²/(4r²)} du ≤ (√π − 11/12)·r`, which at `r = 1/2` is the
tail number `J = ∫_{1/2}^∞ e^{−t²} dt ≤ √π/2 − 11/24 = 0.42789` recorded as `gaussTail_half_le`
(the true value is `0.42495`).  The middle range then integrates to

  `∫_{√d}^{√d/M} gapWeight ≤ (e^{−1/4} + √π − 11/12)·∑ⱼvⱼ⁴/√d = 1.6346·∑ⱼvⱼ⁴/√d`

against `40/9 = 4.4444` from the pointwise cover, a factor `2.72`.  The floor of the exact
calculation is `e^{−1/4} + 2J = 1.6287`, so `1.6346` is within `0.4%` of everything this route
can give.

**The constant.**  `integral_gapWeight_le_midsharp` reads

  `∫₀^∞ ‖φ_μ − φ_ν‖/u² du ≤ (A_mid·∑ⱼvⱼ⁴ + K(M)·M)/√d`,  `A_mid = e^{−1/4} + √π − 3/4 = 1.8013`,

where `A_mid = 1/6 + 1.6346` collects the low range `(0, √d]` of `integral_gapWeight_le_sharp`,
whose coefficient `1/6` is copied verbatim, together with the new middle range.  The tail constant
`K(M) = 1/π + (2√(2π)/π²)M + 16M⁴` of §65 is carried through **unchanged**: nothing here touches
the range `u > √d/M`, and `83/18 = 4.6111` falls to `1.8013`, a factor `2.56`.

**What is *not* done.**  The improvement is confined to the spikiness coefficient.  The `M`-linear
tail term is untouched and is still the dominant one at a near-flat direction, so the composed
bound is still `Θ(spread cbk)` and still not `o(1)`; the rate in `d` is exactly as it was.  The
damped estimate `norm_gap_le_damped` that this integrates is itself a fourth-moment bound and is
not improved, `‖φ_μ‖ ≤ 1` past the period is still the only thing known there, `spread cbk` is
left abstract, and no codebook is constructed or analysed.  `gapWeight_le_mid_sharp`,
`integral_gapWeight_le_sharp` and `integral_gapWeight_le_tailsharp` are untouched, still proved,
and not weakened.

**Numerically, at the dimension the paper ships.**  The leg-1 quadratic coefficient `(2/π)·A_mid`
falls from `2.9355` (shipped rounded up to `3`) to `1.1468`.  At `d = 512`, `c = 3.53` and a
codebook of spread at most `3.0164/√512`, the composed near-flat bound falls from `0.952` to
`0.679`; `Threshold.abs_hadRate_sub_rhoSq_le_of_nearFlat_mid_512` records it as `69/100`.  It is
still far from `ρ² = 0.1171` and cannot reach it on this route, since the coupling leg alone
contributes `2·spread cbk = 0.267` at that spread.  The codebook width is a hypothesis. -/

noncomputable section MidRange

open Orthogonal Hadamard CoordLaw

namespace FourierGap

/-! ### §66a  The Gaussian tail past half a standard deviation

Nothing in this block mentions a rotation, a dimension or a law: it is the half-line Gaussian
integral, a quadratic lower bound on `eˣ`, and their difference. -/

/-- `u ↦ e^{−u²/(4r²)}` is integrable on the line for `r > 0`; it is `integrable_exp_neg_mul_sq`
at `b = 1/(4r²)` after clearing the division. -/
theorem integrable_exp_neg_sq_div {r : ℝ} (hr : 0 < r) :
    Integrable (fun u : ℝ => Real.exp (-(u ^ 2 / (4 * r ^ 2)))) := by
  have hb : (0:ℝ) < 1 / (4 * r ^ 2) := by positivity
  have heq : (fun u : ℝ => Real.exp (-(u ^ 2 / (4 * r ^ 2))))
      = fun u : ℝ => Real.exp (-(1 / (4 * r ^ 2)) * u ^ 2) := by
    funext u; congr 1; field_simp
  rw [heq]; exact integrable_exp_neg_mul_sq hb

/-- **The Gaussian tail past one standard deviation of the scale `2r`.**  For `r > 0`,

  `∫_{u > r} e^{−u²/(4r²)} du ≤ (√π − 11/12)·r`.

The exact half-line mass `√π·r` is `integral_gaussian_Ioi`; the head `(0, r]` is bounded below by
`∫_0^r (1 − u²/(4r²)) du = (11/12)·r`, which is `1 + x ≤ eˣ` and nothing more.  The true constant
is `√π − ∫_0^1 e^{−t²/4} dt = 0.84985`, against `√π − 11/12 = 0.85579`; the quadratic lower bound
costs `0.006` here and is not sharp at larger cut points. -/
theorem integral_Ioi_exp_neg_sq_div_le {r : ℝ} (hr : 0 < r) :
    (∫ u in Set.Ioi r, Real.exp (-(u ^ 2 / (4 * r ^ 2)))) ≤ (Real.sqrt π - 11 / 12) * r := by
  set b : ℝ := 1 / (4 * r ^ 2) with hbdef
  have hb : 0 < b := by rw [hbdef]; positivity
  have hcongr : ∀ u : ℝ, Real.exp (-(u ^ 2 / (4 * r ^ 2))) = Real.exp (-b * u ^ 2) := by
    intro u; congr 1; rw [hbdef]; field_simp
  have hint := integrable_exp_neg_sq_div hr
  have hintA : IntegrableOn (fun u : ℝ => Real.exp (-(u ^ 2 / (4 * r ^ 2)))) (Set.Ioc (0:ℝ) r) :=
    hint.integrableOn
  have hintB : IntegrableOn (fun u : ℝ => Real.exp (-(u ^ 2 / (4 * r ^ 2)))) (Set.Ioi r) :=
    hint.integrableOn
  have htot : (∫ u in Set.Ioi (0:ℝ), Real.exp (-(u ^ 2 / (4 * r ^ 2)))) = Real.sqrt π * r := by
    rw [setIntegral_congr_fun measurableSet_Ioi (fun u _ => hcongr u), integral_gaussian_Ioi]
    have hval : π / b = (2 * r) ^ 2 * π := by rw [hbdef]; field_simp; ring
    rw [hval, Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]
    ring
  have hsplit : (∫ u in Set.Ioi (0:ℝ), Real.exp (-(u ^ 2 / (4 * r ^ 2))))
      = (∫ u in Set.Ioc (0:ℝ) r, Real.exp (-(u ^ 2 / (4 * r ^ 2))))
        + ∫ u in Set.Ioi r, Real.exp (-(u ^ 2 / (4 * r ^ 2))) := by
    rw [Ioi_zero_split _ hr,
      setIntegral_union (Ioc_disjoint_Ioi _ _) measurableSet_Ioi hintA hintB]
  have hlow : 11 / 12 * r ≤ ∫ u in Set.Ioc (0:ℝ) r, Real.exp (-(u ^ 2 / (4 * r ^ 2))) := by
    have hmin : IntegrableOn (fun u : ℝ => 1 - u ^ 2 / (4 * r ^ 2)) (Set.Ioc (0:ℝ) r) := by
      apply Continuous.integrableOn_Ioc; fun_prop
    have hmono : (∫ u in Set.Ioc (0:ℝ) r, (1 - u ^ 2 / (4 * r ^ 2)))
        ≤ ∫ u in Set.Ioc (0:ℝ) r, Real.exp (-(u ^ 2 / (4 * r ^ 2))) := by
      refine setIntegral_mono_on hmin hintA measurableSet_Ioc fun u _ => ?_
      have := Real.add_one_le_exp (-(u ^ 2 / (4 * r ^ 2)))
      linarith
    have hval : (∫ u in Set.Ioc (0:ℝ) r, (1 - u ^ 2 / (4 * r ^ 2))) = 11 / 12 * r := by
      have hsub : (∫ u in Set.Ioc (0:ℝ) r, (1 - u ^ 2 / (4 * r ^ 2)))
          = (∫ u in Set.Ioc (0:ℝ) r, (1:ℝ)) - ∫ u in Set.Ioc (0:ℝ) r, u ^ 2 / (4 * r ^ 2) := by
        refine integral_sub (integrableOn_const (by simp)) ?_
        apply Continuous.integrableOn_Ioc; fun_prop
      rw [hsub, setIntegral_const, Real.volume_real_Ioc_of_le hr.le, smul_eq_mul]
      have hdiv : (∫ u in Set.Ioc (0:ℝ) r, u ^ 2 / (4 * r ^ 2))
          = (∫ u in Set.Ioc (0:ℝ) r, u ^ 2) / (4 * r ^ 2) := by
        simp_rw [div_eq_mul_inv]
        rw [integral_mul_const]
      rw [hdiv, integral_Ioc_sq hr.le]
      field_simp
      ring
    linarith [hval ▸ hmono]
  linarith [htot, hsplit, hlow]

/-- **The tail number.**  `J := ∫_{1/2}^∞ e^{−t²} dt ≤ √π/2 − 11/24`, the `r = 1/2` case of
`integral_Ioi_exp_neg_sq_div_le`.  Numerically `0.42789` against the true `0.42495`.  Mathlib's
`integral_gaussian_Ioi` is over `Ioi 0` only, so the cut at `1/2` has to be paid for by a head
estimate, and the quadratic one used here is what costs the `0.003`. -/
theorem gaussTail_half_le : (∫ t in Set.Ioi (1/2 : ℝ), Real.exp (-t ^ 2))
    ≤ Real.sqrt π / 2 - 11 / 24 := by
  have h := integral_Ioi_exp_neg_sq_div_le (r := 1/2) (by norm_num)
  have hc : ∀ t : ℝ, Real.exp (-(t ^ 2 / (4 * (1/2 : ℝ) ^ 2))) = Real.exp (-t ^ 2) := by
    intro t; congr 1; norm_num
  rw [setIntegral_congr_fun measurableSet_Ioi (fun t _ => hc t)] at h
  linarith

/-- `√π ≤ 1.77246`.  The true value is `1.7724538…`; the margin against `Real.pi_lt_d6` is
`2·10⁻⁵` and nothing here evaluates it further. -/
theorem sqrt_pi_le : Real.sqrt π ≤ 1.77246 := by
  have hpi : π < 3.141593 := Real.pi_lt_d6
  rw [show (1.77246:ℝ) = Real.sqrt (1.77246 ^ 2) from (Real.sqrt_sq (by norm_num)).symm]
  exact Real.sqrt_le_sqrt (by nlinarith)

/-- `1 ≤ √π`, used only for the sign of `midConst`. -/
theorem one_le_sqrt_pi : (1:ℝ) ≤ Real.sqrt π := by
  have h : Real.sqrt 1 ≤ Real.sqrt π := Real.sqrt_le_sqrt (by linarith [Real.pi_gt_three])
  simpa using h

/-- `e^{−1/4} ≤ 250/321 = 0.778816`.  From `Real.exp_one_gt_d9` and `(1.284)⁴ < e`; the true value
is `0.7788008`. -/
theorem exp_neg_quarter_le : Real.exp (-(1/4 : ℝ)) ≤ 250 / 321 := by
  have hx : (0:ℝ) < Real.exp (1/4 : ℝ) := Real.exp_pos _
  have he : (2.7182818283:ℝ) < Real.exp 1 := Real.exp_one_gt_d9
  have hp : Real.exp (1/4 : ℝ) * Real.exp (1/4 : ℝ) * Real.exp (1/4 : ℝ) * Real.exp (1/4 : ℝ)
      = Real.exp 1 := by
    rw [← Real.exp_add, ← Real.exp_add, ← Real.exp_add]
    norm_num
  have hq : (321:ℝ) / 250 ≤ Real.exp (1/4 : ℝ) := by
    by_contra hcon
    rw [not_le] at hcon
    have h2 : Real.exp (1/4 : ℝ) * Real.exp (1/4 : ℝ) < (321:ℝ) / 250 * (321 / 250) := by
      nlinarith
    have h4 : Real.exp (1/4 : ℝ) * Real.exp (1/4 : ℝ) * Real.exp (1/4 : ℝ) * Real.exp (1/4 : ℝ)
        < ((321:ℝ) / 250 * (321 / 250)) * ((321:ℝ) / 250 * (321 / 250)) := by
      nlinarith [hx, h2]
    rw [hp] at h4
    norm_num at h4
    linarith
  have hprod : Real.exp (-(1/4 : ℝ)) * Real.exp (1/4 : ℝ) = 1 := by
    rw [← Real.exp_add]; norm_num
  have h := mul_le_mul_of_nonneg_left hq (Real.exp_pos (-(1/4 : ℝ))).le
  rw [hprod] at h
  linarith

/-! ### §66b  The damped profile, integrated by parts

The one analytic step this section adds: an explicit antiderivative for the part of
`u²e^{−u²/(4r²)}` that has one, and `integral_Ioi_exp_neg_sq_div_le` for the part that does not. -/

/-- The antiderivative.  `d/du [−2r²u·e^{−u²/(4r²)}] = (u² − 2r²)·e^{−u²/(4r²)}`, which is the
integration by parts of `∫u²e^{−u²/(4r²)}` written as a single derivative. -/
theorem hasDerivAt_midAnti {r : ℝ} (hr : 0 < r) (u : ℝ) :
    HasDerivAt (fun x : ℝ => -(2 * r ^ 2 * x * Real.exp (-(x ^ 2 / (4 * r ^ 2)))))
      ((u ^ 2 - 2 * r ^ 2) * Real.exp (-(u ^ 2 / (4 * r ^ 2)))) u := by
  have hr0 : r ≠ 0 := ne_of_gt hr
  have h0 : HasDerivAt (fun x : ℝ => x ^ 2 / (4 * r ^ 2)) (2 * u / (4 * r ^ 2)) u := by
    have h := (hasDerivAt_pow 2 u).div_const (4 * r ^ 2)
    simpa using h
  have h1 : HasDerivAt (fun x : ℝ => -(x ^ 2 / (4 * r ^ 2))) (-(2 * u / (4 * r ^ 2))) u := h0.neg
  have h2 : HasDerivAt (fun x : ℝ => Real.exp (-(x ^ 2 / (4 * r ^ 2))))
      (Real.exp (-(u ^ 2 / (4 * r ^ 2))) * -(2 * u / (4 * r ^ 2))) u := h1.exp
  have h3 : HasDerivAt (fun x : ℝ => 2 * r ^ 2 * x) (2 * r ^ 2) u := by
    simpa using (hasDerivAt_id u).const_mul (2 * r ^ 2)
  have h4 : HasDerivAt (fun x : ℝ => 2 * r ^ 2 * x * Real.exp (-(x ^ 2 / (4 * r ^ 2))))
      (2 * r ^ 2 * Real.exp (-(u ^ 2 / (4 * r ^ 2)))
        + 2 * r ^ 2 * u * (Real.exp (-(u ^ 2 / (4 * r ^ 2))) * -(2 * u / (4 * r ^ 2)))) u :=
    h3.mul h2
  have h5 : HasDerivAt (fun x : ℝ => -(2 * r ^ 2 * x * Real.exp (-(x ^ 2 / (4 * r ^ 2)))))
      (-(2 * r ^ 2 * Real.exp (-(u ^ 2 / (4 * r ^ 2)))
        + 2 * r ^ 2 * u * (Real.exp (-(u ^ 2 / (4 * r ^ 2))) * -(2 * u / (4 * r ^ 2))))) u := h4.neg
  convert h5 using 1
  field_simp
  ring

/-- **The damped profile on the middle range.**  For `0 < r ≤ T`,

  `∫_{(r, T]} u²e^{−u²/(4r²)} du ≤ (2e^{−1/4} + 2√π − 11/6)·r³ = 3.2692·r³`.

The boundary term at `T` is discarded, so the bound is uniform in `T`; the exact value at
`T = ∞` is `2e^{−1/4} + 4J = 3.2574·r³`, `0.4%` below.  It is the integrated form of the damping
factor that `gapWeight_le_mid_sharp` bounds pointwise by its supremum. -/
theorem integral_Ioc_sq_mul_exp_le {r T : ℝ} (hr : 0 < r) (hT : r ≤ T) :
    (∫ u in Set.Ioc r T, u ^ 2 * Real.exp (-(u ^ 2 / (4 * r ^ 2))))
      ≤ (2 * Real.exp (-(1/4)) + 2 * Real.sqrt π - 11/6) * r ^ 3 := by
  have hr0 : r ≠ 0 := ne_of_gt hr
  have hc1 : Continuous fun u : ℝ => (u ^ 2 - 2 * r ^ 2) * Real.exp (-(u ^ 2 / (4 * r ^ 2))) := by
    fun_prop
  have hc2 : Continuous fun u : ℝ => 2 * r ^ 2 * Real.exp (-(u ^ 2 / (4 * r ^ 2))) := by fun_prop
  have hi1 : IntegrableOn (fun u : ℝ => (u ^ 2 - 2 * r ^ 2) * Real.exp (-(u ^ 2 / (4 * r ^ 2))))
      (Set.Ioc r T) := hc1.integrableOn_Ioc
  have hi2 : IntegrableOn (fun u : ℝ => 2 * r ^ 2 * Real.exp (-(u ^ 2 / (4 * r ^ 2))))
      (Set.Ioc r T) := hc2.integrableOn_Ioc
  have hsum : (∫ u in Set.Ioc r T, u ^ 2 * Real.exp (-(u ^ 2 / (4 * r ^ 2))))
      = (∫ u in Set.Ioc r T, (u ^ 2 - 2 * r ^ 2) * Real.exp (-(u ^ 2 / (4 * r ^ 2))))
        + ∫ u in Set.Ioc r T, 2 * r ^ 2 * Real.exp (-(u ^ 2 / (4 * r ^ 2))) := by
    rw [← integral_add hi1 hi2]
    refine setIntegral_congr_fun measurableSet_Ioc fun u _ => ?_
    ring
  have hpart1 : (∫ u in Set.Ioc r T, (u ^ 2 - 2 * r ^ 2) * Real.exp (-(u ^ 2 / (4 * r ^ 2))))
      ≤ 2 * Real.exp (-(1/4)) * r ^ 3 := by
    have hftc := intervalIntegral.integral_eq_sub_of_hasDerivAt
      (f := fun x : ℝ => -(2 * r ^ 2 * x * Real.exp (-(x ^ 2 / (4 * r ^ 2)))))
      (f' := fun u : ℝ => (u ^ 2 - 2 * r ^ 2) * Real.exp (-(u ^ 2 / (4 * r ^ 2))))
      (fun x _ => hasDerivAt_midAnti hr x) (hc1.intervalIntegrable r T)
    rw [intervalIntegral.integral_of_le hT] at hftc
    rw [hftc]
    have hTt : 0 ≤ 2 * r ^ 2 * T * Real.exp (-(T ^ 2 / (4 * r ^ 2))) := by
      have hT0 : 0 < T := lt_of_lt_of_le hr hT
      positivity
    have hrv : -(2 * r ^ 2 * r * Real.exp (-(r ^ 2 / (4 * r ^ 2))))
        = -(2 * Real.exp (-(1/4)) * r ^ 3) := by
      have hq : r ^ 2 / (4 * r ^ 2) = 1 / 4 := by field_simp
      rw [hq]; ring
    rw [hrv]
    linarith
  have hpart2 : (∫ u in Set.Ioc r T, 2 * r ^ 2 * Real.exp (-(u ^ 2 / (4 * r ^ 2))))
      ≤ (2 * Real.sqrt π - 11 / 6) * r ^ 3 := by
    have hintB : IntegrableOn (fun u : ℝ => Real.exp (-(u ^ 2 / (4 * r ^ 2)))) (Set.Ioi r) :=
      (integrable_exp_neg_sq_div hr).integrableOn
    have hmono : (∫ u in Set.Ioc r T, Real.exp (-(u ^ 2 / (4 * r ^ 2))))
        ≤ ∫ u in Set.Ioi r, Real.exp (-(u ^ 2 / (4 * r ^ 2))) := by
      refine setIntegral_mono_set hintB ?_ (Filter.Eventually.of_forall fun u hu => hu.1)
      filter_upwards with u using (Real.exp_pos _).le
    have hg := integral_Ioi_exp_neg_sq_div_le hr
    rw [integral_const_mul]
    have hc0 : (0:ℝ) ≤ 2 * r ^ 2 := by positivity
    calc 2 * r ^ 2 * ∫ u in Set.Ioc r T, Real.exp (-(u ^ 2 / (4 * r ^ 2)))
        ≤ 2 * r ^ 2 * ((Real.sqrt π - 11 / 12) * r) :=
          mul_le_mul_of_nonneg_left (hmono.trans hg) hc0
      _ = (2 * Real.sqrt π - 11 / 6) * r ^ 3 := by ring
  rw [hsum]
  linarith

/-- The same, with the cut written as `√D` so that the deployed statement does not have to carry
`(√d)² = d` through a `field_simp`. -/
theorem integral_Ioc_sq_mul_exp_le' {D T : ℝ} (hD : 0 < D) (hT : Real.sqrt D ≤ T) :
    (∫ u in Set.Ioc (Real.sqrt D) T, u ^ 2 * Real.exp (-(u ^ 2 / (4 * D))))
      ≤ (2 * Real.exp (-(1/4)) + 2 * Real.sqrt π - 11/6) * Real.sqrt D ^ 3 := by
  have h := integral_Ioc_sq_mul_exp_le (r := Real.sqrt D) (Real.sqrt_pos.mpr hD) hT
  rwa [Real.sq_sqrt hD.le] at h

/-! ### §66c  The deployed middle range

From here the pair is fixed, as in §58b and §65b: `coordLaw H v i` against `N(0, 1/d)` at a unit
direction `v` with `|vⱼ| ≤ M`. -/

section MidDeployed

variable {d : ℕ}

/-- **Middle range, damped.**  `norm_gap_le_damped` divided by `u²` and with the damping factor
kept: for `0 < u` with `u·M ≤ √d`,

  `gapWeight ≤ (∑ⱼvⱼ⁴/(2d²))·u²·e^{−u²/(4d)}`.

This is `gapWeight_le_mid_sharp` before the pointwise cover `e^{−z} ≤ (5/9)/z²` is applied, and it
is the only thing §66 integrates.  Nothing beyond `norm_gap_le_damped` is used, and that estimate
is not improved here. -/
theorem gapWeight_le_mid_damped {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d)
    {M : ℝ} (hM : ∀ j, |v j| ≤ M) {u : ℝ} (hu : 0 < u) (huM : u * M ≤ Real.sqrt d) :
    gapWeight H v i u
      ≤ (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * (u ^ 2 * Real.exp (-(u ^ 2 / (4 * (d:ℝ))))) := by
  have h := norm_gap_le_damped hH hv i hM hu huM
  have heq : -(u ^ 2 / (d:ℝ)) / 4 = -(u ^ 2 / (4 * (d:ℝ))) := by ring
  rw [heq] at h
  rw [gapWeight, div_le_iff₀ (pow_pos hu 2)]
  calc ‖charFun (coordLaw H v i) u - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖
      ≤ u ^ 4 * (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) * Real.exp (-(u ^ 2 / (4 * (d:ℝ)))) := h
    _ = (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2)
          * (u ^ 2 * Real.exp (-(u ^ 2 / (4 * (d:ℝ))))) * u ^ 2 := by ring

/-- **The middle range, integrated.**  For `√d ≤ T` with `T·M ≤ √d`,

  `∫_{(√d, T]} gapWeight ≤ (e^{−1/4} + √π − 11/12)·∑ⱼvⱼ⁴/√d = 1.6346·∑ⱼvⱼ⁴/√d`,

against the `(40/9)·∑ⱼvⱼ⁴/√d = 4.4444·∑ⱼvⱼ⁴/√d` that `gapWeight_le_mid_sharp` integrates to, a
factor `2.72`.  The bound does not depend on `T`, which is where the tail range takes over.
`gapWeight_le_mid_sharp` is untouched and remains proved. -/
theorem integral_gapWeight_Ioc_le_mid {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1)
    (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hd0 : (0:ℝ) < (d:ℝ)) {T : ℝ}
    (hT : Real.sqrt d ≤ T) (hTM : T * M ≤ Real.sqrt d) :
    (∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u)
      ≤ (Real.exp (-(1/4)) + Real.sqrt π - 11 / 12) * (∑ j, v j ^ 4) / Real.sqrt d := by
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hM0 : 0 ≤ M := le_trans (abs_nonneg _) (hM i)
  have hS : 0 ≤ ∑ j, v j ^ 4 := Finset.sum_nonneg fun j _ => by positivity
  have hint : IntegrableOn (gapWeight H v i) (Set.Ioc (Real.sqrt d) T) :=
    (integrableOn_gapWeight_Ioi H v i hd0).mono_set fun u hu => hu.1
  have hmaj : IntegrableOn (fun u : ℝ => (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2)
      * (u ^ 2 * Real.exp (-(u ^ 2 / (4 * (d:ℝ)))))) (Set.Ioc (Real.sqrt d) T) := by
    apply Continuous.integrableOn_Ioc; fun_prop
  have hstep : (∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u)
      ≤ ∫ u in Set.Ioc (Real.sqrt d) T, (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2)
          * (u ^ 2 * Real.exp (-(u ^ 2 / (4 * (d:ℝ))))) := by
    refine setIntegral_mono_on hint hmaj measurableSet_Ioc fun u hu => ?_
    refine gapWeight_le_mid_damped hH hv i hM (lt_trans hsd hu.1) ?_
    exact le_trans (mul_le_mul_of_nonneg_right hu.2 hM0) hTM
  have hprof := integral_Ioc_sq_mul_exp_le' hd0 hT
  have hc0 : (0:ℝ) ≤ (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2) := by positivity
  have hval : (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2)
      * ((2 * Real.exp (-(1/4)) + 2 * Real.sqrt π - 11/6) * Real.sqrt d ^ 3)
      = (Real.exp (-(1/4)) + Real.sqrt π - 11 / 12) * (∑ j, v j ^ 4) / Real.sqrt d := by
    have hs2 : Real.sqrt (d:ℝ) * Real.sqrt (d:ℝ) = (d:ℝ) := Real.mul_self_sqrt hd0.le
    set s := Real.sqrt (d:ℝ) with hs
    have hs0 : s ≠ 0 := ne_of_gt hsd
    rw [← hs2]
    field_simp
    ring
  calc (∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u)
      ≤ ∫ u in Set.Ioc (Real.sqrt d) T, (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2)
          * (u ^ 2 * Real.exp (-(u ^ 2 / (4 * (d:ℝ))))) := hstep
    _ = (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2)
          * ∫ u in Set.Ioc (Real.sqrt d) T, u ^ 2 * Real.exp (-(u ^ 2 / (4 * (d:ℝ)))) :=
        integral_const_mul _ _
    _ ≤ (∑ j, v j ^ 4) / (2 * (d:ℝ) ^ 2)
          * ((2 * Real.exp (-(1/4)) + 2 * Real.sqrt π - 11/6) * Real.sqrt d ^ 3) :=
        mul_le_mul_of_nonneg_left hprof hc0
    _ = (Real.exp (-(1/4)) + Real.sqrt π - 11 / 12) * (∑ j, v j ^ 4) / Real.sqrt d := hval

end MidDeployed

/-! ### §66d  The constant, and the five-piece integral -/

/-- **The spikiness constant with the damping integrated.**  `A_mid = e^{−1/4} + √π − 3/4`, the
replacement for the `83/18` of `integral_gapWeight_le_sharp` and `integral_gapWeight_le_tailsharp`.
It is `1/6` from the low range `(0, √d]`, copied verbatim from there, plus
`e^{−1/4} + √π − 11/12 = 1.6346` from the middle range.  `A_mid = 1.8013` against `83/18 = 4.6111`.
The floor for this route is `1/6 + e^{−1/4} + 2J = 1.7954`, reached only with the exact tail
number `J`; `A_mid` is `0.3%` above it.  Nothing about the tail range `u > √d/M` is claimed here. -/
def midConst : ℝ := Real.exp (-(1/4)) + Real.sqrt π - 3/4

/-- `0 < A_mid`, needed only for the sign of the leg-1 coefficient. -/
theorem midConst_pos : 0 < midConst := by
  have h1 := one_le_sqrt_pi
  have h2 : (0:ℝ) < Real.exp (-(1/4)) := Real.exp_pos _
  rw [midConst]; linarith

/-- `A_mid ≤ 18013/10000`.  Both roundings are above the true values
(`e^{−1/4} = 0.7788008`, `√π = 1.7724539`); the true `A_mid` is `1.8012547`. -/
theorem midConst_le : midConst ≤ 18013 / 10000 := by
  have h1 := exp_neg_quarter_le
  have h2 := sqrt_pi_le
  rw [midConst]
  norm_num at h2 ⊢
  linarith

/-- The low and middle ranges recombined: `1/6` from `(0, √d]` and `E` from `(√d, T]`. -/
theorem arith_mid_low {S E r : ℝ} (hr : 0 < r) :
    S / (2 * (r ^ 2) ^ 2) * (r ^ 3 / 3) + E * S / r = (1 / 6 + E) * S / r := by
  field_simp
  ring

section MidIntegral

variable {d : ℕ}

/-- **The five-piece estimate, with the damping integrated.**  Same hypotheses and same shape as
`integral_gapWeight_le_tailsharp`, whose `83/18` is replaced:

  `∫₀^∞ ‖φ_μ − φ_ν‖/u² du ≤ (A_mid·∑ⱼvⱼ⁴ + K(M)·M)/√d`,  `A_mid = e^{−1/4} + √π − 3/4 = 1.8013`,

with `K(M) = 1/π + (2√(2π)/π²)M + 16M⁴` the tail constant of §65, **carried through unchanged**.
The split is at `√d`, `√d/M`, `(π/2)·√d/M` and `π·√d/M` exactly as in §65; the last three pieces
and their constant `K` are copied verbatim, so **nothing here improves the tail term**, which is
still the dominant one at a near-flat direction.  The gain is confined to the second piece and is
a factor `2.72` there, `2.56` on `A_mid` as a whole.

`integral_gapWeight_le_sharp`, `integral_gapWeight_le_tailsharp` and `gapWeight_le_mid_sharp` are
untouched and remain proved.  The rate in `d` is unchanged: this is `Θ(1/√d)` exactly as before. -/
theorem integral_gapWeight_le_midsharp {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1)
    (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M) (hM1 : M ≤ 1)
    (hd0 : (0:ℝ) < (d:ℝ)) :
    (∫ u in Set.Ioi (0:ℝ), gapWeight H v i u)
      ≤ (midConst * (∑ j, v j ^ 4) + tailConst M * M) / Real.sqrt d := by
  have hpi : (3:ℝ) < π := Real.pi_gt_three
  have hpi0 : (0:ℝ) < π := Real.pi_pos
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hsd2 : Real.sqrt d ^ 2 = (d:ℝ) := Real.sq_sqrt hd0.le
  set S : ℝ := ∑ j, v j ^ 4 with hSdef
  have hS : 0 ≤ S := Finset.sum_nonneg fun j _ => by positivity
  set T : ℝ := Real.sqrt d / M with hTdef
  have hTpos : 0 < T := div_pos hsd hM0
  have hTge : Real.sqrt d ≤ T := by
    rw [hTdef, le_div_iff₀ hM0]
    nlinarith
  set E : ℝ := Real.exp (-(1 / (2 * M ^ 2))) with hEdef
  have hE0 : 0 < E := Real.exp_pos _
  have hEb : E ≤ 20 / 9 * M ^ 4 := by
    have hz : (0:ℝ) < 1 / (2 * M ^ 2) := by positivity
    have h := exp_neg_le_five_ninths_div_sq hz
    rw [hEdef]
    refine h.trans (le_of_eq ?_)
    field_simp
    ring
  have hb1 : T ≤ π / 2 * T := by nlinarith
  have hb2 : π / 2 * T ≤ π * T := by nlinarith
  have hpiT : 0 < π * T := by positivity
  have hintA : IntegrableOn (gapWeight H v i) (Set.Ioc 0 (Real.sqrt d)) :=
    integrableOn_gapWeight_Ioc hH hv i hM hM1
  have hintR : IntegrableOn (gapWeight H v i) (Set.Ioi (Real.sqrt d)) :=
    integrableOn_gapWeight_Ioi H v i hd0
  have hintB : IntegrableOn (gapWeight H v i) (Set.Ioc (Real.sqrt d) T) :=
    hintR.mono_set fun u hu => hu.1
  have hintCDE : IntegrableOn (gapWeight H v i) (Set.Ioi T) :=
    hintR.mono_set fun u hu => lt_of_le_of_lt hTge hu
  have hintC : IntegrableOn (gapWeight H v i) (Set.Ioc T (π / 2 * T)) :=
    hintCDE.mono_set fun u hu => hu.1
  have hintDE : IntegrableOn (gapWeight H v i) (Set.Ioi (π / 2 * T)) :=
    hintCDE.mono_set fun u hu => lt_of_le_of_lt hb1 hu
  have hintD : IntegrableOn (gapWeight H v i) (Set.Ioc (π / 2 * T) (π * T)) :=
    hintDE.mono_set fun u hu => hu.1
  have hintF : IntegrableOn (gapWeight H v i) (Set.Ioi (π * T)) :=
    hintDE.mono_set fun u hu => lt_of_le_of_lt hb2 hu
  have hsplit : (∫ u in Set.Ioi (0:ℝ), gapWeight H v i u)
      = (∫ u in Set.Ioc (0:ℝ) (Real.sqrt d), gapWeight H v i u)
        + ((∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u)
          + ((∫ u in Set.Ioc T (π / 2 * T), gapWeight H v i u)
            + ((∫ u in Set.Ioc (π / 2 * T) (π * T), gapWeight H v i u)
              + ∫ u in Set.Ioi (π * T), gapWeight H v i u))) := by
    rw [Ioi_zero_split _ hsd,
      setIntegral_union (Ioc_disjoint_Ioi _ _) measurableSet_Ioi hintA hintR]
    congr 1
    rw [Ioi_split hTge, setIntegral_union (Ioc_disjoint_Ioi _ _) measurableSet_Ioi hintB hintCDE]
    congr 1
    rw [Ioi_split hb1, setIntegral_union (Ioc_disjoint_Ioi _ _) measurableSet_Ioi hintC hintDE]
    congr 1
    rw [Ioi_split hb2, setIntegral_union (Ioc_disjoint_Ioi _ _) measurableSet_Ioi hintD hintF]
  -- piece A, the low range, copied from `integral_gapWeight_le_sharp`
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
  -- piece B, the middle range, the one statement §66 replaces
  have hB : (∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u)
      ≤ (Real.exp (-(1/4)) + Real.sqrt π - 11 / 12) * S / Real.sqrt d := by
    have hTMe : T * M = Real.sqrt d := by rw [hTdef]; field_simp
    have hTM : T * M ≤ Real.sqrt d := le_of_eq hTMe
    exact integral_gapWeight_Ioc_le_mid hH hv i hM hd0 hTge hTM
  -- pieces C, D and F, the tail, copied verbatim from `integral_gapWeight_le_tailsharp`
  have hC : (∫ u in Set.Ioc T (π / 2 * T), gapWeight H v i u) ≤ 2 * E / T := by
    have hmaj : IntegrableOn (fun u : ℝ => 2 * E / u ^ 2) (Set.Ioi T) :=
      integrableOn_const_div_sq hTpos (2 * E)
    have hmajC : IntegrableOn (fun u : ℝ => 2 * E / u ^ 2) (Set.Ioc T (π / 2 * T)) :=
      hmaj.mono_set fun u hu => hu.1
    calc (∫ u in Set.Ioc T (π / 2 * T), gapWeight H v i u)
        ≤ ∫ u in Set.Ioc T (π / 2 * T), 2 * E / u ^ 2 :=
          setIntegral_mono_on hintC hmajC measurableSet_Ioc fun u hu =>
            gapWeight_le_tail_damped hH hv i hM hM0 hd0 hu.1.le hu.2
      _ ≤ ∫ u in Set.Ioi T, 2 * E / u ^ 2 := by
          refine setIntegral_mono_set hmaj ?_ (Filter.Eventually.of_forall fun u hu => hu.1)
          filter_upwards [ae_restrict_mem measurableSet_Ioi] with u hu
          have hu0 : 0 < u := lt_trans hTpos hu
          positivity
      _ = 2 * E / T := integral_Ioi_const_div_sq hTpos _
  have hD : (∫ u in Set.Ioc (π / 2 * T) (π * T), gapWeight H v i u)
      ≤ 4 * M ^ 2 / (π ^ 2 * (d:ℝ)) * (Real.sqrt (2 * π * (d:ℝ)) / 2 + E * (π * T / 2)) := by
    have hcont : Continuous fun u : ℝ =>
        4 * M ^ 2 / (π ^ 2 * (d:ℝ)) * (Real.exp (-(π * T - u) ^ 2 / (2 * (d:ℝ))) + E) := by
      fun_prop
    have hmaj : IntegrableOn (fun u : ℝ =>
        4 * M ^ 2 / (π ^ 2 * (d:ℝ)) * (Real.exp (-(π * T - u) ^ 2 / (2 * (d:ℝ))) + E))
        (Set.Ioc (π / 2 * T) (π * T)) := hcont.integrableOn_Ioc
    have hcont1 : Continuous fun u : ℝ => Real.exp (-(π * T - u) ^ 2 / (2 * (d:ℝ))) := by fun_prop
    have hint1 : IntegrableOn (fun u : ℝ => Real.exp (-(π * T - u) ^ 2 / (2 * (d:ℝ))))
        (Set.Ioc (π / 2 * T) (π * T)) := hcont1.integrableOn_Ioc
    have hint2 : IntegrableOn (fun _ : ℝ => E) (Set.Ioc (π / 2 * T) (π * T)) :=
      integrableOn_const (by simp)
    have hadd : (∫ u in Set.Ioc (π / 2 * T) (π * T),
          (Real.exp (-(π * T - u) ^ 2 / (2 * (d:ℝ))) + E))
        = (∫ u in Set.Ioc (π / 2 * T) (π * T), Real.exp (-(π * T - u) ^ 2 / (2 * (d:ℝ))))
          + E * (π * T / 2) := by
      rw [integral_add hint1 hint2, setIntegral_const,
        Real.volume_real_Ioc_of_le (by nlinarith), smul_eq_mul]
      ring
    have hgauss : (∫ u in Set.Ioc (π / 2 * T) (π * T),
        Real.exp (-(π * T - u) ^ 2 / (2 * (d:ℝ)))) ≤ Real.sqrt (2 * π * (d:ℝ)) / 2 :=
      integral_Ioc_exp_neg_sq_sub_le hd0 hb2
    calc (∫ u in Set.Ioc (π / 2 * T) (π * T), gapWeight H v i u)
        ≤ ∫ u in Set.Ioc (π / 2 * T) (π * T),
            4 * M ^ 2 / (π ^ 2 * (d:ℝ)) * (Real.exp (-(π * T - u) ^ 2 / (2 * (d:ℝ))) + E) :=
          setIntegral_mono_on hintD hmaj measurableSet_Ioc fun u hu =>
            gapWeight_le_tail_reflected hH hv i hM hM0 hd0 hu.1.le hu.2
      _ = 4 * M ^ 2 / (π ^ 2 * (d:ℝ))
            * ((∫ u in Set.Ioc (π / 2 * T) (π * T),
                Real.exp (-(π * T - u) ^ 2 / (2 * (d:ℝ)))) + E * (π * T / 2)) := by
          rw [integral_const_mul, hadd]
      _ ≤ 4 * M ^ 2 / (π ^ 2 * (d:ℝ)) * (Real.sqrt (2 * π * (d:ℝ)) / 2 + E * (π * T / 2)) := by
          have hc0 : (0:ℝ) ≤ 4 * M ^ 2 / (π ^ 2 * (d:ℝ)) := by positivity
          exact mul_le_mul_of_nonneg_left (by linarith) hc0
  have hF : (∫ u in Set.Ioi (π * T), gapWeight H v i u) ≤ (1 + E) / (π * T) := by
    have hmaj : IntegrableOn (fun u : ℝ => (1 + E) / u ^ 2) (Set.Ioi (π * T)) :=
      integrableOn_const_div_sq hpiT (1 + E)
    calc (∫ u in Set.Ioi (π * T), gapWeight H v i u)
        ≤ ∫ u in Set.Ioi (π * T), (1 + E) / u ^ 2 :=
          setIntegral_mono_on hintF hmaj measurableSet_Ioi fun u hu =>
            gapWeight_le_tail_beyond H v i hM0 hd0 hu
      _ = (1 + E) / (π * T) := integral_Ioi_const_div_sq hpiT _
  -- the arithmetic
  set G : ℝ := Real.sqrt (2 * π) with hGdef
  have hG0 : 0 ≤ G := Real.sqrt_nonneg _
  have hGval : Real.sqrt (2 * π * (d:ℝ)) = G * Real.sqrt d := by
    rw [hGdef, ← Real.sqrt_mul (by positivity)]
  have hABv : S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3)
      + (Real.exp (-(1/4)) + Real.sqrt π - 11 / 12) * S / Real.sqrt d
      = midConst * S / Real.sqrt d := by
    have h := arith_mid_low (S := S) (E := Real.exp (-(1/4)) + Real.sqrt π - 11 / 12) hsd
    rw [hsd2] at h
    rw [h, midConst]
    ring_nf
  have hCv : 2 * E / T = 2 * E * M / Real.sqrt d := by
    rw [hTdef]; field_simp
  have hDv : 4 * M ^ 2 / (π ^ 2 * (d:ℝ)) * (Real.sqrt (2 * π * (d:ℝ)) / 2 + E * (π * T / 2))
      = (2 * G / π ^ 2 * M ^ 2 + 2 * M * E / π) / Real.sqrt d := by
    rw [hGval, hTdef]
    exact arith_tail_refl hM0 hsd hsd2
  have hFv : (1 + E) / (π * T) = ((1 + E) * M / π) / Real.sqrt d := by
    rw [hTdef]; field_simp
  have hkey : 2 * E * M + (2 * G / π ^ 2 * M ^ 2 + 2 * M * E / π) + (1 + E) * M / π
      ≤ tailConst M * M := by
    rw [tailConst, ← hGdef]
    have hinv : 3 / π ≤ 1 := by rw [div_le_one hpi0]; linarith
    have hEM0 : (0:ℝ) ≤ E * M := by positivity
    have hEM5 : E * M ≤ 20 / 9 * M ^ 5 := by
      have h := mul_le_mul_of_nonneg_right hEb hM0.le
      calc E * M ≤ 20 / 9 * M ^ 4 * M := h
        _ = 20 / 9 * M ^ 5 := by ring
    have h1 : E * M * (2 + 3 / π) ≤ E * M * 3 := by nlinarith
    have hexpand : 2 * E * M + (2 * G / π ^ 2 * M ^ 2 + 2 * M * E / π) + (1 + E) * M / π
        = 1 / π * M + 2 * G / π ^ 2 * M ^ 2 + E * M * (2 + 3 / π) := by
      field_simp
      ring
    have hrhs : (1 / π + 2 * G / π ^ 2 * M + 16 * M ^ 4) * M
        = 1 / π * M + 2 * G / π ^ 2 * M ^ 2 + 16 * M ^ 5 := by ring
    rw [hexpand, hrhs]
    have hM5 : (0:ℝ) < M ^ 5 := pow_pos hM0 5
    linarith
  have htot : S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3)
      + ((Real.exp (-(1/4)) + Real.sqrt π - 11 / 12) * S / Real.sqrt d
        + (2 * E / T
          + (4 * M ^ 2 / (π ^ 2 * (d:ℝ)) * (Real.sqrt (2 * π * (d:ℝ)) / 2 + E * (π * T / 2))
            + (1 + E) / (π * T))))
      ≤ (midConst * S + tailConst M * M) / Real.sqrt d := by
    rw [hCv, hDv, hFv]
    have hleft : S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3)
        + ((Real.exp (-(1/4)) + Real.sqrt π - 11 / 12) * S / Real.sqrt d
          + (2 * E * M / Real.sqrt d
            + ((2 * G / π ^ 2 * M ^ 2 + 2 * M * E / π) / Real.sqrt d
              + ((1 + E) * M / π) / Real.sqrt d)))
        = (midConst * S
            + (2 * E * M + (2 * G / π ^ 2 * M ^ 2 + 2 * M * E / π) + (1 + E) * M / π))
            / Real.sqrt d := by
      have hstep : S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3)
          + ((Real.exp (-(1/4)) + Real.sqrt π - 11 / 12) * S / Real.sqrt d
            + (2 * E * M / Real.sqrt d
              + ((2 * G / π ^ 2 * M ^ 2 + 2 * M * E / π) / Real.sqrt d
                + ((1 + E) * M / π) / Real.sqrt d)))
          = (S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3)
              + (Real.exp (-(1/4)) + Real.sqrt π - 11 / 12) * S / Real.sqrt d)
            + (2 * E * M + (2 * G / π ^ 2 * M ^ 2 + 2 * M * E / π) + (1 + E) * M / π)
              / Real.sqrt d := by ring
      rw [hstep, hABv]
      ring
    rw [hleft, div_le_div_iff_of_pos_right hsd]
    linarith
  calc (∫ u in Set.Ioi (0:ℝ), gapWeight H v i u)
      = (∫ u in Set.Ioc (0:ℝ) (Real.sqrt d), gapWeight H v i u)
        + ((∫ u in Set.Ioc (Real.sqrt d) T, gapWeight H v i u)
          + ((∫ u in Set.Ioc T (π / 2 * T), gapWeight H v i u)
            + ((∫ u in Set.Ioc (π / 2 * T) (π * T), gapWeight H v i u)
              + ∫ u in Set.Ioi (π * T), gapWeight H v i u))) := hsplit
    _ ≤ S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3)
        + ((Real.exp (-(1/4)) + Real.sqrt π - 11 / 12) * S / Real.sqrt d
          + (2 * E / T
            + (4 * M ^ 2 / (π ^ 2 * (d:ℝ)) * (Real.sqrt (2 * π * (d:ℝ)) / 2 + E * (π * T / 2))
              + (1 + E) / (π * T)))) :=
        add_le_add hA (add_le_add hB (add_le_add hC (add_le_add hD hF)))
    _ ≤ (midConst * S + tailConst M * M) / Real.sqrt d := htot

end MidIntegral

end FourierGap

/-! ### §66e  Leg 1 and the deployed rate with the integrated damping

The substitution is mechanical: `NearFlat.abs_cdfGapInt_le_of_legBound` and
`Threshold.abs_hadRate_sub_rhoSq_le_of_legBound` take the two leg-1 constants as universally
quantified reals, so nothing below reproves anything. -/

section MidLeg

open Orthogonal Hadamard FourierGap

variable {d : ℕ}

namespace NearFlat

/-- **Leg 1 with the damping integrated.**  `abs_cdfGapInt_coordLaw_gaussian_le_tail` with
`integral_gapWeight_le_midsharp` in place of `integral_gapWeight_le_tailsharp`: for `d ≥ 2`, a flat
`H`, a unit direction `v` with `|vⱼ| ≤ M` and `0 < M ≤ 1`, any coordinate `i` and every pair
`a`, `x`,

  `|cdfGapInt (coordLaw H v i) (N(0,1/d)) a x| ≤ ((2/π)·A_mid·∑ⱼvⱼ⁴ + (2/π)·K(M)·M)/√d`,

against the `(3·∑ⱼvⱼ⁴ + (2/π)K(M)M)/√d` of §65 and the `(3·∑ⱼvⱼ⁴ + (13/10)M)/√d` of §60.  The
spikiness coefficient `(2/π)·A_mid = 1.1468` replaces `3`, a factor `2.56`; the `M`-linear term is
**identical** to §65's and is not improved.

**What has not changed.**  It is not a Kolmogorov-Smirnov bound, it is one leg only, the second
law is the Gaussian and not `sphericalCoordLaw d`, and the rate in `d` is the same.  The
`M`-linear term still dominates at a near-flat direction. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_mid (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M)
    (hM1 : M ≤ 1) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (2 / π * midConst * (∑ j, v j ^ 4) + 2 / π * tailConst M * M) / Real.sqrt d := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hpi0 : (0:ℝ) < π := Real.pi_pos
  have hcf : IntegrableOn (fun u : ℝ => ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2) (Set.Ioi 0) :=
    FourierGap.integrableOn_gapWeight hH hv i hM hM1 hd0
  have hI : (∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2)
      ≤ (midConst * (∑ j, v j ^ 4) + tailConst M * M) / Real.sqrt d :=
    FourierGap.integral_gapWeight_le_midsharp hH hv i hM hM0 hM1 hd0
  refine (FourierGap.abs_cdfGapInt_le_of_charFun (coordLaw H v i)
    (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ))))
    (FourierGap.integrable_abs_sub_coordLaw H v i)
    (FourierGap.integrable_abs_sub_gaussianReal _) hcf a x).trans ?_
  rw [FourierGap.dirConst_eq]
  set X := ∫ u in Set.Ioi (0:ℝ), ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ / u ^ 2 with hX
  have hdiv : X / (π / 2) = 2 / π * X := by field_simp
  rw [hdiv]
  calc 2 / π * X ≤ 2 / π * ((midConst * (∑ j, v j ^ 4) + tailConst M * M) / Real.sqrt d) :=
        mul_le_mul_of_nonneg_left hI (by positivity)
    _ = (2 / π * midConst * (∑ j, v j ^ 4) + 2 / π * tailConst M * M) / Real.sqrt d := by ring

end NearFlat

namespace Threshold

open Hadamard Orthogonal CoordLaw NearFlat

variable {K : ℕ}

/-- **The deployed rate at a near-flat direction, with the damping integrated.**  Same hypotheses
as `abs_hadRate_sub_rhoSq_le_of_nearFlat_tail`, whose spikiness coefficient `3` is replaced:

  `|hadRate H (blockStage q) v − ρ²|`
    `≤ 2·((2/π)A_mid·c²/√d + (2/π)K(c/√d)·c + 1 + 1/(4(d−1)))·spread cbk`

with `A_mid = e^{−1/4} + √π − 3/4` and `K` the tail constant of §65, unchanged.  At `d = 512` the
factor falls from `7.14` to `5.10` at `c = 3.53`, and the gain is largest where `c` is largest,
since the replaced term is the one carrying `c²`.

The caveats of `abs_hadRate_sub_rhoSq_le_of_nearFlat_tail` apply verbatim and none of them is
addressed here.  This is `Θ(spread cbk)` and not `o(1)`; `c` still bounds the **largest**
coordinate, so the near-flat class is still not of full measure on the sphere; `spread cbk` is
left abstract and no codebook is constructed.  `abs_hadRate_sub_rhoSq_le_of_nearFlat_tail` is
untouched and not weakened. -/
theorem abs_hadRate_sub_rhoSq_le_of_nearFlat_mid (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) {c : ℝ} (hc0 : 0 < c) (hcd : c ≤ Real.sqrt d)
    (hnf : ∀ j, |v j| ≤ c / Real.sqrt d) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ 2 * (2 / π * FourierGap.midConst * c ^ 2 / Real.sqrt d
          + 2 / π * FourierGap.tailConst (c / Real.sqrt d) * c + 1 + 1 / (4 * ((d:ℝ) - 1)))
          * spread cbk := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hpi0 : (0:ℝ) < π := Real.pi_pos
  have hM0 : 0 < c / Real.sqrt d := div_pos hc0 hsd
  have hM1 : c / Real.sqrt d ≤ 1 := by rw [div_le_one hsd]; exact hcd
  have hA0 : (0:ℝ) ≤ 2 / π * FourierGap.midConst := by
    have := FourierGap.midConst_pos
    positivity
  have hmain := abs_hadRate_sub_rhoSq_le_of_legBound hd hH v hv i cbk hq hqm ha hb hthr
    (A := 2 / π * FourierGap.midConst)
    (B := 2 / π * FourierGap.tailConst (c / Real.sqrt d)) (c := c)
    (fun x => NearFlat.abs_cdfGapInt_coordLaw_gaussian_le_mid hd hH hv i hnf hM0 hM1 a x)
  refine hmain.trans ?_
  have hS := sum_pow_four_le_sq_div_of_nearFlat hd hv hnf
  have hval : (c ^ 2 / (d:ℝ)) * Real.sqrt d = c ^ 2 / Real.sqrt d := by
    have hs2 : Real.sqrt (d:ℝ) * Real.sqrt (d:ℝ) = (d:ℝ) := Real.mul_self_sqrt hd0.le
    set s := Real.sqrt (d:ℝ) with hs
    have hs0 : s ≠ 0 := ne_of_gt hsd
    rw [← hs2]
    field_simp
  have hkey : 2 / π * FourierGap.midConst * (∑ j, v j ^ 4) * Real.sqrt d
      ≤ 2 / π * FourierGap.midConst * c ^ 2 / Real.sqrt d := by
    have h1 : (∑ j, v j ^ 4) * Real.sqrt d ≤ c ^ 2 / Real.sqrt d := by
      rw [← hval]; exact mul_le_mul_of_nonneg_right hS hsd.le
    calc 2 / π * FourierGap.midConst * (∑ j, v j ^ 4) * Real.sqrt d
        = 2 / π * FourierGap.midConst * ((∑ j, v j ^ 4) * Real.sqrt d) := by ring
      _ ≤ 2 / π * FourierGap.midConst * (c ^ 2 / Real.sqrt d) :=
          mul_le_mul_of_nonneg_left h1 hA0
      _ = 2 / π * FourierGap.midConst * c ^ 2 / Real.sqrt d := by ring
  exact mul_le_mul_of_nonneg_right (by linarith) (spread_nonneg cbk)

/-- **The composed bound at the near-flat radius a random direction has.**  At `d = 512`, for a
flat `H`, a unit direction near-flat with `c ≤ 3.53`, any ordered codebook whose thresholds lie in
a window around the support and whose **spread is at most `3.0164/√512`**, and any measurable
nearest-codeword map,

  `|hadRate H (blockStage q) v − ρ²| ≤ 69/100`.

`1` is the bound that holds for free whenever `0` is a codeword, and §65's
`abs_hadRate_sub_rhoSq_le_of_nearFlat_tail_512` reaches `24/25 = 0.96` at the same `c` and the
same spread.  The true value of this bound is `0.679`, and the chain of rational roundings
gives `0.682`; `69/100` is what `√512 ≥ 22.627`, `√(2π) ≤ 2.5067`, `√π ≤ 1.77246` and
`e^{−1/4} ≤ 250/321` support.

**The codebook width is a hypothesis, not a fact proved here.**  No codebook is constructed,
named or analysed, and nothing is claimed about any codebook the paper ships.  Nor is `1` the
target: `ρ² = 0.1171` is roughly six times smaller, and this route cannot reach it, since the
coupling leg alone contributes `2·spread cbk = 0.267` at this spread.  The near-flat hypothesis
is a hypothesis about `v`, and the statement says nothing about directions that fail it.
`abs_hadRate_sub_rhoSq_le_of_nearFlat_tail_512` is untouched and remains proved. -/
theorem abs_hadRate_sub_rhoSq_le_of_nearFlat_mid_512 {H : Od 512} (hH : IsFlat H)
    (v : Ed 512) (hv : ‖v‖ = 1) (i : Fin 512) {c : ℝ} (hc0 : 0 < c) (hc : c ≤ 3.53)
    (hnf : ∀ j, |v j| ≤ c / Real.sqrt 512) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hw : spread cbk ≤ 3.0164 / Real.sqrt 512) :
    |hadRate H (blockStage (m := 512) (F := ℝ) q) v
        - rhoSq 512 q (sphericalCoordLaw 512)| ≤ 69 / 100 := by
  have hs := FourierGap.sqrt_512_ge'
  have hsp : (0:ℝ) < Real.sqrt 512 := by linarith
  have hpi0 : (0:ℝ) < π := Real.pi_pos
  have hpi : (3.141592:ℝ) < π := Real.pi_gt_d6
  have hcast : Real.sqrt ((512:ℕ):ℝ) = Real.sqrt 512 := by norm_num
  have hnf' : ∀ j, |v j| ≤ c / Real.sqrt ((512:ℕ):ℝ) := by simpa [hcast] using hnf
  have hmain := abs_hadRate_sub_rhoSq_le_of_nearFlat_mid (d := 512) (by norm_num) hH v hv i hc0
    (by rw [hcast]; linarith) hnf' cbk hq hqm ha hb hthr
  rw [hcast] at hmain
  push_cast at hmain
  have hsp0 : 0 ≤ spread cbk := spread_nonneg cbk
  -- the tail constant, copied from §65
  have hM0 : 0 ≤ c / Real.sqrt 512 := by positivity
  have hMb : c / Real.sqrt 512 ≤ 1561 / 10000 := by
    rw [div_le_iff₀ hsp]
    nlinarith
  have hKnn : (0:ℝ) ≤ FourierGap.tailConst (c / Real.sqrt 512) := by
    rw [FourierGap.tailConst]; positivity
  have hK : FourierGap.tailConst (c / Real.sqrt 512) ≤ 409 / 1000 := by
    refine (FourierGap.tailConst_le_explicit hM0).trans ?_
    have h4 : (c / Real.sqrt 512) ^ 4 ≤ 1 / 1650 := by
      have h := pow_le_pow_left₀ hM0 hMb 4
      have hnum : ((1561:ℝ) / 10000) ^ 4 ≤ 1 / 1650 := by norm_num
      linarith
    linarith
  have hB : 2 / π * FourierGap.tailConst (c / Real.sqrt 512) ≤ 2604 / 10000 := by
    have h1 : 2 / π ≤ 2 / 3.141592 := by
      rw [div_le_div_iff₀ hpi0 (by norm_num)]; linarith
    calc 2 / π * FourierGap.tailConst (c / Real.sqrt 512)
        ≤ 2 / 3.141592 * FourierGap.tailConst (c / Real.sqrt 512) :=
          mul_le_mul_of_nonneg_right h1 hKnn
      _ ≤ 2 / 3.141592 * (409 / 1000) := mul_le_mul_of_nonneg_left hK (by norm_num)
      _ ≤ 2604 / 10000 := by norm_num
  -- the spikiness constant, new in §66
  have hAnn : (0:ℝ) ≤ FourierGap.midConst := le_of_lt FourierGap.midConst_pos
  have hA : 2 / π * FourierGap.midConst ≤ 11468 / 10000 := by
    have h1 : 2 / π ≤ 2 / 3.141592 := by
      rw [div_le_div_iff₀ hpi0 (by norm_num)]; linarith
    calc 2 / π * FourierGap.midConst
        ≤ 2 / 3.141592 * FourierGap.midConst := mul_le_mul_of_nonneg_right h1 hAnn
      _ ≤ 2 / 3.141592 * (18013 / 10000) :=
          mul_le_mul_of_nonneg_left FourierGap.midConst_le (by norm_num)
      _ ≤ 11468 / 10000 := by norm_num
  have hfac : 2 * (2 / π * FourierGap.midConst * c ^ 2 / Real.sqrt 512
      + 2 / π * FourierGap.tailConst (c / Real.sqrt 512) * c + 1 + 1 / (4 * ((512:ℝ) - 1)))
      ≤ 511 / 100 := by
    have hc2 : c ^ 2 ≤ 3.53 ^ 2 := by nlinarith
    have hA0 : (0:ℝ) ≤ 2 / π * FourierGap.midConst := by positivity
    have h1 : 2 / π * FourierGap.midConst * c ^ 2 / Real.sqrt 512
        ≤ 11468 / 10000 * (3.53 ^ 2) / 22.627 := by
      rw [div_le_div_iff₀ hsp (by norm_num)]
      nlinarith [hs, hc2, hA0, hA]
    have hBnn : (0:ℝ) ≤ 2 / π * FourierGap.tailConst (c / Real.sqrt 512) := by positivity
    have h2 : 2 / π * FourierGap.tailConst (c / Real.sqrt 512) * c ≤ 2604 / 10000 * 3.53 := by
      nlinarith
    norm_num at h1 h2 ⊢
    linarith
  have hstep : |hadRate H (blockStage (m := 512) (F := ℝ) q) v
      - rhoSq 512 q (sphericalCoordLaw 512)| ≤ 511 / 100 * (3.0164 / Real.sqrt 512) := by
    refine hmain.trans ((mul_le_mul_of_nonneg_right hfac hsp0).trans ?_)
    exact mul_le_mul_of_nonneg_left hw (by norm_num)
  refine hstep.trans ?_
  have heq : (511:ℝ) / 100 * (3.0164 / Real.sqrt 512)
      = (511 * 3.0164) / (100 * Real.sqrt 512) := by ring
  rw [heq, div_le_iff₀ (by positivity)]
  nlinarith [hs]

end Threshold

end MidLeg

end MidRange
