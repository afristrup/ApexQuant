
/-! ## §65  The tail constant: the trivial `‖φ_μ − φ_ν‖ ≤ 2` replaced by a periodic bound

`FourierGap.gapWeight_le_high` charges `‖φ_μ − φ_ν‖ ≤ 2` on the whole range `u > √d/M`, and that
one estimate is where the `2M` of `integral_gapWeight_le_sharp`, the `13/10` of
`abs_cdfGapInt_coordLaw_gaussian_le_of_bound_best` and the `(13/10)·c` of
`Threshold.abs_hadRate_sub_rhoSq_le_of_nearFlat` all come from.  It is the dominant term at a
near-flat direction and the reason §60's bound sits above the free bound `1` at the group size
the paper ships.

**What is replaced, and by what.**  Write `T = √d/M` and `s = u/T`.  On `T < u ≤ πT` the atomic
characteristic function is *not* uncontrolled: it is `∏ⱼ cos(u·Hᵢⱼvⱼ)`, every factor is a cosine
of an argument `bⱼ·s` with `bⱼ = |vⱼ|/M ∈ [0,1]`, and `abs_cos_mul_le` bounds each factor by
`exp(−bⱼ²·min(s, π−s)²/2)`.  Since `∑ⱼ bⱼ² = 1/M²` at a unit direction, the product is damped by
`exp(−min(s, π−s)²/(2M²))` for the whole range, which is tiny except in a window of width `Θ(M)`
around `s = π`, where the product genuinely returns to modulus one.  Only past `πT` is a trivial
bound needed, and there the Gaussian factor is already negligible, so `2` can be replaced by `1`
plus an exponentially small remainder.  That is the entire gain, and it is a gain of a factor
`2π = 6.28` on the leading piece and nothing at all on the rate in `d`.

**The constant.**  `integral_gapWeight_le_tailsharp` reads

  `∫₀^∞ ‖φ_μ − φ_ν‖/u² du ≤ (83/18·∑ⱼvⱼ⁴ + K(M)·M)/√d`,   `K(M) = 1/π + (2√(2π)/π²)·M + 16M⁴`,

against the `2M` of `integral_gapWeight_le_sharp`, whose spikiness term `83/18` is copied
verbatim and is **not** improved here.  `1/π` is the floor of `K`: it is the mass of `∫ du/u²`
beyond `πT` and no bound of the form `‖φ_μ‖ ≤ 1` there can do better.  The `M`-linear term is the
reflected Gaussian window at `s = π`, and `16M⁴` absorbs every exponentially small remainder
(`tailConst_le_explicit` and `tailConst_le_fifth` evaluate the whole of `K`).

**What is *not* done.**  The `M`-linear term is kept in the `B` slot of
`NearFlat.abs_cdfGapInt_le_of_legBound` and is **not** absorbed into the `∑ⱼvⱼ⁴` coefficient:
that absorption would need `M² ≤ ∑ⱼvⱼ⁴`, and the true direction is `∑ⱼvⱼ⁴ ≤ M²`.  Nothing here
is asymptotic, no new law is analysed, and the rate in `d` is exactly as it was: the composed
near-flat bound is still `Θ(spread cbk)` and still vanishes only if the codebook is placed at the
coordinate scale, which remains a hypothesis about `cbk` and is proved for no codebook anywhere
in this development.  `integral_gapWeight_le_sharp`, `gapWeight_le_high` and every §60 statement
are kept untouched and are not weakened.

**Numerically, at the dimension the paper ships.**  At `d = 512` and `c = 3.53`, so
`M = c/√d = 0.156`, `K = 0.407` against `2`, and the composed factor
`2·(3c²/√d + (2/π)K·c + 1 + 1/(4(d−1)))` falls from `14.48` to `7.14`.  Against a codebook of
spread at most `3.0164/√512` that is `0.95`, below the free bound `1` that holds whenever `0` is
a codeword; `Threshold.abs_hadRate_sub_rhoSq_le_of_nearFlat_tail_512` records it.  `c = 3.53` is
about `√(2 log 512)`, the scale at which a Haar-random coordinate maximum concentrates, so this
is the first statement in this development that beats the free bound at the near-flat radius a
random direction actually has.  It is still far from `ρ² = 0.1171`, and it cannot reach it on
this route: the coupling leg alone contributes `2·spread cbk = 0.267` at that spread.  The
codebook width is a hypothesis; no codebook is constructed or analysed here. -/

noncomputable section TailConst

open Orthogonal Hadamard CoordLaw

namespace FourierGap

/-! ### §65a  The periodic cosine bound and its product form

Nothing in this block mentions a rotation, a dimension or a law: it is three inequalities about
`sin`, `cos` and a finite product of cosines. -/

/-- `sin t ^ 2 ≥ t² − t⁴/3` for `|t| ≤ 1`, from `sin t ≥ t − t³/6`.  It is used only at
`|t| ≤ 1`, where the quartic remainder has the right sign; nothing is claimed outside. -/
theorem sin_sq_ge {t : ℝ} (ht : |t| ≤ 1) : t ^ 2 - t ^ 4 / 3 ≤ Real.sin t ^ 2 := by
  have key : ∀ s : ℝ, 0 < s → s ≤ 1 → s ^ 2 - s ^ 4 / 3 ≤ Real.sin s ^ 2 := by
    intro s hs hs1
    have h := (Real.sin_gt_sub_cube hs).le
    have hcube : s ^ 3 ≤ s := by
      nlinarith [mul_nonneg (mul_nonneg hs.le (sub_nonneg.mpr hs1))
        (by linarith : (0:ℝ) ≤ 1 + s)]
    have hnn : 0 ≤ s - s ^ 3 / 6 := by nlinarith
    have : (s - s ^ 3 / 6) ^ 2 ≤ Real.sin s ^ 2 := by nlinarith
    nlinarith [sq_nonneg s, pow_nonneg hs.le 6]
  rcases lt_trichotomy t 0 with h | h | h
  · have h1 : 0 < -t := by linarith
    have h2 : -t ≤ 1 := by cases abs_le.mp ht; linarith
    have := key (-t) h1 h2
    rw [Real.sin_neg] at this; nlinarith [this]
  · simp [h]
  · exact key t h (by cases abs_le.mp ht; linarith)

/-- `cos x ≤ e^{−x²/2}` on `|x| ≤ 2`.  The hypothesis is needed: the two sides cross again past
`|x| = 2`, and `Real.cos_bound` expires at `|x| ≤ 1`, which is exactly where the estimate of §58b
has to stop. -/
theorem cos_le_exp_neg_half_sq {x : ℝ} (hx : |x| ≤ 2) : Real.cos x ≤ Real.exp (-(x ^ 2) / 2) := by
  have hx2 : x ^ 2 ≤ 4 := by nlinarith [abs_nonneg x, sq_abs x]
  have hc : Real.cos x = 1 - 2 * Real.sin (x / 2) ^ 2 := by
    have h2 : Real.cos (2 * (x / 2)) = 1 - 2 * Real.sin (x / 2) ^ 2 := by
      rw [Real.cos_two_mul']; nlinarith [Real.sin_sq_add_cos_sq (x / 2)]
    rw [show (2 : ℝ) * (x / 2) = x by ring] at h2; exact h2
  have hthalf : |x / 2| ≤ 1 := by
    have he : |x / 2| = |x| / 2 := by rw [abs_div]; norm_num
    rw [he]; linarith
  have hs := sin_sq_ge hthalf
  have h1 : Real.cos x ≤ 1 - x ^ 2 / 2 + x ^ 4 / 24 := by rw [hc]; nlinarith [hs]
  have h2 : (1 - x ^ 2 / 4) ^ 2 ≤ Real.exp (-(x ^ 2) / 2) := by
    have hL : 1 - x ^ 2 / 4 ≤ Real.exp (-(x ^ 2) / 4) := by
      have := Real.add_one_le_exp (-(x ^ 2) / 4); linarith
    have hnn : 0 ≤ 1 - x ^ 2 / 4 := by nlinarith
    have hsq : (1 - x ^ 2 / 4) ^ 2 ≤ Real.exp (-(x ^ 2) / 4) ^ 2 := by
      nlinarith [Real.exp_pos (-(x ^ 2) / 4)]
    calc (1 - x ^ 2 / 4) ^ 2 ≤ Real.exp (-(x ^ 2) / 4) ^ 2 := hsq
      _ = Real.exp (-(x ^ 2) / 2) := by rw [sq, ← Real.exp_add]; ring_nf
  nlinarith [h1, h2, pow_nonneg (sq_nonneg x) 2]

/-- **The tail's pointwise handle.**  For `s ∈ [1, π]` and `0 ≤ b ≤ 1`,
`|cos (b·s)| ≤ exp(−b²·min(s, π−s)²/2)`.  The reflection is the whole point: at `s` near `π` the
bound degrades to `1`, as it must, since `cos(b·π)` is `±1` at `b = 1`. -/
theorem abs_cos_mul_le {s b : ℝ} (hs1 : 1 ≤ s) (hsp : s ≤ π) (hb0 : 0 ≤ b) (hb1 : b ≤ 1) :
    |Real.cos (b * s)| ≤ Real.exp (-(b ^ 2 * min s (π - s) ^ 2) / 2) := by
  set m : ℝ := min s (π - s) with hm
  have hm0 : 0 ≤ m := le_min (by linarith) (by linarith)
  have hmpi : m ≤ π / 2 := by
    rcases le_total s (π - s) with h | h
    · rw [hm, min_eq_left h]; linarith
    · rw [hm, min_eq_right h]; linarith
  set x : ℝ := b * s with hx
  have hx0 : 0 ≤ x := by positivity
  have hxpi : x ≤ π := by nlinarith [Real.pi_pos]
  set r : ℝ := min x (π - x) with hr
  have hr0 : 0 ≤ r := le_min hx0 (by linarith)
  have hrpi : r ≤ π / 2 := by
    rcases le_total x (π - x) with h | h
    · rw [hr, min_eq_left h]; linarith
    · rw [hr, min_eq_right h]; linarith
  have habs : |Real.cos x| = Real.cos r := by
    rcases le_total x (π / 2) with h | h
    · have : r = x := by
        rw [hr]; exact min_eq_left (by linarith)
      rw [this, abs_of_nonneg (Real.cos_nonneg_of_mem_Icc ⟨by linarith [Real.pi_pos], h⟩)]
    · have : r = π - x := by rw [hr]; exact min_eq_right (by linarith)
      rw [this, Real.cos_pi_sub, abs_of_nonpos
        (Real.cos_nonpos_of_pi_div_two_le_of_le h (by linarith [Real.pi_pos]))]
  have hbm : b * m ≤ r := by
    rw [hr, le_min_iff]
    constructor
    · rw [hx]; rcases le_total s (π - s) with h | h
      · rw [hm, min_eq_left h]
      · rw [hm, min_eq_right h]; nlinarith
    · rcases le_total s (π - s) with h | h
      · rw [hm, min_eq_left h, hx]; nlinarith [Real.pi_pos]
      · rw [hm, min_eq_right h, hx]; nlinarith [Real.pi_pos]
  have hbm0 : 0 ≤ b * m := by positivity
  have hstep1 : Real.cos r ≤ Real.cos (b * m) :=
    Real.cos_le_cos_of_nonneg_of_le_pi hbm0 (by linarith [Real.pi_pos]) hbm
  have hstep2 : Real.cos (b * m) ≤ Real.exp (-((b * m) ^ 2) / 2) := by
    refine cos_le_exp_neg_half_sq ?_
    rw [abs_of_nonneg hbm0]; nlinarith [Real.pi_le_four]
  rw [habs]
  calc Real.cos r ≤ Real.cos (b * m) := hstep1
    _ ≤ Real.exp (-((b * m) ^ 2) / 2) := hstep2
    _ = Real.exp (-(b ^ 2 * m ^ 2) / 2) := by ring_nf

/-- **The product form.**  For coefficients `bⱼ ∈ [0,1]` and `s ∈ [1, π]`,

  `∏ⱼ |cos(bⱼ·s)| ≤ exp(−(∑ⱼbⱼ²)·min(s, π−s)²/2)`,

which is `abs_cos_mul_le` termwise and `Real.exp_sum`.  The damping is governed by `∑ⱼbⱼ²` alone,
so a single coefficient near `1` does not spoil it; what does spoil it is `s` near `π`, where
`min(s, π−s)` vanishes and the bound is vacuous. -/
theorem prod_abs_cos_le {n : ℕ} {b : Fin n → ℝ} (hb0 : ∀ j, 0 ≤ b j) (hb1 : ∀ j, b j ≤ 1)
    {s : ℝ} (hs1 : 1 ≤ s) (hsp : s ≤ π) :
    (∏ j, |Real.cos (b j * s)|)
      ≤ Real.exp (-((∑ j, b j ^ 2) * min s (π - s) ^ 2) / 2) := by
  set m : ℝ := min s (π - s) with hm
  calc (∏ j, |Real.cos (b j * s)|)
      ≤ ∏ j, Real.exp (-(b j ^ 2 * m ^ 2) / 2) :=
        Finset.prod_le_prod (fun j _ => abs_nonneg _)
          (fun j _ => abs_cos_mul_le hs1 hsp (hb0 j) (hb1 j))
    _ = Real.exp (∑ j, -(b j ^ 2 * m ^ 2) / 2) := (Real.exp_sum _ _).symm
    _ = Real.exp (-((∑ j, b j ^ 2) * m ^ 2) / 2) := by
        congr 1
        have h1 : ∀ j : Fin n, -(b j ^ 2 * m ^ 2) / 2 = b j ^ 2 * (-(m ^ 2) / 2) :=
          fun j => by ring
        rw [Finset.sum_congr rfl (fun j _ => h1 j), ← Finset.sum_mul]
        ring

/-- The half-line Gaussian mass, in the shifted form the reflected window needs:
`∫_{(a,b]} e^{−(b−u)²/(2R)} du ≤ √(2πR)/2` for `a ≤ b`.  It is `integral_gaussian_Ioi`, which is
unconditional, after the reflection `w = b − u`; the left endpoint is thrown away. -/
theorem integral_Ioc_exp_neg_sq_sub_le {R a b : ℝ} (hR : 0 < R) (hab : a ≤ b) :
    (∫ u in Set.Ioc a b, Real.exp (-(b - u) ^ 2 / (2 * R)))
      ≤ Real.sqrt (2 * π * R) / 2 := by
  have hbinv : (0:ℝ) < 1 / (2 * R) := by positivity
  have hcongr : ∀ w : ℝ, Real.exp (-w ^ 2 / (2 * R)) = Real.exp (-(1 / (2 * R)) * w ^ 2) := by
    intro w; congr 1; field_simp
  have hint : IntegrableOn (fun w : ℝ => Real.exp (-w ^ 2 / (2 * R))) (Set.Ioi 0) := by
    have h := (integrable_exp_neg_mul_sq hbinv).integrableOn (s := Set.Ioi (0:ℝ))
    exact h.congr_fun (fun w _ => (hcongr w).symm) measurableSet_Ioi
  have hstep : (∫ u in Set.Ioc a b, Real.exp (-(b - u) ^ 2 / (2 * R)))
      = ∫ w in Set.Ioc (0:ℝ) (b - a), Real.exp (-w ^ 2 / (2 * R)) := by
    rw [← intervalIntegral.integral_of_le hab, ← intervalIntegral.integral_of_le (by linarith),
      intervalIntegral.integral_comp_sub_left (fun w : ℝ => Real.exp (-w ^ 2 / (2 * R))) b]
    simp
  rw [hstep]
  calc (∫ w in Set.Ioc (0:ℝ) (b - a), Real.exp (-w ^ 2 / (2 * R)))
      ≤ ∫ w in Set.Ioi (0:ℝ), Real.exp (-w ^ 2 / (2 * R)) := by
        refine setIntegral_mono_set hint ?_ (Filter.Eventually.of_forall fun w hw => hw.1)
        filter_upwards with w using (Real.exp_pos _).le
    _ = Real.sqrt (2 * π * R) / 2 := by
        rw [setIntegral_congr_fun measurableSet_Ioi (fun w _ => hcongr w), integral_gaussian_Ioi]
        congr 2
        field_simp

/-- `Ioc a b` and `Ioi b` are disjoint, the bookkeeping the four-way split of `Ioi 0` repeats. -/
theorem Ioc_disjoint_Ioi (a b : ℝ) : Disjoint (Set.Ioc a b) (Set.Ioi b) := by
  rw [Set.disjoint_left]
  intro u hu hu'
  exact absurd hu' (not_lt.mpr hu.2)

/-! ### §65b  The deployed tail bound

From here the pair is fixed, as in §58b: `coordLaw H v i` against `N(0, 1/d)` at a **unit**
direction `v` with `|vⱼ| ≤ M`. -/

section TailDeployed

variable {d : ℕ}

/-- **The atomic characteristic function on the whole tail range.**  For a flat `H`, a unit
direction with `|vⱼ| ≤ M`, and `√d/M ≤ u ≤ π·√d/M`, writing `s = u·M/√d ∈ [1, π]`,

  `‖φ_{coordLaw H v i}(u)‖ ≤ exp(−min(s, π−s)²/(2M²))`.

This is the statement that replaces `‖φ_μ‖ ≤ 1` on that range.  It says nothing beyond `u > π√d/M`,
where `s` leaves `[1, π]` and the product returns to modulus one; the periodicity is real and is
not an artefact of the estimate. -/
theorem norm_charFun_coordLaw_le_tail {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1)
    (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M) (hd0 : (0:ℝ) < (d:ℝ)) {u : ℝ}
    (hu1 : Real.sqrt d / M ≤ u) (hu2 : u ≤ π * (Real.sqrt d / M)) :
    ‖charFun (coordLaw H v i) u‖
      ≤ Real.exp (-(min (u * M / Real.sqrt d) (π - u * M / Real.sqrt d)) ^ 2 / (2 * M ^ 2)) := by
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hT0 : 0 < Real.sqrt d / M := div_pos hsd hM0
  have hu0 : 0 < u := lt_of_lt_of_le hT0 hu1
  set s : ℝ := u * M / Real.sqrt d with hsdef
  set b : Fin d → ℝ := fun j => |v j| / M with hbdef
  have hs1 : 1 ≤ s := by
    have h := (div_le_iff₀ hM0).mp hu1
    rw [hsdef, le_div_iff₀ hsd, one_mul]
    exact h
  have hsp : s ≤ π := by
    rw [hsdef, div_le_iff₀ hsd]
    have : u * M ≤ π * (Real.sqrt d / M) * M := by nlinarith
    calc u * M ≤ π * (Real.sqrt d / M) * M := this
      _ = π * Real.sqrt d := by field_simp
  have hb0 : ∀ j, 0 ≤ b j := fun j => by rw [hbdef]; positivity
  have hb1 : ∀ j, b j ≤ 1 := fun j => by
    rw [hbdef]; simp only; rw [div_le_one hM0]; exact hM j
  have hbsum : (∑ j, b j ^ 2) = 1 / M ^ 2 := by
    have hvs : ∑ j, v j ^ 2 = 1 := by
      have h := norm_sq_eq_sum_sq v
      rw [hv] at h
      simpa using h.symm
    have hterm : ∀ j : Fin d, b j ^ 2 = v j ^ 2 * (1 / M ^ 2) := by
      intro j
      rw [hbdef]; simp only
      rw [div_pow, sq_abs]
      field_simp
    rw [Finset.sum_congr rfl (fun j _ => hterm j), ← Finset.sum_mul, hvs, one_mul]
  have hfac : ∀ j : Fin d,
      |Real.cos (u * ((H : Matrix (Fin d) (Fin d) ℝ) i j * v j))| = |Real.cos (b j * s)| := by
    intro j
    have habs : |u * ((H : Matrix (Fin d) (Fin d) ℝ) i j * v j)| = b j * s := by
      rw [abs_mul, abs_mul, abs_entry_of_isFlat hH, abs_of_pos hu0, hbdef, hsdef]
      simp only
      field_simp
    have hcos : Real.cos (b j * s)
        = Real.cos (u * ((H : Matrix (Fin d) (Fin d) ℝ) i j * v j)) := by
      rw [← habs, Real.cos_abs]
    rw [hcos]
  rw [charFun_coordLaw, Complex.norm_real, Real.norm_eq_abs, Finset.abs_prod,
    Finset.prod_congr rfl (fun j _ => hfac j)]
  refine (prod_abs_cos_le hb0 hb1 hs1 hsp).trans (le_of_eq ?_)
  rw [hbsum]
  congr 1
  field_simp

/-- **The Gaussian side on the tail range.**  `‖φ_{N(0,1/d)}(u)‖ = e^{−u²/(2d)} ≤ e^{−1/(2M²)}`
for `u ≥ √d/M`.  Beyond that point the Gaussian is already negligible, which is what lets the
trivial bound `2` be replaced by `1` plus a remainder. -/
theorem norm_charFun_gaussian_le_tail {M : ℝ} (hM0 : 0 < M) (hd0 : (0:ℝ) < (d:ℝ)) {u : ℝ}
    (hu1 : Real.sqrt d / M ≤ u) :
    ‖charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖
      ≤ Real.exp (-(1 / (2 * M ^ 2))) := by
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hT0 : 0 < Real.sqrt d / M := div_pos hsd hM0
  have hu0 : 0 < u := lt_of_lt_of_le hT0 hu1
  have hcoe : ((Real.toNNReal (1 / (d:ℝ)) : NNReal) : ℝ) = 1 / (d:ℝ) :=
    Real.coe_toNNReal _ (by positivity)
  rw [norm_charFun_gaussianReal, hcoe]
  refine Real.exp_le_exp.mpr ?_
  have hsq : (Real.sqrt d / M) ^ 2 ≤ u ^ 2 := pow_le_pow_left₀ hT0.le hu1 2
  have hval : (Real.sqrt d / M) ^ 2 = (d:ℝ) / M ^ 2 := by
    rw [div_pow, Real.sq_sqrt hd0.le]
  rw [hval] at hsq
  have hmono : 1 / (d:ℝ) * ((d:ℝ) / M ^ 2) / 2 ≤ 1 / (d:ℝ) * u ^ 2 / 2 := by gcongr
  have heq : 1 / (d:ℝ) * ((d:ℝ) / M ^ 2) / 2 = 1 / (2 * M ^ 2) := by
    field_simp
  rw [heq] at hmono
  linarith

/-- **Tail range, damped half.**  For `√d/M ≤ u ≤ (π/2)·√d/M` both characteristic functions are
below `e^{−1/(2M²)}`, so `gapWeight ≤ 2e^{−1/(2M²)}/u²`.  On this half the bound is exponentially
small in `1/M²` and contributes nothing beyond the `16M⁴` of `tailConst`. -/
theorem gapWeight_le_tail_damped {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1)
    (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M) (hd0 : (0:ℝ) < (d:ℝ)) {u : ℝ}
    (hu1 : Real.sqrt d / M ≤ u) (hu2 : u ≤ π / 2 * (Real.sqrt d / M)) :
    gapWeight H v i u ≤ 2 * Real.exp (-(1 / (2 * M ^ 2))) / u ^ 2 := by
  have hpi : (3:ℝ) < π := Real.pi_gt_three
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hT0 : 0 < Real.sqrt d / M := div_pos hsd hM0
  have hu0 : 0 < u := lt_of_lt_of_le hT0 hu1
  have hu2' : u ≤ π * (Real.sqrt d / M) := by nlinarith
  set s : ℝ := u * M / Real.sqrt d with hsdef
  have hs1 : 1 ≤ s := by
    have h := (div_le_iff₀ hM0).mp hu1
    rw [hsdef, le_div_iff₀ hsd, one_mul]
    exact h
  have hshalf : s ≤ π / 2 := by
    rw [hsdef, div_le_iff₀ hsd]
    have h : u * M ≤ π / 2 * (Real.sqrt d / M) * M := by nlinarith
    calc u * M ≤ π / 2 * (Real.sqrt d / M) * M := h
      _ = π / 2 * Real.sqrt d := by field_simp
  have hmin : min s (π - s) = s := min_eq_left (by linarith)
  have hmu := norm_charFun_coordLaw_le_tail hH hv i hM hM0 hd0 hu1 hu2'
  rw [← hsdef, hmin] at hmu
  have hmu' : ‖charFun (coordLaw H v i) u‖ ≤ Real.exp (-(1 / (2 * M ^ 2))) := by
    refine hmu.trans (Real.exp_le_exp.mpr ?_)
    have h1 : (1:ℝ) ≤ s ^ 2 := by nlinarith
    have h2 : (0:ℝ) < 2 * M ^ 2 := by positivity
    have hle : 1 / (2 * M ^ 2) ≤ s ^ 2 / (2 * M ^ 2) := by gcongr
    rw [neg_div]
    linarith
  have hnu := norm_charFun_gaussian_le_tail (d := d) hM0 hd0 hu1
  rw [gapWeight, div_le_div_iff_of_pos_right (pow_pos hu0 2)]
  calc ‖charFun (coordLaw H v i) u - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖
      ≤ ‖charFun (coordLaw H v i) u‖
        + ‖charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ := norm_sub_le _ _
    _ ≤ Real.exp (-(1 / (2 * M ^ 2))) + Real.exp (-(1 / (2 * M ^ 2))) := add_le_add hmu' hnu
    _ = 2 * Real.exp (-(1 / (2 * M ^ 2))) := by ring

/-- **Tail range, reflected half.**  For `(π/2)·√d/M ≤ u ≤ π·√d/M` the damping is governed by the
reflection `π − s`, which in `u` is the Gaussian window `e^{−(πT − u)²/(2d)}` centred at the
period `u = πT`, and `1/u² ≤ 4M²/(π²d)` there.  This half is the only one that contributes at
order `M`, and it is where the `2√(2π)/π²` of `tailConst` comes from. -/
theorem gapWeight_le_tail_reflected {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1)
    (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M) (hd0 : (0:ℝ) < (d:ℝ)) {u : ℝ}
    (hu1 : π / 2 * (Real.sqrt d / M) ≤ u) (hu2 : u ≤ π * (Real.sqrt d / M)) :
    gapWeight H v i u
      ≤ 4 * M ^ 2 / (π ^ 2 * (d:ℝ))
          * (Real.exp (-(π * (Real.sqrt d / M) - u) ^ 2 / (2 * (d:ℝ)))
            + Real.exp (-(1 / (2 * M ^ 2)))) := by
  have hpi : (3:ℝ) < π := Real.pi_gt_three
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hT0 : 0 < Real.sqrt d / M := div_pos hsd hM0
  have hhalf : 0 < π / 2 * (Real.sqrt d / M) := by positivity
  have hu0 : 0 < u := lt_of_lt_of_le hhalf hu1
  have hu1' : Real.sqrt d / M ≤ u := by nlinarith
  set s : ℝ := u * M / Real.sqrt d with hsdef
  have hshalf : π / 2 ≤ s := by
    rw [hsdef, le_div_iff₀ hsd]
    have h : π / 2 * (Real.sqrt d / M) * M ≤ u * M := by nlinarith
    calc π / 2 * Real.sqrt d = π / 2 * (Real.sqrt d / M) * M := by field_simp
      _ ≤ u * M := h
  have hmin : min s (π - s) = π - s := min_eq_right (by linarith)
  have hmu := norm_charFun_coordLaw_le_tail hH hv i hM hM0 hd0 hu1' hu2
  rw [← hsdef, hmin] at hmu
  have hrefl : (π - s) ^ 2 / (2 * M ^ 2) = (π * (Real.sqrt d / M) - u) ^ 2 / (2 * (d:ℝ)) := by
    have hkey : π - s = (π * (Real.sqrt d / M) - u) * M / Real.sqrt d := by
      rw [hsdef]; field_simp
    rw [hkey, div_pow, mul_pow, Real.sq_sqrt hd0.le]
    field_simp
  have hmu' : ‖charFun (coordLaw H v i) u‖
      ≤ Real.exp (-(π * (Real.sqrt d / M) - u) ^ 2 / (2 * (d:ℝ))) := by
    refine hmu.trans (le_of_eq ?_)
    rw [neg_div, neg_div, hrefl]
  have hnu := norm_charFun_gaussian_le_tail (d := d) hM0 hd0 hu1'
  have hinv : (u ^ 2)⁻¹ ≤ 4 * M ^ 2 / (π ^ 2 * (d:ℝ)) := by
    have hlow : π ^ 2 * (d:ℝ) / (4 * M ^ 2) ≤ u ^ 2 := by
      have h := pow_le_pow_left₀ hhalf.le hu1 2
      have hval : (π / 2 * (Real.sqrt d / M)) ^ 2 = π ^ 2 * (d:ℝ) / (4 * M ^ 2) := by
        rw [mul_pow, div_pow, div_pow, Real.sq_sqrt hd0.le]
        ring
      rw [hval] at h
      exact h
    have hpos : (0:ℝ) < π ^ 2 * (d:ℝ) / (4 * M ^ 2) := by positivity
    have := inv_anti₀ hpos hlow
    rwa [inv_div] at this
  have hnorm : ‖charFun (coordLaw H v i) u
      - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖
      ≤ Real.exp (-(π * (Real.sqrt d / M) - u) ^ 2 / (2 * (d:ℝ)))
        + Real.exp (-(1 / (2 * M ^ 2))) :=
    (norm_sub_le _ _).trans (add_le_add hmu' hnu)
  rw [gapWeight, div_eq_mul_inv]
  calc ‖charFun (coordLaw H v i) u
        - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ * (u ^ 2)⁻¹
      ≤ (Real.exp (-(π * (Real.sqrt d / M) - u) ^ 2 / (2 * (d:ℝ)))
          + Real.exp (-(1 / (2 * M ^ 2)))) * (4 * M ^ 2 / (π ^ 2 * (d:ℝ))) :=
        mul_le_mul hnorm hinv (by positivity) (by positivity)
    _ = 4 * M ^ 2 / (π ^ 2 * (d:ℝ))
          * (Real.exp (-(π * (Real.sqrt d / M) - u) ^ 2 / (2 * (d:ℝ)))
            + Real.exp (-(1 / (2 * M ^ 2)))) := by ring

/-- **Beyond the period.**  For `u > π·√d/M` the atomic characteristic function is genuinely
uncontrolled and only `‖φ_μ‖ ≤ 1` is used, but the Gaussian is already below `e^{−1/(2M²)}`, so
the bound is `1 + e^{−1/(2M²)}` rather than `2`.  Halving that term is the whole of the `1/π` in
`tailConst`, and no argument here improves on `‖φ_μ‖ ≤ 1`. -/
theorem gapWeight_le_tail_beyond (H : Od d) (v : Ed d) (i : Fin d) {M : ℝ} (hM0 : 0 < M)
    (hd0 : (0:ℝ) < (d:ℝ)) {u : ℝ} (hu : π * (Real.sqrt d / M) < u) :
    gapWeight H v i u ≤ (1 + Real.exp (-(1 / (2 * M ^ 2)))) / u ^ 2 := by
  have hpi : (3:ℝ) < π := Real.pi_gt_three
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hT0 : 0 < Real.sqrt d / M := div_pos hsd hM0
  have hu1 : Real.sqrt d / M ≤ u := by nlinarith
  have hu0 : 0 < u := lt_of_lt_of_le hT0 hu1
  have hnu := norm_charFun_gaussian_le_tail (d := d) hM0 hd0 hu1
  rw [gapWeight, div_le_div_iff_of_pos_right (pow_pos hu0 2)]
  calc ‖charFun (coordLaw H v i) u - charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖
      ≤ ‖charFun (coordLaw H v i) u‖
        + ‖charFun (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) u‖ := norm_sub_le _ _
    _ ≤ 1 + Real.exp (-(1 / (2 * M ^ 2))) := add_le_add (norm_charFun_le_one _) hnu

end TailDeployed

/-! ### §65c  The constant, and the four-piece integral -/

/-- **The tail constant** `K(M) = 1/π + (2√(2π)/π²)·M + 16M⁴`, the replacement for the `2` of
`gapWeight_le_high` in `integral_gapWeight_le_sharp`.  `1/π = 0.318` is the mass of `∫du/u²`
beyond the period `π√d/M` and is the floor of this route; `(2√(2π)/π²)·M = 0.508M` is the
reflected Gaussian window at the period; `16M⁴` is a crude cover for every exponentially small
remainder, and is not sharp.  `K` is not claimed to be optimal: the true tail integral is smaller,
since `‖φ_μ‖ ≤ 1` past the period throws away all of the oscillation. -/
def tailConst (M : ℝ) : ℝ := 1 / π + 2 * Real.sqrt (2 * π) / π ^ 2 * M + 16 * M ^ 4

/-- `22.627 ≤ √512`, the sharper form the numerical corollary needs; `NearFlat.sqrt_512_ge`
gives `22.6`, which is not tight enough for the rounding below.  The true value is `22.62741…`. -/
theorem sqrt_512_ge' : (22.627:ℝ) ≤ Real.sqrt 512 := by
  rw [show (22.627:ℝ) = Real.sqrt (22.627 ^ 2) from (Real.sqrt_sq (by norm_num)).symm]
  exact Real.sqrt_le_sqrt (by norm_num)

/-- `√(2π) ≤ 2.5067`.  The true value is `2.50662…`; nothing here evaluates it further. -/
theorem sqrt_two_pi_le : Real.sqrt (2 * π) ≤ 2.5067 := by
  have hpi : π < 3.141593 := Real.pi_lt_d6
  rw [show (2.5067:ℝ) = Real.sqrt (2.5067 ^ 2) from (Real.sqrt_sq (by norm_num)).symm]
  exact Real.sqrt_le_sqrt (by nlinarith)

/-- `K(M) ≤ 0.319 + 0.51M + 16M⁴` for `M ≥ 0`, the rational cover of `tailConst`.  Both rounded
coefficients are above the true ones (`1/π = 0.3184`, `2√(2π)/π² = 0.5079`). -/
theorem tailConst_le_explicit {M : ℝ} (hM : 0 ≤ M) :
    tailConst M ≤ 319 / 1000 + 51 / 100 * M + 16 * M ^ 4 := by
  have hpi : (3.141592:ℝ) < π := Real.pi_gt_d6
  have hpi0 : (0:ℝ) < π := Real.pi_pos
  have hg := sqrt_two_pi_le
  have h1 : 1 / π ≤ 319 / 1000 := by
    rw [div_le_div_iff₀ hpi0 (by norm_num)]
    linarith
  have h2 : 2 * Real.sqrt (2 * π) / π ^ 2 ≤ 51 / 100 := by
    rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    nlinarith
  rw [tailConst]
  have h3 : 2 * Real.sqrt (2 * π) / π ^ 2 * M ≤ 51 / 100 * M :=
    mul_le_mul_of_nonneg_right h2 hM
  linarith

/-- `K(M) ≤ 9/20` for `0 ≤ M ≤ 1/5`.  The threshold that matters downstream is `0.487`: above it
the composed near-flat bound at `d = 512` and `c = 3.53` returns above the free bound `1`, and
`9/20 = 0.45` is below it with room to spare. -/
theorem tailConst_le_fifth {M : ℝ} (hM0 : 0 ≤ M) (hM : M ≤ 1 / 5) : tailConst M ≤ 9 / 20 := by
  refine (tailConst_le_explicit hM0).trans ?_
  have h4 : M ^ 4 ≤ 1 / 625 := by
    have h := pow_le_pow_left₀ hM0 hM 4
    norm_num at h
    linarith
  linarith

section TailIntegral

variable {d : ℕ}

/-- The low and middle ranges of `integral_gapWeight_le_sharp`, recombined; `83/18` is copied from
there and is not improved here. -/
theorem arith_tail_low {S r : ℝ} (hr : 0 < r) :
    S / (2 * (r ^ 2) ^ 2) * (r ^ 3 / 3) + 40 / 9 * S / r = 83 / 18 * S / r := by
  field_simp
  ring

/-- The reflected half of the tail, evaluated: with `sd² = D` the piece-`D` bound collapses to
`(2G/π²·M² + 2ME/π)/sd`.  Pure algebra, stated separately so that `√d` and `d` do not have to be
identified inside a single `field_simp`. -/
theorem arith_tail_refl {M E G sd D : ℝ} (hM : 0 < M) (hsd : 0 < sd) (hD : sd ^ 2 = D) :
    4 * M ^ 2 / (π ^ 2 * D) * (G * sd / 2 + E * (π * (sd / M) / 2))
      = (2 * G / π ^ 2 * M ^ 2 + 2 * M * E / π) / sd := by
  subst hD
  field_simp
  ring

/-- **The four-piece estimate, with the tail constant.**  Same hypotheses as
`integral_gapWeight_le_sharp`, whose `2M` is replaced by `K(M)·M`:

  `∫₀^∞ ‖φ_μ − φ_ν‖/u² du ≤ (83/18·∑ⱼvⱼ⁴ + K(M)·M)/√d`,  `K(M) = 1/π + (2√(2π)/π²)M + 16M⁴`.

The split is at `√d`, `√d/M`, `(π/2)·√d/M` and `π·√d/M`; the first two pieces are exactly those
of `integral_gapWeight_le_sharp` and their constant `83/18` is unchanged, so **nothing here
improves the spikiness term**.  The gain is confined to the last two pieces and is a factor
`2/K(M) = 4.9` at `d = 512`, `c = 3.53` on the `M`-linear term alone.

`integral_gapWeight_le_sharp` and `gapWeight_le_high` are untouched and remain proved.  The rate
in `d` is unchanged: this is `Θ(1/√d)` exactly as before, and the `M`-linear term is still the
dominant one at a near-flat direction. -/
theorem integral_gapWeight_le_tailsharp {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ = 1)
    (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M) (hM1 : M ≤ 1)
    (hd0 : (0:ℝ) < (d:ℝ)) :
    (∫ u in Set.Ioi (0:ℝ), gapWeight H v i u)
      ≤ (83 / 18 * (∑ j, v j ^ 4) + tailConst M * M) / Real.sqrt d := by
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
  -- integrability on each of the five pieces
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
  -- piece A, copied from `integral_gapWeight_le_sharp`
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
  -- piece B, copied from `integral_gapWeight_le_sharp`
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
  -- piece C, the damped half of the tail
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
  -- piece D, the reflected half of the tail
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
  -- piece F, beyond the period
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
  have hABv : S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3) + 40 / 9 * S / Real.sqrt d
      = 83 / 18 * S / Real.sqrt d := by
    have h := arith_tail_low (S := S) hsd
    rw [hsd2] at h
    exact h
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
      + (40 / 9 * S / Real.sqrt d
        + (2 * E / T
          + (4 * M ^ 2 / (π ^ 2 * (d:ℝ)) * (Real.sqrt (2 * π * (d:ℝ)) / 2 + E * (π * T / 2))
            + (1 + E) / (π * T))))
      ≤ (83 / 18 * S + tailConst M * M) / Real.sqrt d := by
    rw [hCv, hDv, hFv]
    have hleft : S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3)
        + (40 / 9 * S / Real.sqrt d
          + (2 * E * M / Real.sqrt d
            + ((2 * G / π ^ 2 * M ^ 2 + 2 * M * E / π) / Real.sqrt d
              + ((1 + E) * M / π) / Real.sqrt d)))
        = (83 / 18 * S
            + (2 * E * M + (2 * G / π ^ 2 * M ^ 2 + 2 * M * E / π) + (1 + E) * M / π))
            / Real.sqrt d := by
      have hstep : S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3)
          + (40 / 9 * S / Real.sqrt d
            + (2 * E * M / Real.sqrt d
              + ((2 * G / π ^ 2 * M ^ 2 + 2 * M * E / π) / Real.sqrt d
                + ((1 + E) * M / π) / Real.sqrt d)))
          = (S / (2 * (d:ℝ) ^ 2) * (Real.sqrt d ^ 3 / 3) + 40 / 9 * S / Real.sqrt d)
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
        + (40 / 9 * S / Real.sqrt d
          + (2 * E / T
            + (4 * M ^ 2 / (π ^ 2 * (d:ℝ)) * (Real.sqrt (2 * π * (d:ℝ)) / 2 + E * (π * T / 2))
              + (1 + E) / (π * T)))) :=
        add_le_add hA (add_le_add hB (add_le_add hC (add_le_add hD hF)))
    _ ≤ (83 / 18 * S + tailConst M * M) / Real.sqrt d := htot

end TailIntegral

end FourierGap

/-! ### §65d  Leg 1 and the deployed rate with the tail constant

The substitution is mechanical: `NearFlat.abs_cdfGapInt_le_of_legBound` and
`Threshold.abs_hadRate_sub_rhoSq_le_of_legBound` take the two leg-1 constants as universally
quantified reals, so nothing below reproves anything. -/

section TailLeg

open Orthogonal Hadamard FourierGap

variable {d : ℕ}

namespace NearFlat

/-- **Leg 1 with the tail constant.**  `abs_cdfGapInt_coordLaw_gaussian_le_of_bound_best` with
`integral_gapWeight_le_tailsharp` in place of `integral_gapWeight_le_sharp`: for `d ≥ 2`, a flat
`H`, a unit direction `v` with `|vⱼ| ≤ M` and `0 < M ≤ 1`, any coordinate `i` and every pair
`a`, `x`,

  `|cdfGapInt (coordLaw H v i) (N(0,1/d)) a x| ≤ (3·∑ⱼvⱼ⁴ + (2/π)·K(M)·M)/√d`,

against `(3·∑ⱼvⱼ⁴ + (13/10)·M)/√d`.  `13/10 = (2/π)·2` and `(2/π)·K(M) = 0.259` at `M = 0.156`,
so the second term falls by a factor `5.0` there and by at most `2π = 6.28` ever.

**What has not changed.**  It is not a Kolmogorov-Smirnov bound, it is one leg only, the second
law is the Gaussian and not `sphericalCoordLaw d`, the spikiness coefficient `3` is the same `3`,
and the rate in `d` is the same.  The `M`-linear term still dominates at a near-flat direction. -/
theorem abs_cdfGapInt_coordLaw_gaussian_le_tail (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    {v : Ed d} (hv : ‖v‖ = 1) (i : Fin d) {M : ℝ} (hM : ∀ j, |v j| ≤ M) (hM0 : 0 < M)
    (hM1 : M ≤ 1) (a x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (3 * (∑ j, v j ^ 4) + 2 / π * tailConst M * M) / Real.sqrt d := by
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
      ≤ (83 / 18 * (∑ j, v j ^ 4) + tailConst M * M) / Real.sqrt d :=
    FourierGap.integral_gapWeight_le_tailsharp hH hv i hM hM0 hM1 hd0
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
  calc 2 / π * X ≤ 2 / π * ((83 / 18 * (∑ j, v j ^ 4) + tailConst M * M) / Real.sqrt d) :=
        mul_le_mul_of_nonneg_left hI (by positivity)
    _ = (2 / π * (83 / 18 * (∑ j, v j ^ 4) + tailConst M * M)) / Real.sqrt d := by ring
    _ ≤ (3 * (∑ j, v j ^ 4) + 2 / π * tailConst M * M) / Real.sqrt d := by
        gcongr
        have hsplit : 2 / π * (83 / 18 * (∑ j, v j ^ 4) + tailConst M * M)
            = (2 / π * (83 / 18)) * (∑ j, v j ^ 4) + (2 / π * tailConst M) * M := by ring
        rw [hsplit]
        have : (2 / π * (83 / 18)) * (∑ j, v j ^ 4) ≤ 3 * (∑ j, v j ^ 4) :=
          mul_le_mul_of_nonneg_right hcoef1 hS
        linarith

end NearFlat

namespace Threshold

open Hadamard Orthogonal CoordLaw NearFlat

variable {K : ℕ}

/-- **The deployed rate at a near-flat direction, with the tail constant.**  Same hypotheses as
`abs_hadRate_sub_rhoSq_le_of_nearFlat`, whose `(13/10)·c` is replaced:

  `|hadRate H (blockStage q) v − ρ²| ≤ 2·(3c²/√d + (2/π)K(c/√d)·c + 1 + 1/(4(d−1)))·spread cbk`

with `K` as in `tailConst`.  At `d = 512` the factor falls from `14.48` to `7.14` at `c = 3.53`
and from `4.867` to `2.70` at `c = 1`.  The gain is a factor `2/K(M)` on the dominant term, and it
is largest where `M` is smallest, so it is not confined to the flat point: at `c = 3.53`, the
near-flat radius a Haar-random direction actually has, it is still `4.9`.

The caveats of `abs_hadRate_sub_rhoSq_le_of_nearFlat` apply verbatim and none of them is
addressed here.  This is `Θ(spread cbk)` and not `o(1)`; `c` still bounds the **largest**
coordinate, so the near-flat class is still not of full measure on the sphere and
`integral_norm_resid_sq_le_had` is still not fed; `spread cbk` is left abstract and no codebook
is constructed.  `abs_hadRate_sub_rhoSq_le_of_nearFlat` is untouched and not weakened. -/
theorem abs_hadRate_sub_rhoSq_le_of_nearFlat_tail (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) {c : ℝ} (hc0 : 0 < c) (hcd : c ≤ Real.sqrt d)
    (hnf : ∀ j, |v j| ≤ c / Real.sqrt d) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ 2 * (3 * c ^ 2 / Real.sqrt d
          + 2 / π * FourierGap.tailConst (c / Real.sqrt d) * c + 1 + 1 / (4 * ((d:ℝ) - 1)))
          * spread cbk := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hM0 : 0 < c / Real.sqrt d := div_pos hc0 hsd
  have hM1 : c / Real.sqrt d ≤ 1 := by rw [div_le_one hsd]; exact hcd
  have hmain := abs_hadRate_sub_rhoSq_le_of_legBound hd hH v hv i cbk hq hqm ha hb hthr
    (A := 3) (B := 2 / π * FourierGap.tailConst (c / Real.sqrt d)) (c := c)
    (fun x => NearFlat.abs_cdfGapInt_coordLaw_gaussian_le_tail hd hH hv i hnf hM0 hM1 a x)
  refine hmain.trans ?_
  have hS := sum_pow_four_le_sq_div_of_nearFlat hd hv hnf
  have hkey : 3 * (∑ j, v j ^ 4) * Real.sqrt d ≤ 3 * c ^ 2 / Real.sqrt d := by
    have h1 : 3 * (∑ j, v j ^ 4) * Real.sqrt d ≤ 3 * (c ^ 2 / (d:ℝ)) * Real.sqrt d := by gcongr
    refine h1.trans (le_of_eq ?_)
    have hs2 : Real.sqrt (d:ℝ) * Real.sqrt (d:ℝ) = (d:ℝ) := Real.mul_self_sqrt hd0.le
    set s := Real.sqrt (d:ℝ) with hs
    rw [← hs2]
    field_simp
  exact mul_le_mul_of_nonneg_right (by linarith) (spread_nonneg cbk)

/-- **Below the free bound at the near-flat radius a random direction has.**  At `d = 512`, for a
flat `H`, a unit direction near-flat with `c ≤ 3.53`, any ordered codebook whose thresholds lie in
a window around the support and whose **spread is at most `3.0164/√512`**, and any measurable
nearest-codeword map,

  `|hadRate H (blockStage q) v − ρ²| ≤ 24/25`.

`1` is the bound that holds for free whenever `0` is a codeword.
`abs_hadRate_sub_rhoSq_lt_one_of_nearFlat_512` reaches below it only for `c ≤ 7/4`; `3.53` is
about `√(2 log 512)`, the scale at which the coordinate maximum of a Haar-random direction
concentrates, and §60's constant gives `1.93` there.  The true value of this bound is `0.95`;
`24/25` is the rounding that `√512 ≥ 22.6` and `√(2π) ≤ 2.5067` support.

**The codebook width is a hypothesis, not a fact proved here.**  No codebook is constructed,
named or analysed, and nothing is claimed about any codebook the paper ships.  Nor is `1` the
target: `ρ² = 0.1171` is roughly eight times smaller, and this route cannot reach it, since the
coupling leg alone contributes `2·spread cbk = 0.267` at this spread.  The near-flat hypothesis
is a hypothesis about `v`, and the statement says nothing about directions that fail it. -/
theorem abs_hadRate_sub_rhoSq_le_of_nearFlat_tail_512 {H : Od 512} (hH : IsFlat H)
    (v : Ed 512) (hv : ‖v‖ = 1) (i : Fin 512) {c : ℝ} (hc0 : 0 < c) (hc : c ≤ 3.53)
    (hnf : ∀ j, |v j| ≤ c / Real.sqrt 512) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hw : spread cbk ≤ 3.0164 / Real.sqrt 512) :
    |hadRate H (blockStage (m := 512) (F := ℝ) q) v
        - rhoSq 512 q (sphericalCoordLaw 512)| ≤ 24 / 25 := by
  have hs := FourierGap.sqrt_512_ge'
  have hsp : (0:ℝ) < Real.sqrt 512 := by linarith
  have hpi0 : (0:ℝ) < π := Real.pi_pos
  have hpi : (3.141592:ℝ) < π := Real.pi_gt_d6
  have hcast : Real.sqrt ((512:ℕ):ℝ) = Real.sqrt 512 := by norm_num
  have hnf' : ∀ j, |v j| ≤ c / Real.sqrt ((512:ℕ):ℝ) := by simpa [hcast] using hnf
  have hmain := abs_hadRate_sub_rhoSq_le_of_nearFlat_tail (d := 512) (by norm_num) hH v hv i hc0
    (by rw [hcast]; linarith) hnf' cbk hq hqm ha hb hthr
  rw [hcast] at hmain
  push_cast at hmain
  have hsp0 : 0 ≤ spread cbk := spread_nonneg cbk
  -- `M = c/√512 ≤ 0.157`, and `K(M) ≤ 41/100` there
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
  have hfac : 2 * (3 * c ^ 2 / Real.sqrt 512
      + 2 / π * FourierGap.tailConst (c / Real.sqrt 512) * c + 1 + 1 / (4 * ((512:ℝ) - 1)))
      ≤ 715 / 100 := by
    have hc2 : c ^ 2 ≤ 3.53 ^ 2 := by nlinarith
    have h1 : 3 * c ^ 2 / Real.sqrt 512 ≤ 3 * (3.53 ^ 2) / 22.627 := by
      rw [div_le_div_iff₀ hsp (by norm_num)]
      nlinarith [hs, hc2]
    have hBnn : (0:ℝ) ≤ 2 / π * FourierGap.tailConst (c / Real.sqrt 512) := by positivity
    have h2 : 2 / π * FourierGap.tailConst (c / Real.sqrt 512) * c ≤ 2604 / 10000 * 3.53 := by
      nlinarith
    norm_num at h1 h2 ⊢
    linarith
  have hstep : |hadRate H (blockStage (m := 512) (F := ℝ) q) v
      - rhoSq 512 q (sphericalCoordLaw 512)| ≤ 715 / 100 * (3.0164 / Real.sqrt 512) := by
    refine hmain.trans ((mul_le_mul_of_nonneg_right hfac hsp0).trans ?_)
    exact mul_le_mul_of_nonneg_left hw (by norm_num)
  refine hstep.trans ?_
  have heq : (715:ℝ) / 100 * (3.0164 / Real.sqrt 512) = (715 * 3.0164) / (100 * Real.sqrt 512) := by
    ring
  rw [heq, div_le_iff₀ (by positivity)]
  nlinarith [hs]

end Threshold

end TailLeg

end TailConst
