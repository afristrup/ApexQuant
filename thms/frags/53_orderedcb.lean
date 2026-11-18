
/-! ## §52  Ordered scalar codebooks: compact support, bounded cells, integrability

The threshold form of the distortion gap is an identity between two *numbers*, so before any of
it can be stated every integral in it has to be a real number rather than `integral`'s junk
value.  That is what this section supplies, and nothing more: it proves no gap identity and no
bound on a gap.

The objects are fixed here once for the sections that follow.  A `Codebook K` is a strictly
increasing list of `K + 1` scalar codewords; `thr` is the midpoint of two consecutive codewords,
`cellIdx t` counts the thresholds lying strictly below `t`, and `stepQuant` rounds `t` to the
codeword of its cell.  `cellIdx` breaks a tie at a threshold downwards, assigning `t = thr k` the
lower codeword `c k`, but `sq_sub_eq_of_isNearest` makes that choice invisible: any two
nearest-codeword maps for the same codebook have the same squared error at *every* point, since
both errors are the distance to the codebook.  So every statement below proved for `stepQuant`
transports verbatim to an arbitrary nearest map, and no theorem here or later needs a tie
convention as a hypothesis.

The support hypothesis is the single measure-theoretic input: `μ (Icc a b)ᶜ = 0`.  Under it
`setIntegral_eq_of_supported` trims any integral to the interval, and `t`, `t²` and `(t − q t)²`
are all integrable, the last because a step quantiser takes finitely many values and is therefore
bounded, by `stepBound`, the larger of the two extreme codewords in absolute value.  `stepBound`
is not the spread `c_K − c_0`, and is not claimed to be sharp; it is the honest crude bound that
the integrability arguments actually use.

Both laws the paper compares satisfy the hypothesis: the deployed coordinate law lives in
`[−‖v‖, ‖v‖]` because a coordinate of a rotated vector is bounded by its norm, and the spherical
coordinate law lives in `[−1, 1]` because it is defined by a density against Lebesgue measure
restricted to that interval.  Neither statement needs `2 ≤ d`.

`cdfGapInt μ ν a x`, the once-integrated CDF difference over `(a, x]`, is defined here because it
is the object the threshold identity is phrased in.  Its integrand is bounded by one and
monotone in each summand, hence integrable on every bounded interval, so `cdfGapInt` is a real
number; and a sup-norm bound `η` on `cdf μ − cdf ν` bounds it by `η · (x − a)`.  That last
inequality is the only estimate in this section, and it is an estimate on `cdfGapInt` alone, not
on any distortion gap. -/

namespace Threshold

open Hadamard Orthogonal

/-! ### Ordered scalar codebooks -/

/-- An ordered scalar codebook: `K + 1` codewords listed in strictly increasing order. -/
structure Codebook (K : ℕ) where
  /-- The codewords. -/
  c : Fin (K + 1) → ℝ
  /-- They are listed in strictly increasing order, so in particular they are distinct. -/
  mono : StrictMono c

variable {K : ℕ}

/-- The `k`-th threshold: the **midpoint** of the two codewords it separates.  That it is the
midpoint, and not some other point of the cell boundary, is what makes the piecewise-linear part
of `(t − q t)²` continuous, and is used in `isNearest_stepQuant`. -/
noncomputable def thr (cbk : Codebook K) (k : Fin K) : ℝ :=
  (cbk.c k.castSucc + cbk.c k.succ) / 2

/-- The thresholds inherit the strict order of the codewords. -/
theorem thr_strictMono (cbk : Codebook K) : StrictMono (thr cbk) := by
  intro k l hkl
  have hk : cbk.c k.castSucc < cbk.c k.succ := cbk.mono Fin.castSucc_lt_succ
  have hl : cbk.c l.castSucc < cbk.c l.succ := cbk.mono Fin.castSucc_lt_succ
  have hmid : cbk.c k.succ ≤ cbk.c l.castSucc := by
    refine cbk.mono.monotone ?_
    rw [Fin.le_def, Fin.val_succ, Fin.val_castSucc]
    have h := Fin.lt_def.mp hkl
    omega
  unfold thr
  linarith

/-- The thresholds lying strictly below `t`. -/
noncomputable def belowThr (cbk : Codebook K) (t : ℝ) : Finset (Fin K) :=
  Finset.univ.filter fun k => thr cbk k < t

theorem mem_belowThr (cbk : Codebook K) (t : ℝ) (k : Fin K) :
    k ∈ belowThr cbk t ↔ thr cbk k < t := by
  simp [belowThr]

theorem card_belowThr_le (cbk : Codebook K) (t : ℝ) : (belowThr cbk t).card ≤ K := by
  have h := Finset.card_filter_le (Finset.univ : Finset (Fin K)) fun k : Fin K => thr cbk k < t
  simpa [belowThr, Finset.card_univ] using h

/-- The index of the cell containing `t`: the number of thresholds lying strictly below `t`.
There are `K` thresholds, so this is an index into the `K + 1` codewords. -/
noncomputable def cellIdx (cbk : Codebook K) (t : ℝ) : Fin (K + 1) :=
  ⟨(belowThr cbk t).card, Nat.lt_succ_of_le (card_belowThr_le cbk t)⟩

theorem cellIdx_val (cbk : Codebook K) (t : ℝ) :
    (cellIdx cbk t : ℕ) = (belowThr cbk t).card := rfl

/-- **The thresholds below `t` form an initial segment.**  Because the thresholds increase, the
`k`-th one lies below `t` exactly when `k` lies below the cell index.  This is the only place the
strict order of the codebook is converted into a statement about `cellIdx`. -/
theorem thr_lt_iff_lt_cellIdx (cbk : Codebook K) (t : ℝ) (k : Fin K) :
    thr cbk k < t ↔ (k : ℕ) < (cellIdx cbk t : ℕ) := by
  rw [cellIdx_val]
  constructor
  · intro hk
    have hsub : Finset.Iic k ⊆ belowThr cbk t := by
      intro j hj
      rw [mem_belowThr]
      exact lt_of_le_of_lt ((thr_strictMono cbk).monotone (Finset.mem_Iic.mp hj)) hk
    have hcard := Finset.card_le_card hsub
    rw [Fin.card_Iic] at hcard
    omega
  · intro hk
    by_contra hkt
    have hsub : belowThr cbk t ⊆ Finset.Iio k := by
      intro j hj
      rw [Finset.mem_Iio]
      by_contra hjk
      exact hkt (lt_of_le_of_lt ((thr_strictMono cbk).monotone (not_lt.mp hjk))
        ((mem_belowThr cbk t j).mp hj))
    have hcard := Finset.card_le_card hsub
    rw [Fin.card_Iio] at hcard
    omega

/-- The step quantiser attached to an ordered codebook: round `t` to the codeword of its cell. -/
noncomputable def stepQuant (cbk : Codebook K) (t : ℝ) : ℝ := cbk.c (cellIdx cbk t)

/-- The spread of the codebook: the distance from the smallest codeword to the largest. -/
noncomputable def spread (cbk : Codebook K) : ℝ := cbk.c (Fin.last K) - cbk.c 0

theorem spread_nonneg (cbk : Codebook K) : 0 ≤ spread cbk :=
  sub_nonneg.mpr (cbk.mono.monotone (Fin.zero_le _))

/-- The cell index is monotone: raising `t` can only add thresholds below it. -/
theorem monotone_cellIdx (cbk : Codebook K) : Monotone (cellIdx cbk) := by
  intro s t hst
  have hsub : belowThr cbk s ⊆ belowThr cbk t := by
    intro k hk
    rw [mem_belowThr] at hk ⊢
    exact lt_of_lt_of_le hk hst
  rw [Fin.le_def, cellIdx_val, cellIdx_val]
  exact Finset.card_le_card hsub

/-- Hence the step quantiser is monotone, and in particular measurable. -/
theorem monotone_stepQuant (cbk : Codebook K) : Monotone (stepQuant cbk) :=
  fun _ _ hst => cbk.mono.monotone (monotone_cellIdx cbk hst)

theorem measurable_stepQuant (cbk : Codebook K) : Measurable (stepQuant cbk) :=
  (monotone_stepQuant cbk).measurable

/-- A crude bound on the values of a step quantiser: the larger of the two extreme codewords in
absolute value.  This is **not** the spread `c_K − c_0`, and it is not claimed to be sharp; it is
what the integrability arguments below use. -/
noncomputable def stepBound (cbk : Codebook K) : ℝ := max |cbk.c 0| |cbk.c (Fin.last K)|

/-- Every value of a step quantiser is bounded by `stepBound`. -/
theorem abs_stepQuant_le (cbk : Codebook K) (t : ℝ) : |stepQuant cbk t| ≤ stepBound cbk := by
  have h0 : cbk.c 0 ≤ stepQuant cbk t := cbk.mono.monotone (Fin.zero_le _)
  have h1 : stepQuant cbk t ≤ cbk.c (Fin.last K) := cbk.mono.monotone (Fin.le_last _)
  have hl : -|cbk.c 0| ≤ cbk.c 0 := neg_abs_le _
  have hr : cbk.c (Fin.last K) ≤ |cbk.c (Fin.last K)| := le_abs_self _
  have hml : |cbk.c 0| ≤ stepBound cbk := le_max_left _ _
  have hmr : |cbk.c (Fin.last K)| ≤ stepBound cbk := le_max_right _ _
  rw [abs_le]
  exact ⟨by linarith, by linarith⟩

/-! ### The step quantiser rounds to a nearest codeword -/

/-- **`stepQuant` is a nearest-codeword map.**  The midpoint placement of the thresholds is
exactly what makes this true: on the cell of `c i` the point `t` satisfies
`c_{i−1} + c_i < 2t ≤ c_i + c_{i+1}` at whichever of the two ends exists, and the comparison
`(t − c_i)² ≤ (t − c_j)²` then factors as `(c_j − c_i)(2t − c_i − c_j) ≤ 0`.  At `t = thr k` both
codewords are nearest and `cellIdx` returns the lower one, `c k`; `sq_sub_eq_of_isNearest` makes
that choice immaterial. -/
theorem isNearest_stepQuant (cbk : Codebook K) :
    IsNearest (Set.range cbk.c) (stepQuant cbk) := by
  intro t
  refine ⟨⟨cellIdx cbk t, rfl⟩, ?_⟩
  rintro _ ⟨j, rfl⟩
  have hsq : (t - cbk.c (cellIdx cbk t)) ^ 2 ≤ (t - cbk.c j) ^ 2 := by
    rcases lt_trichotomy j (cellIdx cbk t) with hji | hji | hji
    · -- `j` is below the cell: the threshold just below `t` separates them
      have hpos : 0 < (cellIdx cbk t : ℕ) := lt_of_le_of_lt (Nat.zero_le _) (Fin.lt_def.mp hji)
      have hK : (cellIdx cbk t : ℕ) - 1 < K := by
        have := (cellIdx cbk t).isLt
        omega
      set m : Fin K := ⟨(cellIdx cbk t : ℕ) - 1, hK⟩ with hm
      have hmlt : (m : ℕ) < (cellIdx cbk t : ℕ) := by simp only [hm]; omega
      have hthr : thr cbk m < t := (thr_lt_iff_lt_cellIdx cbk t m).mpr hmlt
      have hsucc : cbk.c m.succ = cbk.c (cellIdx cbk t) := by
        refine congrArg cbk.c (Fin.val_injective ?_)
        simp only [hm, Fin.val_succ]
        omega
      have hj : cbk.c j ≤ cbk.c m.castSucc := by
        refine cbk.mono.monotone ?_
        rw [Fin.le_def, Fin.val_castSucc]
        simp only [hm]
        have := Fin.lt_def.mp hji
        omega
      have hlt : cbk.c j < cbk.c (cellIdx cbk t) := cbk.mono hji
      have hsum : cbk.c j + cbk.c (cellIdx cbk t) < 2 * t := by
        rw [← hsucc]
        unfold thr at hthr
        linarith
      nlinarith [hlt, hsum]
    · rw [hji]
    · -- `j` is above the cell: the threshold just above `t` separates them
      have hK : (cellIdx cbk t : ℕ) < K := by
        have := j.isLt
        have := Fin.lt_def.mp hji
        omega
      set m : Fin K := ⟨(cellIdx cbk t : ℕ), hK⟩ with hm
      have hmge : ¬ ((m : ℕ) < (cellIdx cbk t : ℕ)) := by simp [hm]
      have hthr : t ≤ thr cbk m := not_lt.mp fun h =>
        hmge ((thr_lt_iff_lt_cellIdx cbk t m).mp h)
      have hcast : cbk.c m.castSucc = cbk.c (cellIdx cbk t) := by
        refine congrArg cbk.c (Fin.val_injective ?_)
        simp [hm]
      have hj : cbk.c m.succ ≤ cbk.c j := by
        refine cbk.mono.monotone ?_
        rw [Fin.le_def, Fin.val_succ]
        simp only [hm]
        have := Fin.lt_def.mp hji
        omega
      have hlt : cbk.c (cellIdx cbk t) < cbk.c j := cbk.mono hji
      have hsum : 2 * t ≤ cbk.c (cellIdx cbk t) + cbk.c j := by
        rw [← hcast]
        unfold thr at hthr
        linarith
      nlinarith [hlt, hsum]
  show ‖t - stepQuant cbk t‖ ≤ ‖t - cbk.c j‖
  rw [Real.norm_eq_abs, Real.norm_eq_abs, ← Real.sqrt_sq_eq_abs, ← Real.sqrt_sq_eq_abs]
  exact Real.sqrt_le_sqrt hsq

/-- **Ties do not matter.**  Two nearest-codeword maps for the same codebook have the same squared
error at *every* point, because both errors equal the distance to the codebook.  Consequently
every statement about `stepQuant` below transports to an arbitrary nearest map, and no tie
convention is ever assumed. -/
theorem sq_sub_eq_of_isNearest {C : Set ℝ} {q q' : ℝ → ℝ} (hq : IsNearest C q)
    (hq' : IsNearest C q') (t : ℝ) : (t - q t) ^ 2 = (t - q' t) ^ 2 := by
  have h := (norm_sub_eq_infDist hq t).trans (norm_sub_eq_infDist hq' t).symm
  rw [Real.norm_eq_abs, Real.norm_eq_abs] at h
  rw [← sq_abs (t - q t), ← sq_abs (t - q' t), h]

/-- The distortion of a codebook against a law does not depend on which nearest map realises it.
No integrability hypothesis is needed: the two integrands are equal pointwise. -/
theorem distortion_congr_of_isNearest {C : Set ℝ} {q q' : ℝ → ℝ} (hq : IsNearest C q)
    (hq' : IsNearest C q') (μ : Measure ℝ) : distortion q μ = distortion q' μ :=
  integral_congr_ae (Filter.Eventually.of_forall fun t => sq_sub_eq_of_isNearest hq hq' t)

/-! ### Compact support, and trimming an integral to it -/

/-- A law giving the complement of `Icc a b` no mass is concentrated on `Icc a b`. -/
theorem ae_mem_Icc_of_supported {μ : Measure ℝ} {a b : ℝ} (hsupp : μ (Set.Icc a b)ᶜ = 0) :
    ∀ᵐ t ∂μ, t ∈ Set.Icc a b := ae_iff.mpr hsupp

/-- **Trimming an integral to the support.**  When `μ` gives the complement of `Icc a b` no mass,
an integral over any set is the integral over its intersection with `Icc a b`.  Neither
measurability of the set nor integrability of `f` is needed: the two restricted measures agree,
so the two integrals are the same term. -/
theorem setIntegral_eq_of_supported {μ : Measure ℝ} {a b : ℝ} (hsupp : μ (Set.Icc a b)ᶜ = 0)
    (f : ℝ → ℝ) (s : Set ℝ) :
    ∫ t in s, f t ∂μ = ∫ t in s ∩ Set.Icc a b, f t ∂μ := by
  have key : μ.restrict s = μ.restrict (s ∩ Set.Icc a b) := by
    refine Measure.restrict_congr_set ?_
    rw [Filter.eventuallyEq_set]
    filter_upwards [ae_mem_Icc_of_supported hsupp] with x hx
    simp [hx]
  rw [key]

/-- The whole-line integral of a compactly supported law is an integral over the interval. -/
theorem integral_eq_setIntegral_Icc_of_supported {μ : Measure ℝ} {a b : ℝ}
    (hsupp : μ (Set.Icc a b)ᶜ = 0) (f : ℝ → ℝ) :
    ∫ t, f t ∂μ = ∫ t in Set.Icc a b, f t ∂μ := by
  rw [← setIntegral_univ, setIntegral_eq_of_supported hsupp f Set.univ, Set.univ_inter]

/-! ### Integrability against a compactly supported law -/

/-- The crude bound on `|t|` for `t` in `Icc a b`. -/
noncomputable def intervalBound (a b : ℝ) : ℝ := max |a| |b|

theorem abs_le_intervalBound {a b t : ℝ} (ht : t ∈ Set.Icc a b) : |t| ≤ intervalBound a b := by
  have h1 : -|a| ≤ a := neg_abs_le a
  have h2 : b ≤ |b| := le_abs_self b
  have h3 : |a| ≤ intervalBound a b := le_max_left _ _
  have h4 : |b| ≤ intervalBound a b := le_max_right _ _
  rw [abs_le]
  exact ⟨by linarith [ht.1], by linarith [ht.2]⟩

/-- `t` is integrable against a finite law supported in a bounded interval. -/
theorem integrable_id_of_supported (μ : Measure ℝ) [IsFiniteMeasure μ] {a b : ℝ}
    (hsupp : μ (Set.Icc a b)ᶜ = 0) : Integrable (fun t : ℝ => t) μ := by
  refine (integrable_const (intervalBound a b)).mono' measurable_id.aestronglyMeasurable ?_
  filter_upwards [ae_mem_Icc_of_supported hsupp] with x hx
  rw [Real.norm_eq_abs]
  exact abs_le_intervalBound hx

/-- `t²` is integrable against a finite law supported in a bounded interval. -/
theorem integrable_sq_of_supported (μ : Measure ℝ) [IsFiniteMeasure μ] {a b : ℝ}
    (hsupp : μ (Set.Icc a b)ᶜ = 0) : Integrable (fun t : ℝ => t ^ 2) μ := by
  refine (integrable_const (intervalBound a b ^ 2)).mono'
    (measurable_id.pow_const 2).aestronglyMeasurable ?_
  filter_upwards [ae_mem_Icc_of_supported hsupp] with x hx
  rw [Real.norm_eq_abs, abs_pow]
  exact pow_le_pow_left₀ (abs_nonneg _) (abs_le_intervalBound hx) 2

/-- `(t − stepQuant t)²` is integrable against a finite law supported in a bounded interval: the
quantiser is bounded by `stepBound`, so the integrand is bounded outright. -/
theorem integrable_sub_stepQuant_sq_of_supported (cbk : Codebook K) (μ : Measure ℝ)
    [IsFiniteMeasure μ] {a b : ℝ} (hsupp : μ (Set.Icc a b)ᶜ = 0) :
    Integrable (fun t : ℝ => (t - stepQuant cbk t) ^ 2) μ := by
  refine (integrable_const ((intervalBound a b + stepBound cbk) ^ 2)).mono'
    (((measurable_id.sub (measurable_stepQuant cbk)).pow_const 2).aestronglyMeasurable) ?_
  filter_upwards [ae_mem_Icc_of_supported hsupp] with x hx
  have hA := abs_le.mp (abs_le_intervalBound hx)
  have hB := abs_le.mp (abs_stepQuant_le cbk x)
  have h1 : |x - stepQuant cbk x| ≤ intervalBound a b + stepBound cbk := by
    rw [abs_le]
    exact ⟨by linarith [hA.1, hB.2], by linarith [hA.2, hB.1]⟩
  rw [Real.norm_eq_abs, abs_pow]
  exact pow_le_pow_left₀ (abs_nonneg _) h1 2

/-- The same for an **arbitrary** nearest-codeword map, by `sq_sub_eq_of_isNearest`.  This is the
form the gap identity needs, since its quantiser is any nearest map and not a chosen one. -/
theorem integrable_sub_isNearest_sq_of_supported {cbk : Codebook K} {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ : Measure ℝ) [IsFiniteMeasure μ] {a b : ℝ}
    (hsupp : μ (Set.Icc a b)ᶜ = 0) : Integrable (fun t : ℝ => (t - q t) ^ 2) μ :=
  (integrable_sub_stepQuant_sq_of_supported cbk μ hsupp).congr
    (Filter.Eventually.of_forall fun t => sq_sub_eq_of_isNearest (isNearest_stepQuant cbk) hq t)

/-! ### Interval masses, and the once-integrated CDF difference -/

/-- **A bounded interval's mass, as a difference of CDF values.**  Mathlib has no
`Set.Iic_diff_Iic`, so the set identity is done by hand. -/
theorem measureReal_Ioc_eq_cdf_sub (μ : Measure ℝ) [IsProbabilityMeasure μ] (u b : ℝ)
    (hub : u ≤ b) : μ.real (Set.Ioc u b) = cdf μ b - cdf μ u := by
  have hdiff : Set.Ioc u b = Set.Iic b \ Set.Iic u := by
    ext t
    simp only [Set.mem_Ioc, Set.mem_sdiff, Set.mem_Iic, not_le]
    exact ⟨fun h => ⟨h.2, h.1⟩, fun h => ⟨h.2, h.1⟩⟩
  rw [cdf_eq_real, cdf_eq_real, hdiff,
    measureReal_sdiff (Set.Iic_subset_Iic.mpr hub) measurableSet_Iic]

/-- The identity is integrable on a bounded interval against any finite measure. -/
theorem integrableOn_id_Ioc (μ : Measure ℝ) [IsFiniteMeasure μ] (a b : ℝ) :
    IntegrableOn (fun t : ℝ => t) (Set.Ioc a b) μ := by
  refine (integrable_const (intervalBound a b)).mono' measurable_id.aestronglyMeasurable ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioc] with t ht
  rw [Real.norm_eq_abs]
  exact abs_le_intervalBound ⟨ht.1.le, ht.2⟩

/-- A CDF is integrable on a bounded interval: it is monotone, hence measurable, and it is bounded
between zero and one. -/
theorem integrableOn_cdf_Ioc (μ : Measure ℝ) (a b : ℝ) :
    IntegrableOn (fun u : ℝ => (cdf μ) u) (Set.Ioc a b) volume := by
  haveI : IsFiniteMeasure ((volume : Measure ℝ).restrict (Set.Ioc a b)) := by
    constructor
    rw [Measure.restrict_apply_univ, Real.volume_Ioc]
    exact ENNReal.ofReal_lt_top
  refine (integrable_const (1 : ℝ)).mono' ((monotone_cdf μ).measurable).aestronglyMeasurable ?_
  filter_upwards with u
  rw [Real.norm_eq_abs, abs_of_nonneg (cdf_nonneg μ u)]
  exact cdf_le_one μ u

/-- **The once-integrated CDF difference** over `(a, x]`: the object the threshold form of the
distortion gap is phrased in. -/
noncomputable def cdfGapInt (μ ν : Measure ℝ) (a x : ℝ) : ℝ :=
  ∫ u in Set.Ioc a x, ((cdf μ) u - (cdf ν) u)

/-- The integrand of `cdfGapInt` is integrable on every bounded interval, so `cdfGapInt` is a real
number and not `integral`'s junk value. -/
theorem integrableOn_cdfGap_Ioc (μ ν : Measure ℝ) (a x : ℝ) :
    IntegrableOn (fun u : ℝ => (cdf μ) u - (cdf ν) u) (Set.Ioc a x) volume :=
  (integrableOn_cdf_Ioc μ a x).sub (integrableOn_cdf_Ioc ν a x)

/-- `cdfGapInt` splits as the difference of the two once-integrated CDFs. -/
theorem cdfGapInt_eq_sub (μ ν : Measure ℝ) (a x : ℝ) :
    cdfGapInt μ ν a x = (∫ u in Set.Ioc a x, (cdf μ) u) - ∫ u in Set.Ioc a x, (cdf ν) u :=
  integral_sub (integrableOn_cdf_Ioc μ a x) (integrableOn_cdf_Ioc ν a x)

/-- `cdfGapInt` of a law against itself is zero. -/
theorem cdfGapInt_self (μ : Measure ℝ) (a x : ℝ) : cdfGapInt μ μ a x = 0 := by
  rw [cdfGapInt_eq_sub, sub_self]

/-- **A sup-norm bound on the CDF difference bounds the once-integrated difference.**  If
`|cdf μ − cdf ν| ≤ η` everywhere, then `|cdfGapInt μ ν a x| ≤ η · (x − a)` for `a ≤ x`.  This is
the only estimate in this section, and it concerns `cdfGapInt` alone: no distortion gap is bounded
here. -/
theorem abs_cdfGapInt_le (μ ν : Measure ℝ) {a x η : ℝ} (hax : a ≤ x)
    (hks : ∀ u, |(cdf μ) u - (cdf ν) u| ≤ η) : |cdfGapInt μ ν a x| ≤ η * (x - a) := by
  have hvol : (volume : Measure ℝ) (Set.Ioc a x) < ⊤ := by
    rw [Real.volume_Ioc]
    exact ENNReal.ofReal_lt_top
  have h := norm_setIntegral_le_of_norm_le_const (μ := (volume : Measure ℝ))
    (s := Set.Ioc a x) (f := fun u : ℝ => (cdf μ) u - (cdf ν) u) (C := η) hvol
    (fun u _ => by rw [Real.norm_eq_abs]; exact hks u)
  rw [Real.norm_eq_abs, Real.volume_real_Ioc, max_eq_left (by linarith)] at h
  exact h

/-! ### The two laws the paper compares are compactly supported -/

/-- A coordinate of a Euclidean vector is bounded by its norm. -/
theorem abs_coord_le_norm {d : ℕ} (x : Ed d) (i : Fin d) : |x i| ≤ ‖x‖ := by
  have h2 : |x i| ^ 2 ≤ ‖x‖ ^ 2 := by
    rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
    exact Finset.single_le_sum (f := fun j => ‖x j‖ ^ 2) (fun j _ => by positivity)
      (Finset.mem_univ i)
  nlinarith [abs_nonneg (x i), norm_nonneg x]

/-- **The deployed coordinate law is supported in `[−‖v‖, ‖v‖]`.**  A rotation preserves the norm
and a coordinate is bounded by the norm, so every value of the pushed-forward coordinate lies in
that interval.  No hypothesis on `d` is used. -/
theorem coordLaw_compl_Icc_eq_zero {d : ℕ} (H : Od d) (v : Ed d) (i : Fin d) :
    (coordLaw H v i) (Set.Icc (-‖v‖) ‖v‖)ᶜ = 0 := by
  have hmeas : Measurable fun s : Signs d => (hadRot H s • v) i := measurable_of_countable _
  have hpre : (fun s : Signs d => (hadRot H s • v) i) ⁻¹' (Set.Icc (-‖v‖) ‖v‖)ᶜ
      = (∅ : Set (Signs d)) := by
    ext s
    simp only [Set.mem_preimage, Set.mem_compl_iff, Set.mem_empty_iff_false, iff_false, not_not]
    have h1 : |(hadRot H s • v) i| ≤ ‖hadRot H s • v‖ := abs_coord_le_norm _ i
    rw [Orthogonal.norm_smul' d (hadRot H s) v] at h1
    exact Set.mem_Icc.mpr (abs_le.mp h1)
  rw [coordLaw, Measure.map_apply hmeas measurableSet_Icc.compl, hpre, measure_empty]

/-- **The spherical coordinate law is supported in `[−1, 1]`**, being a density against Lebesgue
measure restricted to that interval.  No hypothesis on `d` is used; in particular this does not
assume `2 ≤ d`, which is what makes it a probability measure. -/
theorem sphericalCoordLaw_compl_Icc_eq_zero (d : ℕ) :
    (sphericalCoordLaw d) (Set.Icc (-1 : ℝ) 1)ᶜ = 0 := by
  have hsupp : ∀ᵐ t ∂(sphericalCoordLaw d), t ∈ Set.Icc (-1 : ℝ) 1 :=
    (withDensity_absolutelyContinuous _ _).ae_le (ae_restrict_mem measurableSet_Icc)
  exact ae_iff.mp hsupp

/-! ### The hypotheses discharge for both laws

These are the instantiations the later sections use; each is one application of the general
statement above to the support proved for the law in question. -/

/-- `t` is integrable against the deployed coordinate law. -/
theorem integrable_id_coordLaw {d : ℕ} (H : Od d) (v : Ed d) (i : Fin d) :
    Integrable (fun t : ℝ => t) (coordLaw H v i) :=
  integrable_id_of_supported _ (coordLaw_compl_Icc_eq_zero H v i)

/-- `t²` is integrable against the deployed coordinate law. -/
theorem integrable_sq_coordLaw {d : ℕ} (H : Od d) (v : Ed d) (i : Fin d) :
    Integrable (fun t : ℝ => t ^ 2) (coordLaw H v i) :=
  integrable_sq_of_supported _ (coordLaw_compl_Icc_eq_zero H v i)

/-- `(t − q t)²` is integrable against the deployed coordinate law, for every nearest-codeword map
of an ordered codebook. -/
theorem integrable_sub_isNearest_sq_coordLaw {d : ℕ} (H : Od d) (v : Ed d) (i : Fin d)
    {cbk : Codebook K} {q : ℝ → ℝ} (hq : IsNearest (Set.range cbk.c) q) :
    Integrable (fun t : ℝ => (t - q t) ^ 2) (coordLaw H v i) :=
  integrable_sub_isNearest_sq_of_supported hq _ (coordLaw_compl_Icc_eq_zero H v i)

/-- `t` is integrable against the spherical coordinate law, for `2 ≤ d`, which is what makes it a
probability measure. -/
theorem integrable_id_sphericalCoordLaw {d : ℕ} (hd : 2 ≤ d) :
    Integrable (fun t : ℝ => t) (sphericalCoordLaw d) := by
  haveI := isProbabilityMeasure_sphericalCoordLaw hd
  exact integrable_id_of_supported _ (sphericalCoordLaw_compl_Icc_eq_zero d)

/-- `(t − q t)²` is integrable against the spherical coordinate law, for `2 ≤ d`. -/
theorem integrable_sub_isNearest_sq_sphericalCoordLaw {d : ℕ} (hd : 2 ≤ d)
    {cbk : Codebook K} {q : ℝ → ℝ} (hq : IsNearest (Set.range cbk.c) q) :
    Integrable (fun t : ℝ => (t - q t) ^ 2) (sphericalCoordLaw d) := by
  haveI := isProbabilityMeasure_sphericalCoordLaw hd
  exact integrable_sub_isNearest_sq_of_supported hq _ (sphericalCoordLaw_compl_Icc_eq_zero d)

end Threshold
