
/-! ## §62  A coordinate direction, by direct pushforward

§48b's `coordLaw_eq_radLaw` identifies the law of coordinate `i` of `H D_ε v`, for an
**arbitrary** direction `v`, with `radLaw d v`, by reindexing the sign cube through the general
Rademacher-sum congruence `map_radSum_congr_abs`.  This section proves the one instance the
Phase-1 claim actually needs, `v = eⱼ` a coordinate direction, by a different and more
elementary route that does not mention `radLaw` at all.

At `v = eⱼ`, `hadRot_apply_eq_radSum` writes coordinate `i` as the Rademacher sum
`∑ₖ Hᵢₖ (eⱼ)ₖ εₖ`, and every term but `k = j` vanishes because `eⱼ` is the indicator of `j`: the
sum collapses to the single term `Hᵢⱼ εⱼ`.  So the map `ε ↦ (H D_ε eⱼ)ᵢ` **is**, on the nose,
`s ↦ Hᵢⱼ · sgn (s j)`, the characterisation the section is named for
(`coord_basis_apply`), and the coordinate law is the pushforward of `uniformSigns d` along it.
That pushforward is computed directly, by hand: flipping bit `j` of the sign pattern is an
involution of the cube that preserves `uniformSigns d` (`map_uniformSigns_equiv`, from §48b) and
exchanges `{s | s j = true}` with `{s | s j = false}`, so the two fibres carry equal mass, and
since they partition the cube that mass is `1/2` each (`uniformSigns_eval_true`,
`uniformSigns_eval_false`).  Reading the two fibres through `sgn` and then through `Hᵢⱼ`
(`coordLaw_basis_eq_scaled_bernoulli`) turns that into the pushforward measure directly, no
`radLaw`, no general Rademacher-sum congruence: a case split on membership of `Hᵢⱼ` and `-Hᵢⱼ` in
an arbitrary measurable set is the whole proof.

Only the last step, identifying `Hᵢⱼ` and `-Hᵢⱼ` with `±1/√d`, uses flatness
(`coordLaw_basis_eq_twoPoint`); `coordLaw_basis_eq_scaled_bernoulli` itself needs no hypothesis on
`H` beyond membership in `O(d)`, holds even where `Hᵢⱼ = 0` (vacuously, both dirac masses then
coincide at `0`), and holds for the *value* `Hᵢⱼ` before flatness ever enters.  No hypothesis
`2 ≤ d` or `1 ≤ d` is stated separately: `i j : Fin d` are already data, and `Fin d` nonempty
forces `d ≥ 1`.

**What this does not do.**  This is the coordinate law at *one* direction, `v = eⱼ`; it says
nothing about `hadRate` there, nothing about any codebook's distortion against this two-point
law, and nothing about how that distortion compares with `ρ²` or with the spherical marginal
`f_d` that Proposition 2's recursion ranges over.  It does not touch, weaken or reprove §48b's
`coordLaw_eq_radLaw` for a general `v`, and it does not compute `radLaw d eⱼ` or relate it to the
two-point law found here, the point of this section is that the identification of `coordLaw` at
`eⱼ` needs none of that machinery, not that the machinery would give a different answer. -/

namespace CoordLaw

open Hadamard

variable {d : ℕ}

/-- A coordinate direction is a unit vector: `‖eⱼ‖ = 1`.  Recorded so that "coordinate direction"
above is not just a name; nothing below uses it. -/
theorem norm_single_eq_one (j : Fin d) : ‖(EuclideanSpace.single j (1:ℝ) : Ed d)‖ = 1 := by
  rw [PiLp.norm_single]
  simp

/-- **The map `ε ↦ (H D_ε eⱼ)ᵢ` is exactly `s ↦ Hᵢⱼ · sgn (s j)`.**  Every other term of the
Rademacher sum `hadRot_apply_eq_radSum` produces is killed by `eⱼ`, which is `0` off `j`; this is
the collapse the section's whole argument rests on, and it uses no hypothesis on `H`. -/
theorem coord_basis_apply (H : Od d) (i j : Fin d) (s : Signs d) :
    (hadRot H s • (EuclideanSpace.single j (1:ℝ) : Ed d)) i
      = (H : Matrix (Fin d) (Fin d) ℝ) i j * sgn (s j) := by
  rw [hadRot_apply_eq_radSum]
  unfold radSum
  rw [Finset.sum_eq_single j]
  · simp
  · intro k _ hk; simp [hk]
  · intro h; exact absurd (Finset.mem_univ j) h

/-! ### The bit-`j` fibres of the sign cube carry equal mass, each `1/2`

Elementary bookkeeping only: the two fibres partition the cube, and the involution that flips bit
`j` and leaves every other bit alone exchanges them while preserving `uniformSigns d`. -/

theorem uniformSigns_eval_true_add_false (j : Fin d) :
    uniformSigns d {s : Signs d | s j = true} + uniformSigns d {s : Signs d | s j = false} = 1 := by
  have hunion : {s : Signs d | s j = true} ∪ {s : Signs d | s j = false} = Set.univ := by
    ext s
    simp only [Set.mem_union, Set.mem_setOf_eq, Set.mem_univ, iff_true]
    rcases Bool.eq_false_or_eq_true (s j) with h | h <;> simp [h]
  have hdisj : Disjoint {s : Signs d | s j = true} {s : Signs d | s j = false} := by
    rw [Set.disjoint_left]
    intro s h1 h2
    simp only [Set.mem_setOf_eq] at h1 h2
    rw [h1] at h2
    exact Bool.noConfusion h2
  rw [← measure_union hdisj (Set.toFinite _).measurableSet, hunion, measure_univ]

theorem uniformSigns_eval_true (j : Fin d) :
    uniformSigns d {s : Signs d | s j = true} = 1 / 2 := by
  set τ : Fin d → Bool := fun k => decide (k = j) with hτ
  have he : (uniformSigns d).map (flipEquiv τ) = uniformSigns d :=
    map_uniformSigns_equiv (flipEquiv τ)
  have hflip : ∀ s : Signs d, (flipEquiv τ s) j = !(s j) := by
    intro s
    show xor (τ j) (s j) = !(s j)
    simp [hτ]
  have hms : MeasurableSet {s : Signs d | s j = true} := (Set.toFinite _).measurableSet
  have hswap : uniformSigns d {s : Signs d | s j = true}
      = uniformSigns d {s : Signs d | s j = false} := by
    calc uniformSigns d {s : Signs d | s j = true}
        = (uniformSigns d).map (flipEquiv τ) {s : Signs d | s j = true} := by rw [he]
      _ = uniformSigns d ((flipEquiv τ) ⁻¹' {s : Signs d | s j = true}) :=
          Measure.map_apply (measurable_of_countable _) hms
      _ = uniformSigns d {s : Signs d | s j = false} := by
          congr 1
          ext s
          simp only [Set.mem_preimage, Set.mem_setOf_eq, hflip s]
          cases s j <;> simp
  have htot := uniformSigns_eval_true_add_false j
  have h2 : (2:ℝ≥0∞) * uniformSigns d {s : Signs d | s j = true} = 1 := by
    rw [two_mul]
    nth_rewrite 2 [hswap]
    exact htot
  have h3 : (2:ℝ≥0∞) * (1 / 2) = 1 := by
    rw [one_div]; exact ENNReal.mul_inv_cancel two_ne_zero (by norm_num)
  exact (ENNReal.mul_right_inj two_ne_zero (by norm_num)).mp (h2.trans h3.symm)

theorem uniformSigns_eval_false (j : Fin d) :
    uniformSigns d {s : Signs d | s j = false} = 1 / 2 := by
  have h1 := uniformSigns_eval_true_add_false j
  have h2 := uniformSigns_eval_true j
  rw [h2] at h1
  have h3 : (1:ℝ≥0∞) / 2 + 1 / 2 = 1 := by
    rw [one_div, ← two_mul]; exact ENNReal.mul_inv_cancel two_ne_zero (by norm_num)
  exact (ENNReal.add_right_inj (by norm_num : (1:ℝ≥0∞) / 2 ≠ ⊤)).mp (h1.trans h3.symm)

/-! ### The pushforward itself -/

/-- **The coordinate law at a coordinate direction is the pushforward of `sgn` scaled by `Hᵢⱼ`**:
the two-atom law with mass `1/2` at `Hᵢⱼ` and `1/2` at `-Hᵢⱼ`.  Proved directly from
`coord_basis_apply` and the fibre masses above, by cases on membership of `Hᵢⱼ` and `-Hᵢⱼ` in an
arbitrary measurable set; `radLaw` is not used.  No hypothesis on `H` beyond `H ∈ O(d)`: this
holds in particular where `Hᵢⱼ = 0`, though flatness (below) rules that out. -/
theorem coordLaw_basis_eq_scaled_bernoulli (H : Od d) (i j : Fin d) :
    coordLaw H (EuclideanSpace.single j (1:ℝ)) i
      = (1 / 2 : ℝ≥0∞) • Measure.dirac ((H : Matrix (Fin d) (Fin d) ℝ) i j)
          + (1 / 2 : ℝ≥0∞) • Measure.dirac (-(H : Matrix (Fin d) (Fin d) ℝ) i j) := by
  set a : ℝ := (H : Matrix (Fin d) (Fin d) ℝ) i j with ha
  have hpt : ∀ s : Signs d, (hadRot H s • (EuclideanSpace.single j (1:ℝ) : Ed d)) i
      = a * sgn (s j) := fun s => coord_basis_apply H i j s
  apply Measure.ext
  intro S hS
  rw [coordLaw, Measure.map_apply (measurable_of_countable _) hS,
    Measure.add_apply, Measure.smul_apply, Measure.smul_apply,
    Measure.dirac_apply' _ hS, Measure.dirac_apply' _ hS]
  by_cases h1 : a ∈ S <;> by_cases h2 : -a ∈ S
  · have hpre : (fun s => (hadRot H s • (EuclideanSpace.single j (1:ℝ) : Ed d)) i) ⁻¹' S
        = Set.univ := by
      ext s
      simp only [Set.mem_preimage, hpt s, Set.mem_univ, iff_true]
      rcases Bool.eq_false_or_eq_true (s j) with hb | hb
      · rw [hb, sgn_true, mul_one]; exact h1
      · rw [hb, sgn_false, mul_neg_one]; exact h2
    rw [hpre, measure_univ, Set.indicator_of_mem h1, Set.indicator_of_mem h2]
    have h4 : (2:ℝ≥0∞)⁻¹ + 2⁻¹ = 1 := by
      rw [← two_mul]; exact ENNReal.mul_inv_cancel two_ne_zero (by norm_num)
    simp [h4]
  · have hpre : (fun s => (hadRot H s • (EuclideanSpace.single j (1:ℝ) : Ed d)) i) ⁻¹' S
        = {s : Signs d | s j = true} := by
      ext s
      simp only [Set.mem_preimage, hpt s, Set.mem_setOf_eq]
      rcases Bool.eq_false_or_eq_true (s j) with hb | hb
      · rw [hb, sgn_true, mul_one]; simpa using h1
      · rw [hb, sgn_false, mul_neg_one]; simpa using h2
    rw [hpre, uniformSigns_eval_true, Set.indicator_of_mem h1, Set.indicator_of_notMem h2]
    simp
  · have hpre : (fun s => (hadRot H s • (EuclideanSpace.single j (1:ℝ) : Ed d)) i) ⁻¹' S
        = {s : Signs d | s j = false} := by
      ext s
      simp only [Set.mem_preimage, hpt s, Set.mem_setOf_eq]
      rcases Bool.eq_false_or_eq_true (s j) with hb | hb
      · rw [hb, sgn_true, mul_one]; simpa using h1
      · rw [hb, sgn_false, mul_neg_one]; simpa using h2
    rw [hpre, uniformSigns_eval_false, Set.indicator_of_notMem h1, Set.indicator_of_mem h2]
    simp
  · have hpre : (fun s => (hadRot H s • (EuclideanSpace.single j (1:ℝ) : Ed d)) i) ⁻¹' S
        = ∅ := by
      ext s
      simp only [Set.mem_preimage, hpt s, Set.mem_empty_iff_false, iff_false]
      rcases Bool.eq_false_or_eq_true (s j) with hb | hb
      · rw [hb, sgn_true, mul_one]; exact h1
      · rw [hb, sgn_false, mul_neg_one]; exact h2
    rw [hpre, measure_empty, Set.indicator_of_notMem h1, Set.indicator_of_notMem h2]
    simp

/-- **At a coordinate direction, the deployed coordinate law is the two-point law on `±1/√d`.**
Flatness identifies `|Hᵢⱼ|` with `1/√d` (as in §48/§48b); which sign `Hᵢⱼ` itself carries is
immaterial, since swapping `Hᵢⱼ` for `-Hᵢⱼ` only swaps which dirac gets which coefficient, and
both coefficients are `1/2`.  This is the Phase-1 structural claim: `H D_ε eⱼ` deploys a genuine
two-atom law at `+1/√d` and `-1/√d`, each with mass exactly `1/2`, for **every** flat `H` and
**every** coordinate `i`, `j`.  Needs `IsFlat H`; needs no dimension hypothesis beyond
`i j : Fin d` existing, which already forces `d ≥ 1`. -/
theorem coordLaw_basis_eq_twoPoint {H : Od d} (hH : IsFlat H) (i j : Fin d) :
    coordLaw H (EuclideanSpace.single j (1:ℝ)) i
      = (1 / 2 : ℝ≥0∞) • Measure.dirac (1 / Real.sqrt d)
          + (1 / 2 : ℝ≥0∞) • Measure.dirac (-(1 / Real.sqrt d)) := by
  have habs : |(H : Matrix (Fin d) (Fin d) ℝ) i j| = 1 / Real.sqrt d := by
    have h := hH i j
    have h1 : |(H : Matrix (Fin d) (Fin d) ℝ) i j| = Real.sqrt (1 / (d : ℝ)) := by
      rw [← h, Real.sqrt_sq_eq_abs]
    rw [h1, one_div, Real.sqrt_inv, one_div]
  rcases (abs_eq (by positivity : (0:ℝ) ≤ 1 / Real.sqrt d)).mp habs with heq | heq
  · rw [← heq]; exact coordLaw_basis_eq_scaled_bernoulli H i j
  · have hbase := coordLaw_basis_eq_scaled_bernoulli H i j
    rw [heq, neg_neg] at hbase
    rw [hbase, add_comm]

end CoordLaw
