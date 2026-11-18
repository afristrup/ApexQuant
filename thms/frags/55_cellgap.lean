
/-! ## §54  The per-cell distortion gap of two laws, in CDF terms

§53 splits the distortion of a nearest-codeword map into the second moment of the law plus a sum
over cells of a term built from the cell masses and the cell first moments.  This section
subtracts two such decompositions.  Matched **second** moments cancel the second-moment term by
plain algebra, and each surviving cell contributes a quantity built only from the CDF difference
`cdf μ − cdf ν` at the two ends of the cell and from its once-integrated form `cdfGapInt`.  That
is the whole content here: it is an identity between two real numbers, and it is already sharper
than the cell decomposition it comes from, because a sup-norm bound on `cdf μ − cdf ν` applies to
its right-hand side directly.

Matched **means** are *not* assumed and are *not* used.  They belong to the Abel summation that
turns the sum below into a sum over thresholds, where they are what makes the two outer boundary
terms vanish; nothing in this section needs them.

The boundaries are packaged once.  A codebook with `K + 1` codewords has `K + 1` cells and
`K + 2` boundaries: `bdry cbk a b` sends index `0` to `a`, index `Fin.last (K + 1)` to `b`, and
index `k.succ.castSucc` to the threshold `thr cbk k`.  The single geometric fact that makes the
whole section uniform is `cell_inter_Ioc`: intersected with `(a, b]`, the `j`-th cell is exactly
`(bdry j.castSucc, bdry j.succ]`, the two outer cells included.  So masses become CDF differences
by `measureReal_Ioc_eq_cdf_sub`, and first moments become boundary terms minus a once-integrated
CDF difference by `cell_first_moment_gap`, which is proved here from the layer cake since Mathlib
has no Lebesgue or Stieltjes integration by parts.

Two hypotheses deserve comment, since neither is cosmetic.

* The support hypothesis is `μ (Ioc a b)ᶜ = 0`, half-open on the left, not `μ (Icc a b)ᶜ = 0`.
  It implies the closed form, and it is what makes `cdf μ a = 0` and `cdf μ b = 1`, so that both
  outer boundary terms are honest CDF values rather than a convention.  Any law with compact
  support satisfies it after moving `a` strictly below the support, which costs nothing: the two
  laws the paper compares live in `[−‖v‖, ‖v‖]` and `[−1, 1]`.
* `hthr` places every threshold inside `[a, b]`.  It is used to identify the two outer cells with
  `(a, thr 0]` and `(thr (K−1), b]`, and it is carried by every statement below that mentions a
  cell.  Whether it can be dropped is not settled here, in either direction.

No bound is proved in this section.  `abs_bdryGapInt_le` is the one estimate, and it bounds a
single once-integrated CDF difference, not any distortion gap. -/

namespace Threshold

variable {K : ℕ}

/-! ### The layer-cake step, and the first-moment gap on a bounded cell

Mathlib has no Lebesgue or Stieltjes integration by parts, so the one analytic step of the whole
identity is rebuilt from `Integrable.integral_eq_integral_meas_lt`. -/

/-- **The layer-cake atom.**  On a bounded interval the first moment of `t − a` against a finite
law is the integral of the upper-tail mass.  This is the substitute for an integration by parts,
and it is the only genuinely analytic step below. -/
theorem tail_atom (μ : Measure ℝ) [IsFiniteMeasure μ] {a b : ℝ} :
    ∫ t in Set.Ioc a b, (t - a) ∂μ = ∫ u in Set.Ioc a b, μ.real (Set.Ioc u b) := by
  have hmem : ∀ᵐ t ∂(μ.restrict (Set.Ioc a b)), t ∈ Set.Ioc a b :=
    ae_restrict_mem measurableSet_Ioc
  have hnn : 0 ≤ᵐ[μ.restrict (Set.Ioc a b)] (fun t : ℝ => t - a) := by
    filter_upwards [hmem] with t ht
    show (0 : ℝ) ≤ t - a
    linarith [ht.1.le]
  have hint : Integrable (fun t : ℝ => t - a) (μ.restrict (Set.Ioc a b)) :=
    (integrableOn_id_Ioc μ a b).sub integrableOn_const
  rw [hint.integral_eq_integral_meas_lt hnn]
  have hset : Set.EqOn (fun s : ℝ => (μ.restrict (Set.Ioc a b)).real {t : ℝ | s < t - a})
      (fun s : ℝ => μ.real (Set.Ioc (a + s) b)) (Set.Ioi 0) := by
    intro s hs
    have hs' : (0 : ℝ) < s := hs
    have h1 : {t : ℝ | s < t - a} = Set.Ioi (a + s) := by
      ext t; simp only [Set.mem_setOf_eq, Set.mem_Ioi]; constructor <;> intro h <;> linarith
    have h2 : Set.Ioi (a + s) ∩ Set.Ioc a b = Set.Ioc (a + s) b := by
      ext t
      simp only [Set.mem_inter_iff, Set.mem_Ioi, Set.mem_Ioc]
      exact ⟨fun h => ⟨h.1, h.2.2⟩, fun h => ⟨h.1, by linarith, h.2⟩⟩
    simp only [h1, measureReal_def, Measure.restrict_apply measurableSet_Ioi, h2]
  rw [setIntegral_congr_fun measurableSet_Ioi hset]
  have hshift : ∫ s in Set.Ioi (0 : ℝ), μ.real (Set.Ioc (a + s) b)
      = ∫ u in Set.Ioi a, μ.real (Set.Ioc u b) := by
    have hpre : (fun x : ℝ => a + x) ⁻¹' (Set.Ioi a) = Set.Ioi (0 : ℝ) := by
      ext x; simp only [Set.mem_preimage, Set.mem_Ioi]; constructor <;> intro h <;> linarith
    have h := (measurePreserving_add_left (volume : Measure ℝ) a).setIntegral_preimage_emb
      (measurableEmbedding_addLeft a) (fun u : ℝ => μ.real (Set.Ioc u b)) (Set.Ioi a)
    rw [hpre] at h
    exact h
  rw [hshift]
  refine setIntegral_eq_of_subset_of_ae_sdiff_eq_zero measurableSet_Ioi.nullMeasurableSet
    Set.Ioc_subset_Ioi_self (Filter.Eventually.of_forall ?_)
  rintro u ⟨hu1, hu2⟩
  have hub : b < u := by
    by_contra h
    exact hu2 ⟨hu1, not_lt.mp h⟩
  have he : Set.Ioc u b = (∅ : Set ℝ) := Set.Ioc_eq_empty (by linarith)
  simp [he]

/-- **The first-moment gap on a bounded cell.**  On `(a, b]` the first-moment difference of two
probability laws is a boundary term minus the once-integrated CDF difference.  Nothing is assumed
about the two laws beyond being probability measures: no codebook, no quantiser, no matched
moments, and no relation between `a`, `b` and where the laws live. -/
theorem cell_first_moment_gap (μ ν : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    {a b : ℝ} (hab : a ≤ b) :
    (∫ t in Set.Ioc a b, t ∂μ) - (∫ t in Set.Ioc a b, t ∂ν)
      = b * (cdf μ b - cdf ν b) - a * (cdf μ a - cdf ν a) - cdfGapInt μ ν a b := by
  have key : ∀ (ρ : Measure ℝ) (_ : IsProbabilityMeasure ρ),
      (∫ t in Set.Ioc a b, t ∂ρ)
        = a * (cdf ρ b - cdf ρ a) + (b - a) * cdf ρ b - ∫ u in Set.Ioc a b, (cdf ρ) u := by
    intro ρ hρ
    have h1 : ∫ t in Set.Ioc a b, (t - a) ∂ρ
        = (∫ t in Set.Ioc a b, t ∂ρ) - ρ.real (Set.Ioc a b) * a := by
      rw [integral_sub (integrableOn_id_Ioc ρ a b) integrableOn_const, setIntegral_const,
        smul_eq_mul]
    have h2 : ∫ u in Set.Ioc a b, ρ.real (Set.Ioc u b)
        = ∫ u in Set.Ioc a b, ((cdf ρ) b - (cdf ρ) u) :=
      setIntegral_congr_fun measurableSet_Ioc fun u hu => measureReal_Ioc_eq_cdf_sub ρ u b hu.2
    have h3 : ∫ u in Set.Ioc a b, ((cdf ρ) b - (cdf ρ) u)
        = (b - a) * cdf ρ b - ∫ u in Set.Ioc a b, (cdf ρ) u := by
      have hcb : IntegrableOn (fun _ : ℝ => (cdf ρ) b) (Set.Ioc a b) volume := by
        haveI : IsFiniteMeasure ((volume : Measure ℝ).restrict (Set.Ioc a b)) := by
          constructor
          rw [Measure.restrict_apply_univ, Real.volume_Ioc]
          exact ENNReal.ofReal_lt_top
        exact integrable_const _
      rw [integral_sub hcb (integrableOn_cdf_Ioc ρ a b), setIntegral_const, smul_eq_mul,
        Real.volume_real_Ioc, max_eq_left (by linarith)]
    have h4 := tail_atom ρ (a := a) (b := b)
    rw [h1, h2, h3] at h4
    rw [measureReal_Ioc_eq_cdf_sub ρ a b hab] at h4
    linarith
  have hμ := key μ ‹_›
  have hν := key ν ‹_›
  rw [hμ, hν, cdfGapInt_eq_sub]
  ring

/-! ### Additivity of the once-integrated CDF difference -/

/-- `cdfGapInt` is additive over adjoining intervals, which is what lets every cell be measured
from the single left endpoint `a`. -/
theorem cdfGapInt_add (μ ν : Measure ℝ) {a x y : ℝ} (hax : a ≤ x) (hxy : x ≤ y) :
    cdfGapInt μ ν a x + cdfGapInt μ ν x y = cdfGapInt μ ν a y := by
  have hdisj : Disjoint (Set.Ioc a x) (Set.Ioc x y) := by
    rw [Set.disjoint_left]
    rintro t ⟨_, h1⟩ ⟨h2, _⟩
    exact absurd h1 (not_le.mpr h2)
  simp only [cdfGapInt]
  rw [← setIntegral_union hdisj measurableSet_Ioc (integrableOn_cdfGap_Ioc μ ν a x)
    (integrableOn_cdfGap_Ioc μ ν x y), Set.Ioc_union_Ioc_eq_Ioc hax hxy]

/-! ### Support on a half-open interval -/

/-- A law supported in `(a, b]` is supported in `[a, b]`, so every statement of §52 and §53
applies to it. -/
theorem compl_Icc_eq_zero_of_Ioc {ρ : Measure ℝ} {a b : ℝ} (hsupp : ρ (Set.Ioc a b)ᶜ = 0) :
    ρ (Set.Icc a b)ᶜ = 0 :=
  measure_mono_null (Set.compl_subset_compl.mpr Set.Ioc_subset_Icc_self) hsupp

/-- Trimming a mass to a half-open support. -/
theorem measureReal_eq_of_supported_Ioc {ρ : Measure ℝ} {a b : ℝ}
    (hsupp : ρ (Set.Ioc a b)ᶜ = 0) (s : Set ℝ) : ρ.real s = ρ.real (s ∩ Set.Ioc a b) := by
  have hnull : ρ (s \ Set.Ioc a b) = 0 := measure_mono_null (fun _ hx => hx.2) hsupp
  have hadd := measure_inter_add_sdiff (μ := ρ) s (measurableSet_Ioc (a := a) (b := b))
  rw [hnull, add_zero] at hadd
  rw [measureReal_def, measureReal_def, hadd]

/-- Trimming an integral to a half-open support. -/
theorem setIntegral_eq_of_supported_Ioc {ρ : Measure ℝ} {a b : ℝ}
    (hsupp : ρ (Set.Ioc a b)ᶜ = 0) (f : ℝ → ℝ) (s : Set ℝ) :
    ∫ t in s, f t ∂ρ = ∫ t in s ∩ Set.Ioc a b, f t ∂ρ := by
  have key : ρ.restrict s = ρ.restrict (s ∩ Set.Ioc a b) := by
    refine Measure.restrict_congr_set ?_
    rw [Filter.eventuallyEq_set]
    filter_upwards [ae_iff.mpr hsupp] with x hx
    simp [hx]
  rw [key]

/-- **The left endpoint carries no mass below it.**  A law supported in `(a, b]` has `cdf` zero
at `a`; this is the point of taking the support half-open on the left. -/
theorem cdf_eq_zero_of_supported_Ioc {ρ : Measure ℝ} [IsProbabilityMeasure ρ] {a b : ℝ}
    (hsupp : ρ (Set.Ioc a b)ᶜ = 0) : cdf ρ a = 0 := by
  have hsub : Set.Iic a ⊆ (Set.Ioc a b)ᶜ := by
    intro t ht hc
    exact absurd hc.1 (not_lt.mpr ht)
  have h : ρ (Set.Iic a) = 0 := measure_mono_null hsub hsupp
  rw [cdf_eq_real, measureReal_def, h, ENNReal.toReal_zero]

/-- A law supported in `(a, b]` has `cdf` one at `b`. -/
theorem cdf_eq_one_of_supported_Ioc {ρ : Measure ℝ} [IsProbabilityMeasure ρ] {a b : ℝ}
    (hsupp : ρ (Set.Ioc a b)ᶜ = 0) : cdf ρ b = 1 := by
  have hsub : (Set.Iic b)ᶜ ⊆ (Set.Ioc a b)ᶜ := by
    intro t ht hc
    exact ht hc.2
  have h0 : ρ (Set.Iic b)ᶜ = 0 := measure_mono_null hsub hsupp
  have h1 : ρ (Set.Iic b) = ρ Set.univ := by
    refine le_antisymm (measure_mono (Set.subset_univ _)) ?_
    calc ρ Set.univ = ρ (Set.Iic b ∪ (Set.Iic b)ᶜ) := by rw [Set.union_compl_self]
      _ ≤ ρ (Set.Iic b) + ρ (Set.Iic b)ᶜ := measure_union_le _ _
      _ = ρ (Set.Iic b) := by rw [h0, add_zero]
  rw [cdf_eq_real, measureReal_def, h1, measure_univ, ENNReal.toReal_one]

/-! ### The `K + 2` cell boundaries, as one function -/

/-- The thresholds of a codebook, extended to every natural index by the right endpoint `b`.
This is a bookkeeping device: it exists so that `bdry` is a total function on `Fin (K + 2)`. -/
noncomputable def thrExt (cbk : Codebook K) (b : ℝ) (m : ℕ) : ℝ :=
  if h : m < K then thr cbk ⟨m, h⟩ else b

theorem thrExt_of_lt (cbk : Codebook K) (b : ℝ) {m : ℕ} (h : m < K) :
    thrExt cbk b m = thr cbk ⟨m, h⟩ := dif_pos h

theorem thrExt_of_not_lt (cbk : Codebook K) (b : ℝ) {m : ℕ} (h : ¬ m < K) :
    thrExt cbk b m = b := dif_neg h

/-- **The cell boundaries.**  A codebook with `K + 1` codewords has `K + 2` boundaries: the left
endpoint `a` at index `0`, the right endpoint `b` at index `Fin.last (K + 1)`, and the `K`
thresholds in between, the `k`-th at index `k.succ.castSucc`.  Cell `j` runs from boundary
`j.castSucc` to boundary `j.succ`. -/
noncomputable def bdry (cbk : Codebook K) (a b : ℝ) (i : Fin (K + 2)) : ℝ :=
  if (i : ℕ) = 0 then a else thrExt cbk b ((i : ℕ) - 1)

theorem bdry_zero (cbk : Codebook K) (a b : ℝ) : bdry cbk a b 0 = a := by
  simp only [bdry, Fin.val_zero, if_pos]

theorem bdry_last (cbk : Codebook K) (a b : ℝ) : bdry cbk a b (Fin.last (K + 1)) = b := by
  have hv : ((Fin.last (K + 1) : Fin (K + 2)) : ℕ) = K + 1 := Fin.val_last _
  simp only [bdry, hv]
  rw [if_neg (by omega), Nat.add_sub_cancel]
  exact thrExt_of_not_lt cbk b (lt_irrefl K)

/-- The interior boundaries are the thresholds. -/
theorem bdry_succ_castSucc (cbk : Codebook K) (a b : ℝ) (k : Fin K) :
    bdry cbk a b k.succ.castSucc = thr cbk k := by
  have hv : ((k.succ.castSucc : Fin (K + 2)) : ℕ) = (k : ℕ) + 1 := by
    rw [Fin.val_castSucc, Fin.val_succ]
  simp only [bdry, hv]
  rw [if_neg (by omega), Nat.add_sub_cancel, thrExt_of_lt cbk b k.isLt]

/-- **The midpoint property, in the shape the Abel step wants.**  Every interior boundary is the
midpoint of the two codewords flanking it; this is the definition of `thr` and nothing more. -/
theorem c_add_c_eq_two_mul_bdry (cbk : Codebook K) (a b : ℝ) (k : Fin K) :
    cbk.c k.castSucc + cbk.c k.succ = 2 * bdry cbk a b k.succ.castSucc := by
  rw [bdry_succ_castSucc, thr]
  ring

theorem bdry_castSucc_val (cbk : Codebook K) (a b : ℝ) (j : Fin (K + 1)) :
    bdry cbk a b j.castSucc
      = if (j : ℕ) = 0 then a else thrExt cbk b ((j : ℕ) - 1) := by
  simp only [bdry, Fin.val_castSucc]

theorem bdry_succ_val (cbk : Codebook K) (a b : ℝ) (j : Fin (K + 1)) :
    bdry cbk a b j.succ = thrExt cbk b (j : ℕ) := by
  simp only [bdry, Fin.val_succ]
  rw [if_neg (by omega), Nat.add_sub_cancel]

/-! ### The boundaries lie in the interval, and increase along a cell -/

variable {a b : ℝ}

theorem le_thrExt (cbk : Codebook K) (hab : a ≤ b)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) (m : ℕ) : a ≤ thrExt cbk b m := by
  by_cases h : m < K
  · rw [thrExt_of_lt cbk b h]; exact (hthr ⟨m, h⟩).1
  · rw [thrExt_of_not_lt cbk b h]; exact hab

theorem thrExt_le (cbk : Codebook K)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) (m : ℕ) : thrExt cbk b m ≤ b := by
  by_cases h : m < K
  · rw [thrExt_of_lt cbk b h]; exact (hthr ⟨m, h⟩).2
  · rw [thrExt_of_not_lt cbk b h]

theorem le_bdry (cbk : Codebook K) (hab : a ≤ b)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) (i : Fin (K + 2)) :
    a ≤ bdry cbk a b i := by
  simp only [bdry]
  split_ifs
  · exact le_rfl
  · exact le_thrExt cbk hab hthr _

theorem bdry_le (cbk : Codebook K) (hab : a ≤ b)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) (i : Fin (K + 2)) :
    bdry cbk a b i ≤ b := by
  simp only [bdry]
  split_ifs
  · exact hab
  · exact thrExt_le cbk hthr _

/-- A cell's left boundary is at or below its right boundary. -/
theorem bdry_castSucc_le_bdry_succ (cbk : Codebook K) (hab : a ≤ b)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) (j : Fin (K + 1)) :
    bdry cbk a b j.castSucc ≤ bdry cbk a b j.succ := by
  rw [bdry_castSucc_val, bdry_succ_val]
  by_cases hj : (j : ℕ) = 0
  · rw [if_pos hj]
    exact le_thrExt cbk hab hthr _
  · rw [if_neg hj]
    by_cases hK : (j : ℕ) < K
    · have hp : (j : ℕ) - 1 < K := by omega
      rw [thrExt_of_lt cbk b hp, thrExt_of_lt cbk b hK]
      exact (thr_strictMono cbk).monotone (Fin.le_def.mpr (by show (j : ℕ) - 1 ≤ (j : ℕ); omega))
    · rw [thrExt_of_not_lt cbk b hK]
      exact thrExt_le cbk hthr _

/-! ### The one geometric fact: a cell, trimmed, is the interval between its boundaries -/

/-- **Every cell, trimmed to `(a, b]`, is the half-open interval between its two boundaries.**
This holds for the two outer cells, which are unbounded rays, as well as for the interior ones,
and it is the single statement that makes the rest of the section uniform in `j`. -/
theorem cell_inter_Ioc (cbk : Codebook K) (hab : a ≤ b)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) (j : Fin (K + 1)) :
    cell cbk j ∩ Set.Ioc a b = Set.Ioc (bdry cbk a b j.castSucc) (bdry cbk a b j.succ) := by
  ext t
  simp only [Set.mem_inter_iff, mem_cell_iff, Set.mem_Ioc, bdry_castSucc_val, bdry_succ_val]
  constructor
  · rintro ⟨⟨h1, h2⟩, hta, htb⟩
    constructor
    · by_cases hj : (j : ℕ) = 0
      · rw [if_pos hj]; exact hta
      · rw [if_neg hj]
        have hp : (j : ℕ) - 1 < K := by have := j.isLt; omega
        rw [thrExt_of_lt cbk b hp]
        exact h1 ⟨(j : ℕ) - 1, hp⟩ (by show (j : ℕ) - 1 < (j : ℕ); omega)
    · by_cases hK : (j : ℕ) < K
      · rw [thrExt_of_lt cbk b hK]
        exact h2 ⟨(j : ℕ), hK⟩ (by show (j : ℕ) ≤ (j : ℕ); omega)
      · rw [thrExt_of_not_lt cbk b hK]; exact htb
  · rintro ⟨hlo, hhi⟩
    have hAle : a ≤ (if (j : ℕ) = 0 then a else thrExt cbk b ((j : ℕ) - 1)) := by
      split_ifs
      · exact le_rfl
      · exact le_thrExt cbk hab hthr _
    have hBle : thrExt cbk b (j : ℕ) ≤ b := thrExt_le cbk hthr _
    refine ⟨⟨fun k hk => ?_, fun k hk => ?_⟩, lt_of_le_of_lt hAle hlo, le_trans hhi hBle⟩
    · have hj : ¬ ((j : ℕ) = 0) := by omega
      rw [if_neg hj] at hlo
      have hp : (j : ℕ) - 1 < K := by have := j.isLt; omega
      rw [thrExt_of_lt cbk b hp] at hlo
      refine lt_of_le_of_lt ?_ hlo
      exact (thr_strictMono cbk).monotone
        (Fin.le_def.mpr (by show (k : ℕ) ≤ (j : ℕ) - 1; omega))
    · have hK : (j : ℕ) < K := lt_of_le_of_lt hk k.isLt
      rw [thrExt_of_lt cbk b hK] at hhi
      exact le_trans hhi ((thr_strictMono cbk).monotone
        (Fin.le_def.mpr (by show (j : ℕ) ≤ (k : ℕ); omega)))

/-! ### The boundary-indexed CDF gap and its once-integrated form -/

/-- The CDF difference of the two laws at a cell boundary. -/
noncomputable def bdryCdfGap (cbk : Codebook K) (μ ν : Measure ℝ) (a b : ℝ) (i : Fin (K + 2)) :
    ℝ := cdf μ (bdry cbk a b i) - cdf ν (bdry cbk a b i)

/-- The once-integrated CDF difference from `a` up to a cell boundary. -/
noncomputable def bdryGapInt (cbk : Codebook K) (μ ν : Measure ℝ) (a b : ℝ) (i : Fin (K + 2)) :
    ℝ := cdfGapInt μ ν a (bdry cbk a b i)

/-- At the left boundary the CDF gap vanishes, both laws being supported in `(a, b]`. -/
theorem bdryCdfGap_zero (cbk : Codebook K) {μ ν : Measure ℝ} [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0) :
    bdryCdfGap cbk μ ν a b 0 = 0 := by
  rw [bdryCdfGap, bdry_zero, cdf_eq_zero_of_supported_Ioc hsupp_mu,
    cdf_eq_zero_of_supported_Ioc hsupp_nu, sub_self]

/-- At the right boundary the CDF gap vanishes, both laws being probability measures supported in
`(a, b]`. -/
theorem bdryCdfGap_last (cbk : Codebook K) {μ ν : Measure ℝ} [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0) :
    bdryCdfGap cbk μ ν a b (Fin.last (K + 1)) = 0 := by
  rw [bdryCdfGap, bdry_last, cdf_eq_one_of_supported_Ioc hsupp_mu,
    cdf_eq_one_of_supported_Ioc hsupp_nu, sub_self]

/-- The once-integrated gap is zero at the left boundary, the interval being empty. -/
theorem bdryGapInt_zero (cbk : Codebook K) (μ ν : Measure ℝ) (a b : ℝ) :
    bdryGapInt cbk μ ν a b 0 = 0 := by
  rw [bdryGapInt, bdry_zero, cdfGapInt]
  simp

/-- The interior values of the two boundary-indexed functions are the ones at the thresholds. -/
theorem bdryCdfGap_succ_castSucc (cbk : Codebook K) (μ ν : Measure ℝ) (a b : ℝ) (k : Fin K) :
    bdryCdfGap cbk μ ν a b k.succ.castSucc = cdf μ (thr cbk k) - cdf ν (thr cbk k) := by
  rw [bdryCdfGap, bdry_succ_castSucc]

theorem bdryGapInt_succ_castSucc (cbk : Codebook K) (μ ν : Measure ℝ) (a b : ℝ) (k : Fin K) :
    bdryGapInt cbk μ ν a b k.succ.castSucc = cdfGapInt μ ν a (thr cbk k) := by
  rw [bdryGapInt, bdry_succ_castSucc]

/-- **The only estimate here.**  A sup-norm bound `η` on the CDF difference bounds every
boundary-indexed once-integrated gap by `η · (b − a)`.  This bounds one term of the identity
below; it bounds no distortion gap. -/
theorem abs_bdryGapInt_le (cbk : Codebook K) (μ ν : Measure ℝ) {η : ℝ} (hab : a ≤ b)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hks : ∀ u, |(cdf μ) u - (cdf ν) u| ≤ η) (i : Fin (K + 2)) :
    |bdryGapInt cbk μ ν a b i| ≤ η * (b - a) := by
  have hη : 0 ≤ η := le_trans (abs_nonneg _) (hks a)
  have h3 : a ≤ bdry cbk a b i := le_bdry cbk hab hthr i
  have h2 : bdry cbk a b i ≤ b := bdry_le cbk hab hthr i
  have h1 : |cdfGapInt μ ν a (bdry cbk a b i)| ≤ η * (bdry cbk a b i - a) :=
    abs_cdfGapInt_le μ ν h3 hks
  rw [bdryGapInt]
  refine le_trans h1 ?_
  nlinarith

/-! ### The mass and the first moment of a cell, as CDF quantities -/

/-- **A cell's mass gap is a CDF difference at its two boundaries.** -/
theorem cellMass_sub_cellMass (cbk : Codebook K) {μ ν : Measure ℝ} [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (hab : a ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0) (j : Fin (K + 1)) :
    cellMass cbk μ j - cellMass cbk ν j
      = bdryCdfGap cbk μ ν a b j.succ - bdryCdfGap cbk μ ν a b j.castSucc := by
  have hle := bdry_castSucc_le_bdry_succ cbk hab hthr j
  have hm : ∀ (ρ : Measure ℝ), IsProbabilityMeasure ρ → ρ (Set.Ioc a b)ᶜ = 0 →
      cellMass cbk ρ j = cdf ρ (bdry cbk a b j.succ) - cdf ρ (bdry cbk a b j.castSucc) := by
    intro ρ hρ hs
    rw [cellMass, measureReal_eq_of_supported_Ioc hs (cell cbk j), cell_inter_Ioc cbk hab hthr j,
      measureReal_Ioc_eq_cdf_sub ρ _ _ hle]
  rw [hm μ ‹_› hsupp_mu, hm ν ‹_› hsupp_nu, bdryCdfGap, bdryCdfGap]
  ring

/-- **A cell's first-moment gap is a boundary term minus a once-integrated CDF difference.**  This
is `cell_first_moment_gap` on the trimmed cell, with the once-integrated term measured from the
common left endpoint `a` by `cdfGapInt_add`. -/
theorem cellMoment_sub_cellMoment (cbk : Codebook K) {μ ν : Measure ℝ} [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (hab : a ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0) (j : Fin (K + 1)) :
    cellMoment cbk μ j - cellMoment cbk ν j
      = bdry cbk a b j.succ * bdryCdfGap cbk μ ν a b j.succ
        - bdry cbk a b j.castSucc * bdryCdfGap cbk μ ν a b j.castSucc
        - (bdryGapInt cbk μ ν a b j.succ - bdryGapInt cbk μ ν a b j.castSucc) := by
  have hle := bdry_castSucc_le_bdry_succ cbk hab hthr j
  have hstart := le_bdry cbk hab hthr j.castSucc
  have hmμ : cellMoment cbk μ j = ∫ t in Set.Ioc (bdry cbk a b j.castSucc)
      (bdry cbk a b j.succ), t ∂μ := by
    rw [cellMoment, setIntegral_eq_of_supported_Ioc hsupp_mu (fun t : ℝ => t) (cell cbk j),
      cell_inter_Ioc cbk hab hthr j]
  have hmν : cellMoment cbk ν j = ∫ t in Set.Ioc (bdry cbk a b j.castSucc)
      (bdry cbk a b j.succ), t ∂ν := by
    rw [cellMoment, setIntegral_eq_of_supported_Ioc hsupp_nu (fun t : ℝ => t) (cell cbk j),
      cell_inter_Ioc cbk hab hthr j]
  have hgap := cell_first_moment_gap μ ν hle
  have hadd := cdfGapInt_add μ ν hstart hle
  rw [hmμ, hmν, hgap, bdryCdfGap, bdryCdfGap, bdryGapInt, bdryGapInt]
  linarith

/-! ### The per-cell gap -/

/-- **The per-cell distortion gap.**  For two probability laws supported in `(a, b]` with *equal
second moments*, and any nearest-codeword map of an ordered codebook whose thresholds lie in
`[a, b]`, the distortion gap is a sum over the `K + 1` cells of a term built only from the CDF
difference at the two cell boundaries and from its once-integrated form.

The second moments cancel by plain algebra; that is the only use of `hsq`.  Equal **means** are
neither assumed nor used here.  No bound is claimed: this is an identity between two real numbers,
and it is the summand of §50's `sum_cell_gap_eq_boundary_sub_threshold`, so the Abel step can be
applied to it verbatim with `c := cbk.c`, `x := bdry cbk a b`, `F := bdryCdfGap cbk μ ν a b` and
`G := bdryGapInt cbk μ ν a b`. -/
theorem distortion_sub_distortion_eq_cellGap (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (hab : a ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν) :
    distortion q μ - distortion q ν
      = ∑ j : Fin (K + 1),
          (cbk.c j ^ 2
              * (bdryCdfGap cbk μ ν a b j.succ - bdryCdfGap cbk μ ν a b j.castSucc)
            - 2 * cbk.c j
              * (bdry cbk a b j.succ * bdryCdfGap cbk μ ν a b j.succ
                  - bdry cbk a b j.castSucc * bdryCdfGap cbk μ ν a b j.castSucc
                  - (bdryGapInt cbk μ ν a b j.succ
                      - bdryGapInt cbk μ ν a b j.castSucc))) := by
  have hIccμ : μ (Set.Icc a b)ᶜ = 0 := compl_Icc_eq_zero_of_Ioc hsupp_mu
  have hIccν : ν (Set.Icc a b)ᶜ = 0 := compl_Icc_eq_zero_of_Ioc hsupp_nu
  have hμ := distortion_isNearest_eq_sq_add_cross hq μ hIccμ
  have hν := distortion_isNearest_eq_sq_add_cross hq ν hIccν
  have hcancel : ∀ X S T : ℝ, (X + S) - (X + T) = S - T := fun X S T => by ring
  rw [hμ, hν, hsq, hcancel, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  have h1 := cellMass_sub_cellMass cbk hab hthr hsupp_mu hsupp_nu j
  have h2 := cellMoment_sub_cellMoment cbk hab hthr hsupp_mu hsupp_nu j
  linear_combination (cbk.c j ^ 2) * h1 - 2 * cbk.c j * h2

end Threshold
