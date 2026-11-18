
/-! ## §53  The cells of a scalar codebook, and the distortion integral over them

The threshold form of the distortion gap is reached by splitting the distortion integral over the
`K + 1` cells of the codebook, one per codeword, and treating each cell separately.  This section
performs that split and nothing else: it proves no gap identity, compares no two laws, and
assumes nothing about matched moments.

The cell of a codeword index is the level set `cell cbk j = {t | cellIdx cbk t = j}`, so the cells
are pairwise disjoint and cover the line by construction.  `cell_eq_iInter` writes each of them as
a finite intersection of rays cut at the thresholds, which is where measurability comes from, and
`cell_castSucc_eq_Ioc`, `cell_zero_eq_Iic` and `cell_last_eq_Ioi` identify them concretely: the
interior cells are half-open to the right, and the two outer cells are unbounded.  That the outer
cells are unbounded is the reason the support hypothesis is carried everywhere below; a bounded
cell would not need it.

`integral_eq_sum_cell` is the spine, `MeasureTheory.integral_iUnion` in its countable `tsum` form
collapsed by `tsum_fintype`.  It needs the integrand to be integrable against the whole law, not
merely on each cell, which is what the support hypothesis supplies through §52.

Two consequences are recorded, in the two shapes a later stage wants.
`distortion_stepQuant_eq_sum` is the raw cell decomposition, with each cell trimmed to `Icc a b`;
`distortion_eq_sq_add_cross` is the split of the square, `(t - c)² = t² - 2ct + c²`, in which the
`t²` term is a single law-level integral that matched second moments will cancel, and everything
that depends on the codebook is carried by `cellMass` and `cellMoment`, the mass and the first
moment of a cell.  Both are then transported from `stepQuant` to an arbitrary nearest-codeword map
by §52's `distortion_congr_of_isNearest`, so no tie convention appears in any statement.

`cellMass` and `cellMoment` are defined against the untrimmed cell.  They are real numbers, not
junk values, because the law is finite and compactly supported; `cellMass_eq_inter` and
`cellMoment_eq_inter` say the trimming does not change them.  No claim is made here that any cell
carries positive mass, and nothing here expresses `cellMass` or `cellMoment` in terms of a CDF. -/

namespace Threshold

variable {K : ℕ}

/-! ### The cells of a codebook -/

/-- The cell of a codeword index: the scalars that `cellIdx` sends to `j`.  As a level set of
`cellIdx`, the cells are pairwise disjoint and cover the line by construction. -/
noncomputable def cell (cbk : Codebook K) (j : Fin (K + 1)) : Set ℝ :=
  {t : ℝ | cellIdx cbk t = j}

theorem mem_cell (cbk : Codebook K) (j : Fin (K + 1)) (t : ℝ) :
    t ∈ cell cbk j ↔ cellIdx cbk t = j := Iff.rfl

/-- `stepQuant` is constant on each cell, with the value that names the cell. -/
theorem stepQuant_eq_of_mem_cell (cbk : Codebook K) {j : Fin (K + 1)} {t : ℝ}
    (ht : t ∈ cell cbk j) : stepQuant cbk t = cbk.c j := by
  rw [stepQuant, (mem_cell cbk j t).mp ht]

/-- **Membership in a cell, threshold by threshold.**  `t` lies in cell `j` exactly when it is
strictly above every threshold of index below `j` and at or below every threshold of index at
least `j`.  This is the counting lemma `thr_lt_iff_lt_cellIdx` read twice. -/
theorem mem_cell_iff (cbk : Codebook K) (j : Fin (K + 1)) (t : ℝ) :
    t ∈ cell cbk j ↔ (∀ k : Fin K, (k : ℕ) < (j : ℕ) → thr cbk k < t)
      ∧ (∀ k : Fin K, (j : ℕ) ≤ (k : ℕ) → t ≤ thr cbk k) := by
  rw [mem_cell]
  constructor
  · rintro rfl
    refine ⟨fun k hk => (thr_lt_iff_lt_cellIdx cbk t k).mpr hk, fun k hk => ?_⟩
    by_contra hcon
    exact absurd ((thr_lt_iff_lt_cellIdx cbk t k).mp (not_le.mp hcon)) (not_lt.mpr hk)
  · rintro ⟨h1, h2⟩
    refine Fin.ext ?_
    rcases lt_trichotomy ((cellIdx cbk t : ℕ)) ((j : ℕ)) with h | h | h
    · exfalso
      have hK : (cellIdx cbk t : ℕ) < K := lt_of_lt_of_le h (Nat.lt_succ_iff.mp j.isLt)
      have hlt : thr cbk ⟨(cellIdx cbk t : ℕ), hK⟩ < t := h1 ⟨(cellIdx cbk t : ℕ), hK⟩ h
      have hcount := (thr_lt_iff_lt_cellIdx cbk t ⟨(cellIdx cbk t : ℕ), hK⟩).mp hlt
      simp only at hcount
      omega
    · exact h
    · exfalso
      have hK : (j : ℕ) < K := lt_of_lt_of_le h (Nat.lt_succ_iff.mp (cellIdx cbk t).isLt)
      have hle : t ≤ thr cbk ⟨(j : ℕ), hK⟩ := h2 ⟨(j : ℕ), hK⟩ (le_refl _)
      have hlt : thr cbk ⟨(j : ℕ), hK⟩ < t :=
        (thr_lt_iff_lt_cellIdx cbk t ⟨(j : ℕ), hK⟩).mpr h
      linarith

/-- **Each cell is a finite intersection of rays cut at the thresholds.**  This is the form
measurability is read off, and it is the only place the cells are described without reference to
`cellIdx`. -/
theorem cell_eq_iInter (cbk : Codebook K) (j : Fin (K + 1)) :
    cell cbk j = ⋂ k : Fin K,
      (if (k : ℕ) < (j : ℕ) then Set.Ioi (thr cbk k) else Set.Iic (thr cbk k)) := by
  ext t
  rw [mem_cell_iff, Set.mem_iInter]
  constructor
  · rintro ⟨h1, h2⟩ k
    by_cases hk : (k : ℕ) < (j : ℕ)
    · rw [if_pos hk]; exact h1 k hk
    · rw [if_neg hk]; exact h2 k (not_lt.mp hk)
  · intro h
    refine ⟨fun k hk => ?_, fun k hk => ?_⟩
    · have hmem := h k
      rw [if_pos hk] at hmem
      exact hmem
    · have hmem := h k
      rw [if_neg (not_lt.mpr hk)] at hmem
      exact hmem

/-- Every cell is measurable. -/
theorem measurableSet_cell (cbk : Codebook K) (j : Fin (K + 1)) :
    MeasurableSet (cell cbk j) := by
  rw [cell_eq_iInter]
  refine MeasurableSet.iInter fun k => ?_
  by_cases hk : (k : ℕ) < (j : ℕ)
  · rw [if_pos hk]; exact measurableSet_Ioi
  · rw [if_neg hk]; exact measurableSet_Iic

/-- The cells cover the line: every scalar lies in the cell of its own index. -/
theorem iUnion_cell (cbk : Codebook K) : (⋃ j : Fin (K + 1), cell cbk j) = Set.univ := by
  ext t
  simp only [Set.mem_iUnion, Set.mem_univ, iff_true]
  exact ⟨cellIdx cbk t, rfl⟩

/-- The cells are pairwise disjoint, being distinct level sets of one function. -/
theorem pairwise_disjoint_cell (cbk : Codebook K) :
    Pairwise (Function.onFun Disjoint (cell cbk)) := by
  intro i j hij
  rw [Function.onFun, Set.disjoint_left]
  intro t hi hj
  exact hij (((mem_cell cbk i t).mp hi).symm.trans ((mem_cell cbk j t).mp hj))

/-! ### The cells, concretely -/

/-- **An interior cell is half-open to the right.**  For consecutive threshold indices `i` and
`j`, the cell of the codeword `c j.castSucc` is `(thr i, thr j]`. -/
theorem cell_castSucc_eq_Ioc (cbk : Codebook K) {i j : Fin K} (hij : (i : ℕ) + 1 = (j : ℕ)) :
    cell cbk j.castSucc = Set.Ioc (thr cbk i) (thr cbk j) := by
  ext t
  rw [mem_cell_iff, Set.mem_Ioc, Fin.val_castSucc]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨h1 i (by omega), h2 j (by omega)⟩
  · rintro ⟨h1, h2⟩
    refine ⟨fun k hk => ?_, fun k hk => ?_⟩
    · have hle : thr cbk k ≤ thr cbk i :=
        (thr_strictMono cbk).monotone (Fin.le_def.mpr (by omega))
      linarith
    · have hle : thr cbk j ≤ thr cbk k :=
        (thr_strictMono cbk).monotone (Fin.le_def.mpr (by omega))
      linarith

/-- **The lowest cell is an unbounded left ray.**  When `i` is the first threshold index, the cell
of the smallest codeword is `(-∞, thr i]`. -/
theorem cell_zero_eq_Iic (cbk : Codebook K) {i : Fin K} (hi : (i : ℕ) = 0) :
    cell cbk 0 = Set.Iic (thr cbk i) := by
  ext t
  rw [mem_cell_iff, Set.mem_Iic]
  constructor
  · rintro ⟨_, h2⟩
    exact h2 i (by simp)
  · intro h
    refine ⟨fun k hk => by simp at hk, fun k _ => ?_⟩
    have hle : thr cbk i ≤ thr cbk k :=
      (thr_strictMono cbk).monotone (Fin.le_def.mpr (by omega))
    linarith

/-- **The highest cell is an unbounded right ray.**  When `i` is the last threshold index, the cell
of the largest codeword is `(thr i, ∞)`. -/
theorem cell_last_eq_Ioi (cbk : Codebook K) {i : Fin K} (hi : (i : ℕ) + 1 = K) :
    cell cbk (Fin.last K) = Set.Ioi (thr cbk i) := by
  ext t
  rw [mem_cell_iff, Set.mem_Ioi, Fin.val_last]
  constructor
  · rintro ⟨h1, _⟩
    exact h1 i (by omega)
  · intro h
    refine ⟨fun k _ => ?_, fun k hk => ?_⟩
    · have hle : thr cbk k ≤ thr cbk i :=
        (thr_strictMono cbk).monotone (Fin.le_def.mpr (by omega))
      linarith
    · exact absurd hk (by have := k.isLt; omega)

/-- With no thresholds at all there is a single cell, the whole line. -/
theorem cell_of_zero (cbk : Codebook 0) : cell cbk 0 = Set.univ := by
  ext t
  rw [mem_cell_iff]
  exact ⟨fun _ => trivial, fun _ => ⟨fun k => k.elim0, fun k => k.elim0⟩⟩

/-! ### Trimming a cell to the support -/

/-- **Trimming a mass to the support.**  When the law gives the complement of `Icc a b` no mass,
intersecting any set with `Icc a b` does not change its mass. -/
theorem measureReal_eq_of_supported {ρ : Measure ℝ} {a b : ℝ} (hsupp : ρ (Set.Icc a b)ᶜ = 0)
    (s : Set ℝ) : ρ.real s = ρ.real (s ∩ Set.Icc a b) := by
  have hnull : ρ (s \ Set.Icc a b) = 0 := measure_mono_null (fun _ hx => hx.2) hsupp
  have hadd := measure_inter_add_sdiff (μ := ρ) s (measurableSet_Icc (a := a) (b := b))
  rw [hnull, add_zero] at hadd
  rw [measureReal_def, measureReal_def, hadd]

/-! ### The mass and the first moment of a cell -/

/-- The mass a law puts on a cell.  A real number, not `ENNReal`; against a finite law it is
finite, and no positivity beyond nonnegativity is claimed. -/
noncomputable def cellMass (cbk : Codebook K) (ρ : Measure ℝ) (j : Fin (K + 1)) : ℝ :=
  ρ.real (cell cbk j)

/-- The first moment a law puts on a cell.  Against a finite law supported in a bounded interval
this is a real number, by §52's `integrable_id_of_supported`; with no support hypothesis it may be
`integral`'s junk value, and nothing here says otherwise. -/
noncomputable def cellMoment (cbk : Codebook K) (ρ : Measure ℝ) (j : Fin (K + 1)) : ℝ :=
  ∫ t in cell cbk j, t ∂ρ

theorem cellMass_nonneg (cbk : Codebook K) (ρ : Measure ℝ) (j : Fin (K + 1)) :
    0 ≤ cellMass cbk ρ j := measureReal_nonneg

/-- The mass of a cell is the mass of the part of it inside the support. -/
theorem cellMass_eq_inter (cbk : Codebook K) {ρ : Measure ℝ} {a b : ℝ}
    (hsupp : ρ (Set.Icc a b)ᶜ = 0) (j : Fin (K + 1)) :
    cellMass cbk ρ j = ρ.real (cell cbk j ∩ Set.Icc a b) :=
  measureReal_eq_of_supported hsupp _

/-- The first moment of a cell is the first moment of the part of it inside the support. -/
theorem cellMoment_eq_inter (cbk : Codebook K) {ρ : Measure ℝ} {a b : ℝ}
    (hsupp : ρ (Set.Icc a b)ᶜ = 0) (j : Fin (K + 1)) :
    cellMoment cbk ρ j = ∫ t in cell cbk j ∩ Set.Icc a b, t ∂ρ :=
  setIntegral_eq_of_supported hsupp _ _

/-! ### The spine: an integral is the sum of its integrals over the cells -/

/-- **The cell decomposition of an integral.**  For an integrable `f`, the integral against a law
is the finite sum of the integrals over the cells.  This is `MeasureTheory.integral_iUnion` in its
countable `tsum` form, collapsed by `tsum_fintype`; the cells being disjoint and measurable and
covering the line is all that is used.  Integrability against the whole law, not merely on each
cell, is genuinely needed and is what the support hypothesis buys elsewhere. -/
theorem integral_eq_sum_cell (cbk : Codebook K) (ρ : Measure ℝ) {f : ℝ → ℝ}
    (hf : Integrable f ρ) :
    ∫ t, f t ∂ρ = ∑ j : Fin (K + 1), ∫ t in cell cbk j, f t ∂ρ := by
  have hu : ∫ t, f t ∂ρ = ∫ t in (⋃ j : Fin (K + 1), cell cbk j), f t ∂ρ := by
    rw [iUnion_cell, setIntegral_univ]
  rw [hu, integral_iUnion (measurableSet_cell cbk) (pairwise_disjoint_cell cbk)
    (by rw [iUnion_cell]; exact hf.integrableOn), tsum_fintype]

/-! ### The distortion of the step quantiser, cell by cell -/

/-- **The cell decomposition of the distortion.**  Against a finite law supported in `Icc a b`,
the distortion of the step quantiser is the sum over cells of the squared error to that cell's
codeword, each cell trimmed to the support.  The trimming is what makes the two outer cells,
which are unbounded rays, contribute over a bounded set. -/
theorem distortion_stepQuant_eq_sum (cbk : Codebook K) (ρ : Measure ℝ) [IsFiniteMeasure ρ]
    {a b : ℝ} (hsupp : ρ (Set.Icc a b)ᶜ = 0) :
    distortion (stepQuant cbk) ρ
      = ∑ j : Fin (K + 1), ∫ t in cell cbk j ∩ Set.Icc a b, (t - cbk.c j) ^ 2 ∂ρ := by
  simp only [distortion]
  rw [integral_eq_sum_cell cbk ρ (integrable_sub_stepQuant_sq_of_supported cbk ρ hsupp)]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← setIntegral_eq_of_supported hsupp (fun t : ℝ => (t - cbk.c j) ^ 2) (cell cbk j)]
  refine setIntegral_congr_fun (measurableSet_cell cbk j) fun t ht => ?_
  rw [stepQuant_eq_of_mem_cell cbk ht]

/-- The same decomposition with the cells left untrimmed.  Both sides are real numbers under the
support hypothesis; the trimmed form above is the one a bounded-interval argument wants. -/
theorem distortion_stepQuant_eq_sum_cell (cbk : Codebook K) (ρ : Measure ℝ) [IsFiniteMeasure ρ]
    {a b : ℝ} (hsupp : ρ (Set.Icc a b)ᶜ = 0) :
    distortion (stepQuant cbk) ρ
      = ∑ j : Fin (K + 1), ∫ t in cell cbk j, (t - cbk.c j) ^ 2 ∂ρ := by
  rw [distortion_stepQuant_eq_sum cbk ρ hsupp]
  exact Finset.sum_congr rfl fun j _ =>
    (setIntegral_eq_of_supported hsupp (fun t : ℝ => (t - cbk.c j) ^ 2) (cell cbk j)).symm

/-- **Completing the square, cell by cell.**  On one cell the squared error to the codeword is the
second moment of the cell plus a term built only from the cell's mass and first moment. -/
theorem setIntegral_cell_sub_sq (cbk : Codebook K) (ρ : Measure ℝ) [IsFiniteMeasure ρ]
    {a b : ℝ} (hsupp : ρ (Set.Icc a b)ᶜ = 0) (j : Fin (K + 1)) :
    ∫ t in cell cbk j, (t - cbk.c j) ^ 2 ∂ρ
      = (∫ t in cell cbk j, t ^ 2 ∂ρ)
        + cbk.c j * (cbk.c j * cellMass cbk ρ j - 2 * cellMoment cbk ρ j) := by
  have hid : IntegrableOn (fun t : ℝ => t) (cell cbk j) ρ :=
    (integrable_id_of_supported ρ hsupp).integrableOn
  have hsq : IntegrableOn (fun t : ℝ => t ^ 2) (cell cbk j) ρ :=
    (integrable_sq_of_supported ρ hsupp).integrableOn
  have hlin : IntegrableOn (fun t : ℝ => 2 * cbk.c j * t) (cell cbk j) ρ :=
    hid.const_mul (2 * cbk.c j)
  have hconst : IntegrableOn (fun _ : ℝ => cbk.c j ^ 2) (cell cbk j) ρ :=
    (integrable_const (cbk.c j ^ 2)).integrableOn
  have hsub : IntegrableOn (fun t : ℝ => t ^ 2 - 2 * cbk.c j * t) (cell cbk j) ρ := hsq.sub hlin
  have hmul : ∫ t in cell cbk j, 2 * cbk.c j * t ∂ρ = 2 * cbk.c j * cellMoment cbk ρ j := by
    simp only [cellMoment]
    rw [integral_const_mul]
  have hcst : ∫ _t in cell cbk j, cbk.c j ^ 2 ∂ρ = cbk.c j ^ 2 * cellMass cbk ρ j := by
    rw [setIntegral_const, smul_eq_mul]
    simp only [cellMass]
    ring
  have hstep : ∫ t in cell cbk j, (t - cbk.c j) ^ 2 ∂ρ
      = ∫ t in cell cbk j, (t ^ 2 - 2 * cbk.c j * t + cbk.c j ^ 2) ∂ρ :=
    setIntegral_congr_fun (measurableSet_cell cbk j) (fun t _ => by ring)
  rw [hstep, integral_add hsub hconst, integral_sub hsq hlin, hmul, hcst]
  ring

/-- **The split of the square.**  Against a finite law supported in `Icc a b`, the distortion of
the step quantiser is the law's second moment plus a sum over cells of a term built only from the
codewords and the cells' masses and first moments.  The second-moment term does not mention the
codebook at all, which is what lets matched second moments cancel it when two laws are compared;
no such comparison is made here. -/
theorem distortion_eq_sq_add_cross (cbk : Codebook K) (ρ : Measure ℝ) [IsFiniteMeasure ρ]
    {a b : ℝ} (hsupp : ρ (Set.Icc a b)ᶜ = 0) :
    distortion (stepQuant cbk) ρ
      = (∫ t, t ^ 2 ∂ρ)
        + ∑ j : Fin (K + 1),
            cbk.c j * (cbk.c j * cellMass cbk ρ j - 2 * cellMoment cbk ρ j) := by
  rw [distortion_stepQuant_eq_sum_cell cbk ρ hsupp,
    integral_eq_sum_cell cbk ρ (integrable_sq_of_supported ρ hsupp), ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun j _ => setIntegral_cell_sub_sq cbk ρ hsupp j

/-! ### Transport to an arbitrary nearest-codeword map

Every statement above is about the one explicit step function `stepQuant`.  §52's
`distortion_congr_of_isNearest` moves each of them to an arbitrary nearest-codeword map for the
same codebook, at no cost and with no tie convention: the two integrands agree at every point. -/

/-- The cell decomposition, for an arbitrary nearest-codeword map. -/
theorem distortion_isNearest_eq_sum {cbk : Codebook K} {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (ρ : Measure ℝ) [IsFiniteMeasure ρ]
    {a b : ℝ} (hsupp : ρ (Set.Icc a b)ᶜ = 0) :
    distortion q ρ
      = ∑ j : Fin (K + 1), ∫ t in cell cbk j ∩ Set.Icc a b, (t - cbk.c j) ^ 2 ∂ρ := by
  rw [distortion_congr_of_isNearest hq (isNearest_stepQuant cbk) ρ]
  exact distortion_stepQuant_eq_sum cbk ρ hsupp

/-- **The split of the square, for an arbitrary nearest-codeword map.**  This is the form the gap
identity consumes: the quantiser is any nearest map of the codebook, and the only law-dependent
quantities on the right are the second moment, the cell masses and the cell first moments. -/
theorem distortion_isNearest_eq_sq_add_cross {cbk : Codebook K} {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (ρ : Measure ℝ) [IsFiniteMeasure ρ]
    {a b : ℝ} (hsupp : ρ (Set.Icc a b)ᶜ = 0) :
    distortion q ρ
      = (∫ t, t ^ 2 ∂ρ)
        + ∑ j : Fin (K + 1),
            cbk.c j * (cbk.c j * cellMass cbk ρ j - 2 * cellMoment cbk ρ j) := by
  rw [distortion_congr_of_isNearest hq (isNearest_stepQuant cbk) ρ]
  exact distortion_eq_sq_add_cross cbk ρ hsupp

end Threshold
