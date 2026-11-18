
/-! ## §49  The scalar gap is an exact finite cell sum

A deployed scalar quantiser has finitely many codewords, so `cb` is a step function: its values
lie in a finite list `S ⊆ ℝ`, and the *cell* of a codeword `c`, the set `cell cb c = cb ⁻¹' {c}`
of scalars that `cb` rounds to `c`, is measurable as soon as `cb` is.  Measurability of one cell
needs no finiteness at all, since `ℝ` has measurable singletons; finiteness of the range is what
makes the family of cells a *finite* measurable partition of the line, and hence what turns an
integral into a finite sum.

That is the whole content of `integral_eq_sum_setIntegral_cell` and `integral_mul_comp_eq_sum`:
an integral against `ν` splits over the cells, and a factor depending on `t` only through `cb t`
is constant on each cell and comes out of that cell's integral.  Expanding the square in
`distortion` and applying both gives `distortion_eq_cells`, which writes the distortion of `cb`
against `ν` in the three families of numbers a cell decomposition exposes: the second moment of
`ν`, the first moment of `ν` on each cell, and the mass `ν` gives each cell.  Subtracting two
copies of that gives `distortion_sub_distortion_eq_cells`, the gap between the distortions of one
quantiser against two laws, term by term; and reading the same identity the other way gives
`distortion_congr_of_cells`, which says that two laws agreeing in those three families of numbers
give `cb` the same distortion, however else they differ.

**These are identities, not bounds.**  Nothing is dropped, nothing is estimated, and no term is
signed.  In particular none of them says the gap between two laws is small, or that it is
controlled by any single summary statistic of the two laws: the right-hand side of
`distortion_sub_distortion_eq_cells` is a sum of `2 · card S + 1` signed differences of real
numbers, with no sign and no bound attached to any of them, and a bound on the gap would have to
come from separate control of those differences.  What the identity does establish is that the
gap depends on the pair of laws *only* through the cell masses, the cell first moments, and the
second moments, which is the precise form of `distortion_congr_of_cells`.  Whether any coarser
statistic of the two laws determines those numbers is not addressed here, in either direction.

The hypotheses are that `cb` is measurable with values in `S`, that each law is a finite measure,
and that each has a finite second moment.  The last two are not removable: without them the
second moment need not be a real number and the cross term `∫ t · cb t` need not converge, and
`∫` would silently return its junk value.  They are also sufficient, which is why no separate
integrability hypothesis on `cb` appears: a finite range bounds `|cb|` by `codewordBound S`, so
`∫ (cb t)²` and `∫ t · cb t` are controlled by the mass of `ν` and by its second moment.  By
`integrable_sub_cb_sq_iff` the second-moment hypothesis is not an extra assumption on top of
"the distortion is finite": under the other hypotheses the two are equivalent.

Nothing here is specific to the spherical coordinate law or to any particular codebook, and no
threshold or order structure on the codewords is used or assumed. -/

namespace ScalarGap

variable {cb : ℝ → ℝ} {ν ν' : Measure ℝ} {S : Finset ℝ}

/-- The cell of the codeword `c`: the scalars that `cb` rounds to `c`. -/
def cell (cb : ℝ → ℝ) (c : ℝ) : Set ℝ := cb ⁻¹' {c}

/-- A cell is measurable as soon as `cb` is.  Finiteness of the range plays no part: `ℝ` has
measurable singletons, so a cell is the preimage of a measurable set. -/
theorem measurableSet_cell (hcb : Measurable cb) (c : ℝ) : MeasurableSet (cell cb c) :=
  hcb (measurableSet_singleton c)

/-- Cells of distinct codewords are disjoint.  This holds for any `cb` whatsoever. -/
theorem pairwiseDisjoint_cell (cb : ℝ → ℝ) (S : Finset ℝ) :
    Set.PairwiseDisjoint (↑S) (cell cb) :=
  Set.pairwiseDisjoint_fiber cb ↑S

/-- If every value of `cb` lies in `S`, the cells of `S` cover the line.  Together with
`measurableSet_cell` and `pairwiseDisjoint_cell` this is the finite measurable partition. -/
theorem iUnion_cell_eq_univ (hS : Set.range cb ⊆ ↑S) : ⋃ c ∈ S, cell cb c = Set.univ := by
  have h : ⋃ c ∈ S, cell cb c = cb ⁻¹' ↑S := Finset.set_biUnion_preimage_singleton cb S
  rw [h, Set.eq_univ_iff_forall]
  exact fun t => hS ⟨t, rfl⟩

/-- **The cell decomposition of an integral.**  An integral against `ν` is the sum of the
integrals over the cells of a codeword list for `cb`.  Integrability of `f` on the whole line is
what is assumed; integrability on each cell follows. -/
theorem integral_eq_sum_setIntegral_cell (hcb : Measurable cb) (hS : Set.range cb ⊆ ↑S)
    {f : ℝ → ℝ} (hf : Integrable f ν) :
    ∫ t, f t ∂ν = ∑ c ∈ S, ∫ t in cell cb c, f t ∂ν := by
  conv_lhs => rw [← setIntegral_univ]
  rw [← iUnion_cell_eq_univ hS]
  exact integral_biUnion_finset S (fun c _ => measurableSet_cell hcb c)
    (pairwiseDisjoint_cell cb S) fun c _ => hf.integrableOn

/-- **A factor depending on `t` only through `cb t` comes out cell by cell.**  On the cell of `c`
the composite `g ∘ cb` is the constant `g c`, so the integral of `f · (g ∘ cb)` is the sum over
codewords of `g c` times the integral of `f` over the cell of `c`.  No hypothesis on `g` is needed
beyond integrability of the product: `g` is not assumed measurable, because `g ∘ cb` is constant
on each cell of the partition. -/
theorem integral_mul_comp_eq_sum (hcb : Measurable cb) (hS : Set.range cb ⊆ ↑S)
    (f g : ℝ → ℝ) (hfg : Integrable (fun t => f t * g (cb t)) ν) :
    ∫ t, f t * g (cb t) ∂ν = ∑ c ∈ S, g c * ∫ t in cell cb c, f t ∂ν := by
  rw [integral_eq_sum_setIntegral_cell hcb hS hfg]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [← integral_const_mul]
  refine setIntegral_congr_fun (measurableSet_cell hcb c) fun t ht => ?_
  have hct : cb t = c := ht
  rw [hct]; ring

/-! ### Integrability from a finite codeword list

Everything below needs to know that the cross term and the codeword square are integrable.  Both
follow from a single crude bound, the sum of the absolute codewords, so no integrability
hypothesis on `cb` itself ever appears in a statement. -/

/-- The sum of the absolute values of the codewords.  It dominates `|cb t|` for every `t`, which
is the only use made of finiteness of the range in the integrability lemmas below. -/
noncomputable def codewordBound (S : Finset ℝ) : ℝ := ∑ c ∈ S, |c|

/-- Every value of a quantiser with codeword list `S` is bounded by `codewordBound S`. -/
theorem abs_cb_le_codewordBound (hS : Set.range cb ⊆ ↑S) (t : ℝ) :
    |cb t| ≤ codewordBound S :=
  Finset.single_le_sum (f := fun c => |c|) (fun c _ => abs_nonneg c)
    (Finset.mem_coe.mp (hS ⟨t, rfl⟩))

/-- On a finite measure a square-integrable function is absolutely integrable, by `|x| ≤ 1 + x²`.
This is the only place finiteness of `ν` is used to get a first moment from a second. -/
theorem integrable_abs_of_integrable_sq [IsFiniteMeasure ν] {f : ℝ → ℝ}
    (hf : AEStronglyMeasurable f ν) (h2 : Integrable (fun t => f t ^ 2) ν) :
    Integrable (fun t => |f t|) ν := by
  have hb : Integrable (fun t : ℝ => 1 + f t ^ 2) ν := (integrable_const (1 : ℝ)).add h2
  refine Integrable.mono' hb (continuous_abs.comp_aestronglyMeasurable hf)
    (Filter.Eventually.of_forall fun t => ?_)
  rw [Real.norm_eq_abs, abs_abs]
  nlinarith [sq_nonneg (|f t| - 1), sq_abs (f t), abs_nonneg (f t)]

/-- `(cb ·)²` is integrable against any finite measure: it is measurable and bounded by
`codewordBound S ^ 2`.  No moment hypothesis on `ν` is used. -/
theorem integrable_cb_sq [IsFiniteMeasure ν] (hcb : Measurable cb) (hS : Set.range cb ⊆ ↑S) :
    Integrable (fun t => cb t ^ 2) ν := by
  refine Integrable.mono' (integrable_const (codewordBound S ^ 2))
    (hcb.pow_const 2).aestronglyMeasurable (Filter.Eventually.of_forall fun t => ?_)
  rw [Real.norm_eq_abs, abs_pow]
  exact pow_le_pow_left₀ (abs_nonneg _) (abs_cb_le_codewordBound hS t) 2

/-- The cross term `t · cb t` is integrable: `|cb|` is bounded by `codewordBound S` and `ν` has a
finite second moment, hence a finite first absolute moment. -/
theorem integrable_mul_cb [IsFiniteMeasure ν] (hcb : Measurable cb) (hS : Set.range cb ⊆ ↑S)
    (h2 : Integrable (fun t : ℝ => t ^ 2) ν) : Integrable (fun t => t * cb t) ν := by
  refine Integrable.mono'
    ((integrable_abs_of_integrable_sq aestronglyMeasurable_id h2).const_mul (codewordBound S))
    (measurable_id.mul hcb).aestronglyMeasurable (Filter.Eventually.of_forall fun t => ?_)
  rw [Real.norm_eq_abs, abs_mul]
  calc |t| * |cb t| ≤ |t| * codewordBound S :=
        mul_le_mul_of_nonneg_left (abs_cb_le_codewordBound hS t) (abs_nonneg t)
    _ = codewordBound S * |t| := mul_comm _ _

/-- The same bound with a general square-integrable `f` in place of the identity: a bounded
quantiser multiplies a square-integrable function into an integrable one. -/
theorem integrable_mul_cb' [IsFiniteMeasure ν] (hcb : Measurable cb) (hS : Set.range cb ⊆ ↑S)
    {f : ℝ → ℝ} (hf : AEStronglyMeasurable f ν) (h2 : Integrable (fun t => f t ^ 2) ν) :
    Integrable (fun t => f t * cb t) ν := by
  refine Integrable.mono'
    ((integrable_abs_of_integrable_sq hf h2).const_mul (codewordBound S))
    (hf.mul hcb.aestronglyMeasurable) (Filter.Eventually.of_forall fun t => ?_)
  rw [Real.norm_eq_abs, abs_mul]
  calc |f t| * |cb t| ≤ |f t| * codewordBound S :=
        mul_le_mul_of_nonneg_left (abs_cb_le_codewordBound hS t) (abs_nonneg _)
    _ = codewordBound S * |f t| := mul_comm _ _

/-- **A finite distortion and a finite second moment are the same hypothesis** for a measurable
quantiser with finitely many codewords against a finite measure, because the two functions differ
by the bounded `cb`.  So `distortion_eq_cells` may be read as taking finiteness of its own
left-hand side as its moment hypothesis. -/
theorem integrable_sub_cb_sq_iff [IsFiniteMeasure ν] (hcb : Measurable cb)
    (hS : Set.range cb ⊆ ↑S) :
    Integrable (fun t : ℝ => (t - cb t) ^ 2) ν ↔ Integrable (fun t : ℝ => t ^ 2) ν := by
  have hsq := integrable_cb_sq (ν := ν) hcb hS
  constructor
  · intro h
    have hmix : Integrable (fun t : ℝ => 2 * ((t - cb t) * cb t)) ν :=
      (integrable_mul_cb' hcb hS (measurable_id.sub hcb).aestronglyMeasurable h).const_mul 2
    have hall : Integrable
        (fun t : ℝ => (t - cb t) ^ 2 + 2 * ((t - cb t) * cb t) + cb t ^ 2) ν :=
      (h.add hmix).add hsq
    exact hall.congr (Filter.Eventually.of_forall fun t => by ring)
  · intro h
    have hmix : Integrable (fun t : ℝ => 2 * (t * cb t)) ν :=
      (integrable_mul_cb hcb hS h).const_mul 2
    have hall : Integrable (fun t : ℝ => t ^ 2 - 2 * (t * cb t) + cb t ^ 2) ν :=
      (h.sub hmix).add hsq
    exact hall.congr (Filter.Eventually.of_forall fun t => by ring)

/-! ### The identity, the gap, and what the gap depends on -/

/-- **The distortion in cell form.**  For a measurable `cb` whose values all lie in the finite
list `S`, and a finite `ν` of finite second moment, the distortion is the second moment of `ν`,
less twice the codeword-weighted first moments of `ν` on the cells, plus the
codeword-squared-weighted cell masses.  This is an identity, and every term on the right is a real
number under those hypotheses. -/
theorem distortion_eq_cells [IsFiniteMeasure ν] (hcb : Measurable cb)
    (hS : Set.range cb ⊆ ↑S) (h2 : Integrable (fun t : ℝ => t ^ 2) ν) :
    distortion cb ν
      = (∫ t, t ^ 2 ∂ν)
        - 2 * ∑ c ∈ S, c * ∫ t in cell cb c, t ∂ν
        + ∑ c ∈ S, c ^ 2 * ν.real (cell cb c) := by
  have hmul := integrable_mul_cb hcb hS h2
  have hsq := integrable_cb_sq (ν := ν) hcb hS
  have hA : ∫ t, t * cb t ∂ν = ∑ c ∈ S, c * ∫ t in cell cb c, t ∂ν :=
    integral_mul_comp_eq_sum hcb hS (fun t => t) (fun t => t) hmul
  have hB : ∫ t, cb t ^ 2 ∂ν = ∑ c ∈ S, c ^ 2 * ν.real (cell cb c) := by
    have h := integral_mul_comp_eq_sum hcb hS (fun _ => (1 : ℝ)) (fun c => c ^ 2)
      (by simpa using hsq)
    simpa using h
  have hcross : Integrable (fun t : ℝ => 2 * (t * cb t)) ν := hmul.const_mul 2
  have hhead : Integrable (fun t : ℝ => t ^ 2 - 2 * (t * cb t)) ν := h2.sub hcross
  have hexp : ∀ t : ℝ, (t - cb t) ^ 2 = t ^ 2 - 2 * (t * cb t) + cb t ^ 2 := fun t => by ring
  unfold distortion
  simp_rw [hexp]
  rw [integral_add hhead hsq, integral_sub h2 hcross, integral_const_mul, hA, hB]

/-- **The exact gap in cell form.**  Two copies of `distortion_eq_cells` for the same quantiser
against two laws, subtracted: the gap between the distortions is the gap between the second
moments, less twice the codeword-weighted gaps between the cell first moments, plus the
codeword-squared-weighted gaps between the cell masses.  This is an identity and not a bound:
nothing is dropped, no term is signed, and it does not by itself say the gap is small. -/
theorem distortion_sub_distortion_eq_cells [IsFiniteMeasure ν] [IsFiniteMeasure ν']
    (hcb : Measurable cb) (hS : Set.range cb ⊆ ↑S)
    (h2 : Integrable (fun t : ℝ => t ^ 2) ν) (h2' : Integrable (fun t : ℝ => t ^ 2) ν') :
    distortion cb ν - distortion cb ν'
      = ((∫ t, t ^ 2 ∂ν) - ∫ t, t ^ 2 ∂ν')
        - 2 * ∑ c ∈ S, c * ((∫ t in cell cb c, t ∂ν) - ∫ t in cell cb c, t ∂ν')
        + ∑ c ∈ S, c ^ 2 * (ν.real (cell cb c) - ν'.real (cell cb c)) := by
  rw [distortion_eq_cells hcb hS h2, distortion_eq_cells hcb hS h2']
  simp only [mul_sub, Finset.sum_sub_distrib]
  ring

/-- **The distortion depends on the law only through the second moment, the cell masses and the
cell first moments.**  Two laws that agree in those numbers, on the cells of the codeword list `S`
alone, give `cb` the same distortion; they are otherwise unrelated, and no bound on any distance
between them is assumed or produced.  It is read off from two copies of `distortion_eq_cells`,
so like that identity it estimates nothing. -/
theorem distortion_congr_of_cells [IsFiniteMeasure ν] [IsFiniteMeasure ν']
    (hcb : Measurable cb) (hS : Set.range cb ⊆ ↑S)
    (h2 : Integrable (fun t : ℝ => t ^ 2) ν) (h2' : Integrable (fun t : ℝ => t ^ 2) ν')
    (hmom : ∫ t, t ^ 2 ∂ν = ∫ t, t ^ 2 ∂ν')
    (hmass : ∀ c ∈ S, ν.real (cell cb c) = ν'.real (cell cb c))
    (hmean : ∀ c ∈ S, (∫ t in cell cb c, t ∂ν) = ∫ t in cell cb c, t ∂ν') :
    distortion cb ν = distortion cb ν' := by
  have hX : ∑ c ∈ S, c * ∫ t in cell cb c, t ∂ν = ∑ c ∈ S, c * ∫ t in cell cb c, t ∂ν' :=
    Finset.sum_congr rfl fun c hc => by rw [hmean c hc]
  have hY : ∑ c ∈ S, c ^ 2 * ν.real (cell cb c) = ∑ c ∈ S, c ^ 2 * ν'.real (cell cb c) :=
    Finset.sum_congr rfl fun c hc => by rw [hmass c hc]
  rw [distortion_eq_cells hcb hS h2, distortion_eq_cells hcb hS h2', hmom, hX, hY]

/-! ### The same three statements from finiteness of the range alone

`Set.range cb ⊆ ↑S` is the convenient hypothesis, since a deployed codebook is given as a list and
the sums then range over the nominal codebook rather than over the realised range.  When only
`(Set.range cb).Finite` is on hand, `Set.Finite.toFinset` supplies the list. -/

/-- The cell decomposition with the codeword list taken to be the range itself. -/
theorem integral_eq_sum_setIntegral_cell_of_finite (hcb : Measurable cb)
    (hfin : (Set.range cb).Finite) {f : ℝ → ℝ} (hf : Integrable f ν) :
    ∫ t, f t ∂ν = ∑ c ∈ hfin.toFinset, ∫ t in cell cb c, f t ∂ν :=
  integral_eq_sum_setIntegral_cell hcb hfin.coe_toFinset.ge hf

/-- The distortion in cell form with the codeword list taken to be the range itself. -/
theorem distortion_eq_cells_of_finite [IsFiniteMeasure ν] (hcb : Measurable cb)
    (hfin : (Set.range cb).Finite) (h2 : Integrable (fun t : ℝ => t ^ 2) ν) :
    distortion cb ν
      = (∫ t, t ^ 2 ∂ν)
        - 2 * ∑ c ∈ hfin.toFinset, c * ∫ t in cell cb c, t ∂ν
        + ∑ c ∈ hfin.toFinset, c ^ 2 * ν.real (cell cb c) :=
  distortion_eq_cells hcb hfin.coe_toFinset.ge h2

/-- The exact gap with the codeword list taken to be the range itself.  Still an identity, still
not a bound. -/
theorem distortion_sub_distortion_eq_cells_of_finite [IsFiniteMeasure ν] [IsFiniteMeasure ν']
    (hcb : Measurable cb) (hfin : (Set.range cb).Finite)
    (h2 : Integrable (fun t : ℝ => t ^ 2) ν) (h2' : Integrable (fun t : ℝ => t ^ 2) ν') :
    distortion cb ν - distortion cb ν'
      = ((∫ t, t ^ 2 ∂ν) - ∫ t, t ^ 2 ∂ν')
        - 2 * ∑ c ∈ hfin.toFinset, c * ((∫ t in cell cb c, t ∂ν) - ∫ t in cell cb c, t ∂ν')
        + ∑ c ∈ hfin.toFinset, c ^ 2 * (ν.real (cell cb c) - ν'.real (cell cb c)) :=
  distortion_sub_distortion_eq_cells hcb hfin.coe_toFinset.ge h2 h2'

/-- Agreement of the second moments and of the cell masses and cell first moments at *every* real
`c` forces equal distortion.  Quantifying over all of `ℝ` assumes more than the proof uses: it is
restricted at once to the range of `cb`, which is the codeword list here. -/
theorem distortion_congr_of_cells_of_finite [IsFiniteMeasure ν] [IsFiniteMeasure ν']
    (hcb : Measurable cb) (hfin : (Set.range cb).Finite)
    (h2 : Integrable (fun t : ℝ => t ^ 2) ν) (h2' : Integrable (fun t : ℝ => t ^ 2) ν')
    (hmom : ∫ t, t ^ 2 ∂ν = ∫ t, t ^ 2 ∂ν')
    (hmass : ∀ c, ν.real (cell cb c) = ν'.real (cell cb c))
    (hmean : ∀ c, (∫ t in cell cb c, t ∂ν) = ∫ t in cell cb c, t ∂ν') :
    distortion cb ν = distortion cb ν' :=
  distortion_congr_of_cells hcb hfin.coe_toFinset.ge h2 h2' hmom
    (fun c _ => hmass c) fun c _ => hmean c

/-! ### Two probability laws of equal second moment

The pair the paper cares about is a deployed coordinate law against the Haar coordinate law of the
same dimension, and `integral_sq_sphericalCoordLaw` pins the second moment of the latter at `1/d`.
Whenever the two second moments agree, the first term of the gap drops out and the gap is carried
entirely by the cells. -/

/-- The gap between two laws of equal second moment is exactly the cell sum, with no second-moment
term.  This is `distortion_sub_distortion_eq_cells` with its first difference set to zero; it is
still an identity, and it still asserts nothing about the size of what remains. -/
theorem distortion_sub_distortion_eq_cells_of_moment_eq [IsFiniteMeasure ν] [IsFiniteMeasure ν']
    (hcb : Measurable cb) (hS : Set.range cb ⊆ ↑S)
    (h2 : Integrable (fun t : ℝ => t ^ 2) ν) (h2' : Integrable (fun t : ℝ => t ^ 2) ν')
    (hmom : ∫ t, t ^ 2 ∂ν = ∫ t, t ^ 2 ∂ν') :
    distortion cb ν - distortion cb ν'
      = -(2 * ∑ c ∈ S, c * ((∫ t in cell cb c, t ∂ν) - ∫ t in cell cb c, t ∂ν'))
        + ∑ c ∈ S, c ^ 2 * (ν.real (cell cb c) - ν'.real (cell cb c)) := by
  rw [distortion_sub_distortion_eq_cells hcb hS h2 h2', hmom]
  ring

/-! ### A two-level quantiser, to check the hypotheses are met by something

`signCb` is the one-bit quantiser with codewords `±1`.  It is measurable and its range is contained
in `signCodewords`, so `distortion_eq_cells` applies to it against any finite `ν` of finite second
moment.  This is a check on the shape of the hypotheses, not a result about the paper's
quantisers. -/

/-- The codeword list of the one-bit sign quantiser. -/
noncomputable def signCodewords : Finset ℝ := {-1, 1}

/-- The one-bit sign quantiser. -/
noncomputable def signCb (t : ℝ) : ℝ := if t < 0 then -1 else 1

/-- The one-bit sign quantiser is measurable, being a two-branch `if` on a measurable set. -/
theorem measurable_signCb : Measurable signCb :=
  Measurable.ite (measurableSet_lt measurable_id measurable_const)
    measurable_const measurable_const

/-- Every value of the one-bit sign quantiser is a codeword of `signCodewords`. -/
theorem range_signCb_subset : Set.range signCb ⊆ ↑signCodewords := by
  rintro x ⟨t, rfl⟩
  by_cases h : t < 0 <;> simp [signCb, signCodewords, h]

/-- The hypotheses of `distortion_eq_cells` are dischargeable: here they are, discharged, for the
one-bit sign quantiser against an arbitrary finite law of finite second moment. -/
theorem distortion_signCb_eq_cells [IsFiniteMeasure ν]
    (h2 : Integrable (fun t : ℝ => t ^ 2) ν) :
    distortion signCb ν
      = (∫ t, t ^ 2 ∂ν)
        - 2 * ∑ c ∈ signCodewords, c * ∫ t in cell signCb c, t ∂ν
        + ∑ c ∈ signCodewords, c ^ 2 * ν.real (cell signCb c) :=
  distortion_eq_cells measurable_signCb range_signCb_subset h2

end ScalarGap
