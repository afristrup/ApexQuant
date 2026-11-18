import Mathlib

open Finset MeasureTheory

namespace ApexQuant.Oblivious

variable {n : Type*} [Fintype n]

/-- The quadratic form `uᵀ H u`. -/
def q (H : Matrix n n ℝ) (u : n → ℝ) : ℝ :=
  ∑ i, ∑ j, u i * H i j * u j

/-- A random vector whose second-moment matrix is `c` times the identity has
expected quadratic form `c * trace H`. -/
theorem integral_q_of_second_moment [DecidableEq n]
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    {U : Ω → n → ℝ} {c : ℝ} (H : Matrix n n ℝ)
    (hint : ∀ i j, Integrable (fun ω => U ω i * U ω j) μ)
    (hsecond : ∀ i j, ∫ ω, U ω i * U ω j ∂μ = if i = j then c else 0) :
    ∫ ω, q H (U ω) ∂μ = c * Matrix.trace H := by
  have hq : (fun ω => q H (U ω)) =
      fun ω => ∑ i, ∑ j, H i j * (U ω i * U ω j) := by
    funext ω
    unfold q
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
  rw [hq, integral_finsetSum _ fun i _ =>
    integrable_finsetSum _ fun j _ => (hint i j).const_mul (H i j)]
  simp_rw [integral_finsetSum _ fun j _ => (hint _ j).const_mul (H _ j),
    integral_const_mul, hsecond]
  simp [Matrix.trace, Finset.mul_sum, mul_comm]

/-- A random direction with isotropic second moments `I / d` sees the average
eigenvalue: its expected quadratic form is `trace H / d`. -/
theorem expected_quadratic_form_of_isotropic [DecidableEq n]
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {U : Ω → n → ℝ} (H : Matrix n n ℝ)
    (hint : ∀ i j, Integrable (fun ω => U ω i * U ω j) μ)
    (hsecond : ∀ i j, ∫ ω, U ω i * U ω j ∂μ =
      if i = j then 1 / (Fintype.card n : ℝ) else 0) :
    ∫ ω, q H (U ω) ∂μ = Matrix.trace H / Fintype.card n := by
  rw [integral_q_of_second_moment H hint hsecond]
  ring

/-- An average on a probability space is at most any pointwise upper bound. -/
theorem sup_ge_average
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} {M : ℝ} (hX : Integrable X μ) (hle : ∀ ω, X ω ≤ M) :
    ∫ ω, X ω ∂μ ≤ M := by
  have h := integral_mono hX (integrable_const M) hle
  simpa using h

/-- A worst case is at least the average: some outcome has loss at least the
expected loss. -/
theorem exists_ge_average
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {L : Ω → ℝ} {a : ℝ} (hL : Integrable L μ) (hmean : ∫ ω, L ω ∂μ = a) :
    ∃ ω, a ≤ L ω := by
  obtain ⟨ω, hω⟩ := exists_integral_le hL
  exact ⟨ω, hmean ▸ hω⟩

/-- The contrapositive form: the loss cannot be strictly below its average at
every outcome. -/
theorem not_forall_lt_average
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {L : Ω → ℝ} {a : ℝ} (hL : Integrable L μ) (hmean : ∫ ω, L ω ∂μ = a) :
    ¬ ∀ ω, L ω < a := by
  intro h
  obtain ⟨ω, hω⟩ := exists_ge_average hL hmean
  exact absurd (h ω) (not_lt.mpr hω)

/-! The declarations below are the arithmetic of the Gaussian reverse
water-filling formula. They are not the rate-distortion coding theorem, which
is not formalized here. -/

/-- The geometric mean of the component variances. -/
noncomputable def gm (lam : n → ℝ) : ℝ :=
  Real.exp ((∑ i, Real.log (lam i)) / (Fintype.card n : ℝ))

/-- The arithmetic mean of the component variances. -/
noncomputable def am (lam : n → ℝ) : ℝ :=
  (∑ i, lam i) / (Fintype.card n : ℝ)

/-- The reverse water-filling level at rate `R` bits per component. -/
noncomputable def level (σ2 : ℝ) (lam : n → ℝ) (R : ℝ) : ℝ :=
  σ2 * gm lam * (4 : ℝ) ^ (-R)

/-- The rate gap between an oblivious and an aware code. -/
noncomputable def gap (lam : n → ℝ) : ℝ :=
  (1 / 2) * Real.logb 2 (am lam / gm lam)

/-- `log 4 = 2 log 2`, used to convert between base 4 powers and bits. -/
theorem log_four : Real.log 4 = 2 * Real.log 2 := by
  rw [show (4 : ℝ) = 2 ^ (2 : ℕ) by norm_num, Real.log_pow]
  norm_num

/-- The geometric mean is positive. -/
theorem gm_pos (lam : n → ℝ) : 0 < gm lam :=
  Real.exp_pos _

/-- The logarithm of the geometric mean is the mean log variance. -/
theorem log_gm (lam : n → ℝ) :
    Real.log (gm lam) = (∑ i, Real.log (lam i)) / (Fintype.card n : ℝ) :=
  Real.log_exp _

/-- The logarithm of the water level, in closed form. -/
theorem log_level {σ2 : ℝ} (hσ : 0 < σ2) (lam : n → ℝ) (R : ℝ) :
    Real.log (level σ2 lam R) =
      Real.log σ2 + (∑ i, Real.log (lam i)) / (Fintype.card n : ℝ) -
        R * (2 * Real.log 2) := by
  unfold level gm
  rw [Real.log_mul (by positivity) (by positivity),
    Real.log_mul hσ.ne' (Real.exp_pos _).ne', Real.log_exp,
    Real.log_rpow (by norm_num), log_four]
  ring

/-- Reverse water-filling arithmetic: the per-component rates at the water
level average to exactly `R`. This is arithmetic only, not the coding theorem. -/
theorem rate_identity [Nonempty n] {lam : n → ℝ} {σ2 : ℝ}
    (hlam : ∀ i, 0 < lam i) (hσ : 0 < σ2) (R : ℝ) :
    (1 / (Fintype.card n : ℝ)) *
      ∑ i, (1 / 2) * Real.logb 2 (σ2 * lam i / level σ2 lam R) = R := by
  have hd : (0 : ℝ) < Fintype.card n := by exact_mod_cast Fintype.card_pos
  have h2 : 0 < Real.log 2 := Real.log_pos one_lt_two
  have hν : 0 < level σ2 lam R := by unfold level gm; positivity
  have key : ∀ i, (1 / 2) * Real.logb 2 (σ2 * lam i / level σ2 lam R) =
      (Real.log (lam i) - (∑ i, Real.log (lam i)) / (Fintype.card n : ℝ)) /
        (2 * Real.log 2) + R := by
    intro i
    rw [Real.logb, Real.log_div (mul_pos hσ (hlam i)).ne' hν.ne',
      Real.log_mul hσ.ne' (hlam i).ne', log_level hσ]
    field_simp
    ring
  simp_rw [key]
  rw [Finset.sum_add_distrib, ← Finset.sum_div, Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp
  ring

/-- Reverse water-filling arithmetic: the level is below component `i` exactly
when the rate covers `(1/2) log2 (gm / lam i)`. Not the coding theorem. -/
theorem level_le_iff {lam : n → ℝ} {σ2 : ℝ}
    (hlam : ∀ i, 0 < lam i) (hσ : 0 < σ2) (R : ℝ) (i : n) :
    level σ2 lam R ≤ σ2 * lam i ↔
      (1 / 2) * Real.logb 2 (gm lam / lam i) ≤ R := by
  have h2 : 0 < Real.log 2 := Real.log_pos one_lt_two
  have hν : 0 < level σ2 lam R := by unfold level gm; positivity
  rw [← Real.log_le_log_iff hν (mul_pos hσ (hlam i)), log_level hσ,
    Real.log_mul hσ.ne' (hlam i).ne', Real.logb,
    Real.log_div (gm_pos lam).ne' (hlam i).ne', log_gm,
    show (1 / 2 : ℝ) * ((((∑ i, Real.log (lam i)) /
      (Fintype.card n : ℝ)) - Real.log (lam i)) / Real.log 2) =
      (((∑ i, Real.log (lam i)) / (Fintype.card n : ℝ)) - Real.log (lam i)) /
        (2 * Real.log 2) by field_simp,
    div_le_iff₀ (by positivity)]
  constructor <;> intro h <;> linarith

/-- AM-GM: the geometric mean of positive variances is at most the arithmetic
mean. -/
theorem am_gm [Nonempty n] {lam : n → ℝ} (hlam : ∀ i, 0 < lam i) :
    gm lam ≤ am lam := by
  have hd : (0 : ℝ) < Fintype.card n := by exact_mod_cast Fintype.card_pos
  have h := Real.geom_mean_le_arith_mean_weighted Finset.univ
    (fun _ : n => 1 / (Fintype.card n : ℝ)) lam
    (fun _ _ => by positivity)
    (by simp [Finset.sum_const, Finset.card_univ, hd.ne'])
    (fun i _ => (hlam i).le)
  have hprod : ∏ i, lam i ^ (1 / (Fintype.card n : ℝ)) = gm lam := by
    unfold gm
    rw [Finset.sum_div, Real.exp_sum]
    refine Finset.prod_congr rfl fun i _ => ?_
    rw [Real.rpow_def_of_pos (hlam i)]
    congr 1
    ring
  rw [hprod] at h
  unfold am
  rw [Finset.sum_div]
  refine h.trans (le_of_eq (Finset.sum_congr rfl fun i _ => ?_))
  ring

/-- The oblivious rate gap is nonnegative. -/
theorem gap_nonneg [Nonempty n] {lam : n → ℝ} (hlam : ∀ i, 0 < lam i) :
    0 ≤ gap lam := by
  have hgm : 0 < gm lam := Real.exp_pos _
  have h1 : 1 ≤ am lam / gm lam := (one_le_div hgm).mpr (am_gm hlam)
  unfold gap
  have := Real.logb_nonneg one_lt_two h1
  positivity

/-- Reverse water-filling arithmetic: an oblivious code at `R` bits has the
reference loss of an aware code at `R - gap` bits. Not the coding theorem. -/
theorem oblivious_rate_shift [Nonempty n] {lam : n → ℝ}
    (hlam : ∀ i, 0 < lam i) (R : ℝ) :
    am lam * (4 : ℝ) ^ (-R) = gm lam * (4 : ℝ) ^ (-(R - gap lam)) := by
  have hgm : 0 < gm lam := Real.exp_pos _
  have ham : 0 < am lam := hgm.trans_le (am_gm hlam)
  have h2 : 0 < Real.log 2 := Real.log_pos one_lt_two
  have h4 : (4 : ℝ) ^ gap lam = am lam / gm lam := by
    rw [Real.rpow_def_of_pos (by norm_num), log_four, gap, Real.logb,
      show 2 * Real.log 2 * (1 / 2 * (Real.log (am lam / gm lam) / Real.log 2)) =
        Real.log (am lam / gm lam) by field_simp,
      Real.exp_log (div_pos ham hgm)]
  rw [show -(R - gap lam) = -R + gap lam by ring,
    Real.rpow_add (by norm_num), h4]
  field_simp

/-! The declarations below are the bit-allocation optimization. Component `i`
given `r i` bits has weighted distortion `σ2 * lam i * 4 ^ (-(r i))`. The
optimum over all allocations with mean rate `R` is computed directly by AM-GM,
so no rate-distortion coding theorem is used. -/

/-- Total weighted distortion of the allocation `r`. -/
noncomputable def allocCost (σ2 : ℝ) (lam : n → ℝ) (r : n → ℝ) : ℝ :=
  ∑ i, σ2 * lam i * (4 : ℝ) ^ (-(r i))

/-- The optimal allocation at mean rate `R`. -/
noncomputable def rstar (lam : n → ℝ) (R : ℝ) (i : n) : ℝ :=
  R + (1 / 2) * Real.logb 2 (lam i / gm lam)

/-- `4 ^ ((1/2) log2 x) = x`. -/
theorem four_rpow_half_logb {x : ℝ} (hx : 0 < x) :
    (4 : ℝ) ^ ((1 / 2) * Real.logb 2 x) = x := by
  have h2 : 0 < Real.log 2 := Real.log_pos one_lt_two
  rw [Real.rpow_def_of_pos (by norm_num), log_four, Real.logb,
    show 2 * Real.log 2 * (1 / 2 * (Real.log x / Real.log 2)) = Real.log x by
      field_simp,
    Real.exp_log hx]

/-- The geometric mean of the terms `lam i * 4 ^ (-(r i))` is
`gm lam * 4 ^ (-R)` with `R` the mean rate. -/
theorem gm_allocation [Nonempty n] {lam : n → ℝ} (hlam : ∀ i, 0 < lam i)
    (r : n → ℝ) :
    gm (fun i => lam i * (4 : ℝ) ^ (-(r i))) =
      gm lam * (4 : ℝ) ^ (-((∑ i, r i) / (Fintype.card n : ℝ))) := by
  have hd : (0 : ℝ) < Fintype.card n := by exact_mod_cast Fintype.card_pos
  have h4 : ∀ x : ℝ, (0 : ℝ) < (4 : ℝ) ^ x := fun x =>
    Real.rpow_pos_of_pos (by norm_num) x
  have hlog : ∀ i, Real.log (lam i * (4 : ℝ) ^ (-(r i))) =
      Real.log (lam i) + -(r i) * Real.log 4 := fun i => by
    rw [Real.log_mul (hlam i).ne' (h4 _).ne', Real.log_rpow (by norm_num)]
  unfold gm
  simp_rw [hlog]
  rw [Real.rpow_def_of_pos (by norm_num), ← Real.exp_add,
    Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_neg_distrib]
  congr 1
  field_simp

/-- (a) Every allocation with mean rate `R` costs at least
`d * σ2 * gm lam * 4 ^ (-R)`. No sign constraint on the rates is needed. -/
theorem allocation_lower_bound [Nonempty n] {lam : n → ℝ} {σ2 : ℝ}
    (hlam : ∀ i, 0 < lam i) (hσ : 0 < σ2) {r : n → ℝ} {R : ℝ}
    (hR : (∑ i, r i) / (Fintype.card n : ℝ) = R) :
    (Fintype.card n : ℝ) * σ2 * gm lam * (4 : ℝ) ^ (-R) ≤
      ∑ i, σ2 * lam i * (4 : ℝ) ^ (-(r i)) := by
  have hd : (0 : ℝ) < Fintype.card n := by exact_mod_cast Fintype.card_pos
  have hpos : ∀ i, 0 < lam i * (4 : ℝ) ^ (-(r i)) := fun i =>
    mul_pos (hlam i) (Real.rpow_pos_of_pos (by norm_num) _)
  have h := am_gm hpos
  rw [gm_allocation hlam r, hR, am, le_div_iff₀ hd] at h
  calc (Fintype.card n : ℝ) * σ2 * gm lam * (4 : ℝ) ^ (-R)
      = σ2 * (gm lam * (4 : ℝ) ^ (-R) * (Fintype.card n : ℝ)) := by ring
    _ ≤ σ2 * ∑ i, lam i * (4 : ℝ) ^ (-(r i)) :=
        mul_le_mul_of_nonneg_left h hσ.le
    _ = ∑ i, σ2 * lam i * (4 : ℝ) ^ (-(r i)) := by
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun i _ => by ring

/-- (e) The lower bound restricted to nonnegative allocations: the constrained
optimum is never smaller than `d * σ2 * gm lam * 4 ^ (-R)`, at any rate. -/
theorem nonneg_allocation_lower_bound [Nonempty n] {lam : n → ℝ} {σ2 : ℝ}
    (hlam : ∀ i, 0 < lam i) (hσ : 0 < σ2) {r : n → ℝ} {R : ℝ}
    (_hr : ∀ i, 0 ≤ r i) (hR : (∑ i, r i) / (Fintype.card n : ℝ) = R) :
    (Fintype.card n : ℝ) * σ2 * gm lam * (4 : ℝ) ^ (-R) ≤
      ∑ i, σ2 * lam i * (4 : ℝ) ^ (-(r i)) :=
  allocation_lower_bound hlam hσ hR

/-- The optimal allocation has mean rate `R`. -/
theorem rstar_mean [Nonempty n] {lam : n → ℝ} (hlam : ∀ i, 0 < lam i)
    (R : ℝ) :
    (∑ i, rstar lam R i) / (Fintype.card n : ℝ) = R := by
  have hd : (0 : ℝ) < Fintype.card n := by exact_mod_cast Fintype.card_pos
  have h2 : 0 < Real.log 2 := Real.log_pos one_lt_two
  have key : ∀ i, rstar lam R i = R + (Real.log (lam i) -
      (∑ i, Real.log (lam i)) / (Fintype.card n : ℝ)) / (2 * Real.log 2) := by
    intro i
    rw [rstar, Real.logb, Real.log_div (hlam i).ne' (gm_pos lam).ne', log_gm]
    field_simp
  simp_rw [key]
  rw [Finset.sum_add_distrib, ← Finset.sum_div, Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp
  ring

/-- The distortion of component `i` under the optimal allocation is the water
level, the same for every component. -/
theorem rstar_term {lam : n → ℝ} (hlam : ∀ i, 0 < lam i) (σ2 R : ℝ) (i : n) :
    σ2 * lam i * (4 : ℝ) ^ (-(rstar lam R i)) = σ2 * gm lam * (4 : ℝ) ^ (-R) := by
  have hx : 0 < lam i / gm lam := div_pos (hlam i) (gm_pos lam)
  rw [rstar, neg_add, Real.rpow_add (by norm_num),
    Real.rpow_neg (by norm_num) ((1 / 2) * Real.logb 2 (lam i / gm lam)),
    four_rpow_half_logb hx]
  have := (hlam i).ne'
  have := (gm_pos lam).ne'
  field_simp

/-- (b) The allocation `rstar` has mean rate `R` and attains the lower bound. -/
theorem allocation_attained [Nonempty n] {lam : n → ℝ}
    (hlam : ∀ i, 0 < lam i) (σ2 R : ℝ) :
    (∑ i, rstar lam R i) / (Fintype.card n : ℝ) = R ∧
      ∑ i, σ2 * lam i * (4 : ℝ) ^ (-(rstar lam R i)) =
        (Fintype.card n : ℝ) * σ2 * gm lam * (4 : ℝ) ^ (-R) := by
  refine ⟨rstar_mean hlam R, ?_⟩
  simp_rw [rstar_term hlam]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  ring

/-- (b) The optimal allocation gives component `i` a nonnegative rate exactly
when the rate covers `(1/2) log2 (gm / lam i)`. -/
theorem rstar_nonneg_iff {lam : n → ℝ} (R : ℝ) (i : n) :
    0 ≤ rstar lam R i ↔ (1 / 2) * Real.logb 2 (gm lam / lam i) ≤ R := by
  rw [rstar, ← inv_div (lam i) (gm lam), Real.logb_inv]
  constructor <;> intro h <;> linarith

/-- (c) The uniform allocation `r i = R` costs `d * σ2 * am lam * 4 ^ (-R)`. -/
theorem uniform_allocation [Nonempty n] (lam : n → ℝ) (σ2 R : ℝ) :
    ∑ i, σ2 * lam i * (4 : ℝ) ^ (-R) =
      (Fintype.card n : ℝ) * σ2 * am lam * (4 : ℝ) ^ (-R) := by
  have hd : (0 : ℝ) < Fintype.card n := by exact_mod_cast Fintype.card_pos
  simp only [am]
  rw [← Finset.sum_mul, ← Finset.mul_sum]
  field_simp

/-- (d) The price of not seeing the Hessian: the uniform allocation at rate
`R` costs exactly what the optimal allocation costs at rate `R - gap lam`,
and it costs at least what the optimal allocation costs at rate `R`. -/
theorem oblivious_vs_aware [Nonempty n] {lam : n → ℝ}
    (hlam : ∀ i, 0 < lam i) {σ2 : ℝ} (hσ : 0 < σ2) (R : ℝ) :
    allocCost σ2 lam (fun _ => R) =
        allocCost σ2 lam (rstar lam (R - gap lam)) ∧
      allocCost σ2 lam (rstar lam R) ≤ allocCost σ2 lam (fun _ => R) := by
  have hu := uniform_allocation lam σ2 R
  refine ⟨?_, ?_⟩
  · unfold allocCost
    rw [hu, (allocation_attained hlam σ2 (R - gap lam)).2,
      mul_assoc _ (am lam), oblivious_rate_shift hlam R]
    ring
  · unfold allocCost
    rw [(allocation_attained hlam σ2 R).2]
    exact allocation_lower_bound hlam hσ (by simp)

/-! The declarations below lift the gap from an eigenvalue family to a positive
definite matrix, and show that it is additive over Kronecker products. -/

section MatrixGap

variable [DecidableEq n]

/-- The arithmetic mean of the eigenvalues is the normalized trace. -/
theorem am_eigenvalues {H : Matrix n n ℝ} (hH : H.IsHermitian) :
    am hH.eigenvalues = H.trace / (Fintype.card n : ℝ) := by
  rw [am, hH.trace_eq_sum_eigenvalues]
  simp

/-- The geometric mean of the eigenvalues of a positive definite matrix is the
normalized determinant. -/
theorem gm_eigenvalues {H : Matrix n n ℝ} (hH : H.PosDef) :
    gm hH.isHermitian.eigenvalues = H.det ^ (1 / (Fintype.card n : ℝ)) := by
  have hpos := hH.eigenvalues_pos
  have hdet : H.det = ∏ i, hH.isHermitian.eigenvalues i := by
    rw [hH.isHermitian.det_eq_prod_eigenvalues]
    simp
  have hdpos : 0 < H.det := hH.det_pos
  rw [gm, Real.rpow_def_of_pos hdpos, hdet, Real.log_prod (fun i _ => (hpos i).ne')]
  congr 1
  ring

/-- The oblivious rate gap of a matrix, in terms of its trace and determinant. -/
noncomputable def matGap (H : Matrix n n ℝ) : ℝ :=
  (1 / 2) * Real.logb 2
    ((H.trace / (Fintype.card n : ℝ)) / H.det ^ (1 / (Fintype.card n : ℝ)))

/-- For a positive definite matrix the matrix gap is the gap of its eigenvalues. -/
theorem matGap_eq_gap {H : Matrix n n ℝ} (hH : H.PosDef) :
    matGap H = gap hH.isHermitian.eigenvalues := by
  rw [matGap, gap, am_eigenvalues, gm_eigenvalues hH]

/-- Kronecker additivity: the gap of `A ⊗ B` is the sum of the gaps. -/
theorem matGap_kronecker {m : Type*} [Fintype m] [DecidableEq m] [Nonempty m]
    [Nonempty n] {A : Matrix m m ℝ} {B : Matrix n n ℝ}
    (hA : A.PosDef) (hB : B.PosDef) :
    matGap (Matrix.kroneckerMap (· * ·) A B) = matGap A + matGap B := by
  have hm : (0 : ℝ) < Fintype.card m := by exact_mod_cast Fintype.card_pos
  have hn : (0 : ℝ) < Fintype.card n := by exact_mod_cast Fintype.card_pos
  have hdA : 0 < A.det := hA.det_pos
  have hdB : 0 < B.det := hB.det_pos
  have htA : 0 < A.trace := hA.trace_pos
  have htB : 0 < B.trace := hB.trace_pos
  have hdet : (Matrix.kroneckerMap (· * ·) A B).det ^ (1 / (Fintype.card (m × n) : ℝ)) =
      A.det ^ (1 / (Fintype.card m : ℝ)) * B.det ^ (1 / (Fintype.card n : ℝ)) := by
    rw [Matrix.det_kronecker, Fintype.card_prod,
      Real.mul_rpow (pow_pos hdA _).le (pow_pos hdB _).le, ← Real.rpow_natCast,
      ← Real.rpow_natCast, ← Real.rpow_mul hdA.le, ← Real.rpow_mul hdB.le]
    congr 2 <;> · push_cast; field_simp
  have htr : (Matrix.kroneckerMap (· * ·) A B).trace / (Fintype.card (m × n) : ℝ) =
      (A.trace / (Fintype.card m : ℝ)) * (B.trace / (Fintype.card n : ℝ)) := by
    rw [Matrix.trace_kronecker, Fintype.card_prod]
    push_cast
    field_simp
  have hxA : 0 < A.det ^ (1 / (Fintype.card m : ℝ)) := Real.rpow_pos_of_pos hdA _
  have hxB : 0 < B.det ^ (1 / (Fintype.card n : ℝ)) := Real.rpow_pos_of_pos hdB _
  rw [matGap, hdet, htr, matGap, matGap, ← mul_add,
    ← Real.logb_mul (div_pos (div_pos htA hm) hxA).ne' (div_pos (div_pos htB hn) hxB).ne']
  congr 2
  field_simp

end MatrixGap

end ApexQuant.Oblivious
