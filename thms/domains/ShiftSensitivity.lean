import Mathlib

open Finset MeasureTheory

namespace ApexQuant.ShiftSensitivity

variable {m n : Type*} [Fintype m] [Fintype n]

/-- The quadratic form `eᵀ Σ e` of a second-moment matrix on an error row. -/
def q (S : Matrix n n ℝ) (e : n → ℝ) : ℝ :=
  ∑ i, ∑ j, e i * S i j * e j

/-- The activation-weighted error proxy: the sum of the row quadratic forms. -/
def proxy (S : Matrix n n ℝ) (E : Matrix m n ℝ) : ℝ :=
  ∑ r, q S (E r)

/-- The quadratic form is the dot product of `e` with `Σ e`. -/
theorem q_eq_dotProduct (S : Matrix n n ℝ) (e : n → ℝ) :
    q S e = dotProduct e (S.mulVec e) := by
  simp only [q, dotProduct, Matrix.mulVec, Finset.mul_sum, mul_assoc]

/-- The proxy is the trace of `E Σ Eᵀ`. -/
theorem proxy_eq_trace [DecidableEq n] (S : Matrix n n ℝ) (E : Matrix m n ℝ) :
    proxy S E = Matrix.trace (E * S * E.transpose) := by
  simp only [proxy, q, Matrix.trace, Matrix.diag, Matrix.mul_apply,
    Matrix.transpose_apply, Finset.sum_mul]
  refine Finset.sum_congr rfl fun r _ => ?_
  exact Finset.sum_comm

/-- A Loewner bound `Σ_t ≼ κ Σ_s` on quadratic forms transfers to the proxy. -/
theorem proxy_le_of_loewner {Sigma_s Sigma_t : Matrix n n ℝ} {κ : ℝ}
    (hκ : ∀ e, q Sigma_t e ≤ κ * q Sigma_s e) (E : Matrix m n ℝ) :
    proxy Sigma_t E ≤ κ * proxy Sigma_s E := by
  unfold proxy
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun r _ => hκ (E r)

/-- If every error row attains the Loewner constant, the proxy bound is an
equality. -/
theorem proxy_eq_of_top_direction {Sigma_s Sigma_t : Matrix n n ℝ} {κ : ℝ}
    (E : Matrix m n ℝ)
    (hrow : ∀ r, q Sigma_t (E r) = κ * q Sigma_s (E r)) :
    proxy Sigma_t E = κ * proxy Sigma_s E := by
  unfold proxy
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun r _ => hrow r

/-- Sharpness: if some direction attains `κ` with positive source energy, no
constant below `κ` bounds the target proxy for every error matrix. -/
theorem no_smaller_constant {Sigma_s Sigma_t : Matrix n n ℝ} {κ c : ℝ}
    {e₀ : n → ℝ}
    (hattain : q Sigma_t e₀ = κ * q Sigma_s e₀)
    (hpos : 0 < q Sigma_s e₀) (hc : c < κ) :
    ∃ E : Matrix Unit n ℝ, c * proxy Sigma_s E < proxy Sigma_t E := by
  refine ⟨fun _ => e₀, ?_⟩
  simp only [proxy, Finset.univ_unique, Finset.sum_singleton]
  rw [hattain]
  exact mul_lt_mul_of_pos_right hc hpos

/-- A random vector whose second-moment matrix is `c` times the identity has
expected quadratic form `c * trace Σ`. -/
theorem integral_q_of_second_moment [DecidableEq n]
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    {e : Ω → n → ℝ} {c : ℝ} (S : Matrix n n ℝ)
    (hint : ∀ i j, Integrable (fun ω => e ω i * e ω j) μ)
    (hiso : ∀ i j, ∫ ω, e ω i * e ω j ∂μ = if i = j then c else 0) :
    ∫ ω, q S (e ω) ∂μ = c * Matrix.trace S := by
  have hq : (fun ω => q S (e ω)) =
      fun ω => ∑ i, ∑ j, S i j * (e ω i * e ω j) := by
    funext ω
    unfold q
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
  rw [hq, integral_finsetSum _ fun i _ =>
    integrable_finsetSum _ fun j _ => (hint i j).const_mul (S i j)]
  simp_rw [integral_finsetSum _ fun j _ => (hint _ j).const_mul (S _ j),
    integral_const_mul, hiso]
  simp [Matrix.trace, Finset.mul_sum, mul_comm]

/-- Isotropic error on a probability space: the expected proxy of a row is
`σ² * trace Σ`. -/
theorem expected_q_of_isotropic [DecidableEq n]
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {e : Ω → n → ℝ} {σ2 : ℝ} (S : Matrix n n ℝ)
    (hint : ∀ i j, Integrable (fun ω => e ω i * e ω j) μ)
    (hiso : ∀ i j, ∫ ω, e ω i * e ω j ∂μ = if i = j then σ2 else 0) :
    ∫ ω, q S (e ω) ∂μ = σ2 * Matrix.trace S :=
  integral_q_of_second_moment S hint hiso

/-- Isotropic error: the target to source ratio of expected proxies is the
ratio of traces. -/
theorem expected_q_ratio_of_isotropic [DecidableEq n]
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {e : Ω → n → ℝ} {σ2 : ℝ} (Sigma_s Sigma_t : Matrix n n ℝ)
    (hint : ∀ i j, Integrable (fun ω => e ω i * e ω j) μ)
    (hiso : ∀ i j, ∫ ω, e ω i * e ω j ∂μ = if i = j then σ2 else 0)
    (htr : Matrix.trace Sigma_s ≠ 0) :
    ∫ ω, q Sigma_t (e ω) ∂μ =
      (Matrix.trace Sigma_t / Matrix.trace Sigma_s) *
        ∫ ω, q Sigma_s (e ω) ∂μ := by
  rw [expected_q_of_isotropic Sigma_t hint hiso,
    expected_q_of_isotropic Sigma_s hint hiso]
  field_simp

/-! ### The sharp shift constant as a largest generalized Rayleigh quotient -/

/-- The quadratic form vanishes at zero. -/
theorem q_zero (S : Matrix n n ℝ) : q S 0 = 0 := by
  simp [q]

/-- The quadratic form is homogeneous of degree two. -/
theorem q_smul (S : Matrix n n ℝ) (c : ℝ) (e : n → ℝ) :
    q S (c • e) = c ^ 2 * q S e := by
  simp only [q, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

/-- The quadratic form is continuous in the error row. -/
theorem continuous_q (S : Matrix n n ℝ) : Continuous (q S) := by
  unfold q
  fun_prop

/-- A positive definite matrix has a positive quadratic form off zero. -/
theorem q_pos_of_posDef {S : Matrix n n ℝ} (hS : S.PosDef) {e : n → ℝ}
    (he : e ≠ 0) : 0 < q S e := by
  have h := hS.dotProduct_mulVec_pos he
  rwa [star_trivial, ← q_eq_dotProduct] at h

/-- Rescaling a nonzero row to unit source energy. -/
theorem q_normalize {S : Matrix n n ℝ} (T : Matrix n n ℝ) {e : n → ℝ}
    (hpos : 0 < q S e) :
    q S ((Real.sqrt (q S e))⁻¹ • e) = 1 ∧
      q T ((Real.sqrt (q S e))⁻¹ • e) = q T e / q S e := by
  have hsq : (Real.sqrt (q S e))⁻¹ ^ 2 = (q S e)⁻¹ := by
    rw [inv_pow, Real.sq_sqrt hpos.le]
  refine ⟨?_, ?_⟩
  · rw [q_smul, hsq, inv_mul_cancel₀ hpos.ne']
  · rw [q_smul, hsq, div_eq_inv_mul]

/-- The sharp shift constant: the largest value of the target quadratic form
on the unit level set of the source quadratic form, that is the largest
generalized Rayleigh quotient of the pair `(Sigma_t, Sigma_s)`. -/
noncomputable def kappa (Sigma_s Sigma_t : Matrix n n ℝ) : ℝ :=
  sSup {ρ | ∃ e, q Sigma_s e = 1 ∧ ρ = q Sigma_t e}

section Attained

variable [Nonempty n] {Sigma_s : Matrix n n ℝ} (Sigma_t : Matrix n n ℝ)

/-- The target form attains a maximum on the unit level set of a positive
definite source form. Compactness comes from the unit sphere, on which the
Rayleigh quotient is continuous. -/
theorem exists_max_on_level (hS : Sigma_s.PosDef) :
    ∃ e₀, q Sigma_s e₀ = 1 ∧ ∀ e, q Sigma_s e = 1 → q Sigma_t e ≤ q Sigma_t e₀ := by
  have hne : (Metric.sphere (0 : n → ℝ) 1).Nonempty :=
    NormedSpace.sphere_nonempty.mpr zero_le_one
  have hnz : ∀ e ∈ Metric.sphere (0 : n → ℝ) 1, e ≠ 0 := fun e he h0 => by
    rw [h0] at he
    simp at he
  have hcont : ContinuousOn (fun e => q Sigma_t e / q Sigma_s e)
      (Metric.sphere (0 : n → ℝ) 1) :=
    (continuous_q Sigma_t).continuousOn.div (continuous_q Sigma_s).continuousOn
      fun e he => (q_pos_of_posDef hS (hnz e he)).ne'
  obtain ⟨u, hu, hmax⟩ := (isCompact_sphere (0 : n → ℝ) 1).exists_isMaxOn hne hcont
  have hupos : 0 < q Sigma_s u := q_pos_of_posDef hS (hnz u hu)
  obtain ⟨h1, h2⟩ := q_normalize Sigma_t hupos
  refine ⟨(Real.sqrt (q Sigma_s u))⁻¹ • u, h1, fun e he => ?_⟩
  have he0 : e ≠ 0 := fun h0 => by
    rw [h0, q_zero] at he
    exact zero_ne_one he
  have hnorm : 0 < ‖e‖ := norm_pos_iff.mpr he0
  have hw : ‖e‖⁻¹ • e ∈ Metric.sphere (0 : n → ℝ) 1 := by
    rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, norm_norm,
      inv_mul_cancel₀ hnorm.ne']
  have hle := isMaxOn_iff.mp hmax _ hw
  have hsq : (‖e‖⁻¹) ^ 2 ≠ 0 := pow_ne_zero 2 (inv_ne_zero hnorm.ne')
  simp only [q_smul] at hle
  rw [mul_div_mul_left _ _ hsq, he, div_one] at hle
  rw [h2]
  exact hle

/-- `kappa` is the greatest element of the set of target energies at unit
source energy, so the supremum in its definition is a maximum. -/
theorem isGreatest_kappa (hS : Sigma_s.PosDef) :
    IsGreatest {ρ | ∃ e, q Sigma_s e = 1 ∧ ρ = q Sigma_t e} (kappa Sigma_s Sigma_t) := by
  obtain ⟨e₀, h1, hmax⟩ := exists_max_on_level Sigma_t hS
  have hg : IsGreatest {ρ | ∃ e, q Sigma_s e = 1 ∧ ρ = q Sigma_t e} (q Sigma_t e₀) :=
    ⟨⟨e₀, h1, rfl⟩, fun ρ ⟨e, he, hρ⟩ => hρ ▸ hmax e he⟩
  rw [kappa, hg.csSup_eq]
  exact hg

/-- The sharp constant is attained on a direction of unit source energy. -/
theorem exists_kappa_attained (hS : Sigma_s.PosDef) :
    ∃ e₀, q Sigma_s e₀ = 1 ∧ 0 < q Sigma_s e₀ ∧
      q Sigma_t e₀ = kappa Sigma_s Sigma_t * q Sigma_s e₀ := by
  obtain ⟨e₀, h1, hρ⟩ := (isGreatest_kappa Sigma_t hS).1
  exact ⟨e₀, h1, h1 ▸ one_pos, by rw [h1, mul_one, hρ]⟩

/-- The Loewner bound `Σ_t ≼ κ Σ_s` holds with `κ = kappa`. -/
theorem loewner_of_kappa (hS : Sigma_s.PosDef) (e : n → ℝ) :
    q Sigma_t e ≤ kappa Sigma_s Sigma_t * q Sigma_s e := by
  by_cases he : e = 0
  · rw [he, q_zero, q_zero, mul_zero]
  · have hpos : 0 < q Sigma_s e := q_pos_of_posDef hS he
    obtain ⟨h1, h2⟩ := q_normalize Sigma_t hpos
    have hle : q Sigma_t e / q Sigma_s e ≤ kappa Sigma_s Sigma_t :=
      (isGreatest_kappa Sigma_t hS).2 ⟨_, h1, h2.symm⟩
    exact (div_le_iff₀ hpos).mp hle

/-- Shift sensitivity with the explicit sharp constant: the only hypothesis is
positive definiteness of the source second-moment matrix. -/
theorem proxy_le_kappa (hS : Sigma_s.PosDef) (E : Matrix m n ℝ) :
    proxy Sigma_t E ≤ kappa Sigma_s Sigma_t * proxy Sigma_s E :=
  proxy_le_of_loewner (loewner_of_kappa Sigma_t hS) E

/-- Sharpness: no constant below `kappa` bounds the target proxy for every
error matrix. -/
theorem kappa_sharp (hS : Sigma_s.PosDef) {c : ℝ} (hc : c < kappa Sigma_s Sigma_t) :
    ∃ E : Matrix Unit n ℝ, c * proxy Sigma_s E < proxy Sigma_t E := by
  obtain ⟨e₀, _, hpos, hattain⟩ := exists_kappa_attained Sigma_t hS
  exact no_smaller_constant hattain hpos hc

/-- Minimality: `kappa` is the least constant in the Loewner bound. -/
theorem kappa_least (hS : Sigma_s.PosDef) {c : ℝ}
    (hc : ∀ e, q Sigma_t e ≤ c * q Sigma_s e) : kappa Sigma_s Sigma_t ≤ c := by
  obtain ⟨e₀, h1, _, hattain⟩ := exists_kappa_attained Sigma_t hS
  have h := hc e₀
  rwa [hattain, h1, mul_one, mul_one] at h

end Attained

end ApexQuant.ShiftSensitivity
