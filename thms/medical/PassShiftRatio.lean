import Mathlib

open Matrix Finset

namespace ApexQuant.Medical.PassShift

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- Output energy of an error with second moment `a r rᵀ + b (I - r rᵀ)` on inputs
with second moment `Σ`: the trace contraction of the pass covariance. -/
theorem trace_passCov (S : Matrix n n ℝ) (r : n → ℝ) (a b : ℝ) :
    trace (S * (a • vecMulVec r r + b • (1 - vecMulVec r r))) =
      a * (r ⬝ᵥ S *ᵥ r) + b * (trace S - r ⬝ᵥ S *ᵥ r) := by
  have h : trace (S * vecMulVec r r) = r ⬝ᵥ S *ᵥ r := by
    simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.vecMulVec_apply, dotProduct, Matrix.mulVec]
    apply Finset.sum_congr rfl
    intro i _
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [Matrix.mul_add, Matrix.mul_smul, Matrix.mul_smul, Matrix.mul_sub, Matrix.mul_one,
    trace_add, trace_smul, trace_smul, trace_sub, h, smul_eq_mul, smul_eq_mul]

/-- Mediant inequality: a ratio of sums with positive denominators is at most the
larger of the two part ratios. -/
theorem mediant_le_max {a₁ b₁ a₂ b₂ : ℝ} (ha₂ : 0 < a₂) (hb₂ : 0 < b₂) :
    (a₁ + b₁) / (a₂ + b₂) ≤ max (a₁ / a₂) (b₁ / b₂) := by
  rw [div_le_iff₀ (by linarith)]
  set m := max (a₁ / a₂) (b₁ / b₂)
  have h1 : a₁ ≤ m * a₂ := by
    have : a₁ / a₂ ≤ m := le_max_left _ _
    rwa [div_le_iff₀ ha₂] at this
  have h2 : b₁ ≤ m * b₂ := by
    have : b₁ / b₂ ≤ m := le_max_right _ _
    rwa [div_le_iff₀ hb₂] at this
  nlinarith

/-- Target-to-source ratio of a Haar pass. With axial weight `α ≥ 0` and transverse
weight `c ≥ 0`, the expected output error on the target divided by that on the
source is at most the larger of the ratio along the residual direction and the
ratio of the transverse traces. No calibration objective chooses either. -/
theorem pass_shift_ratio_le {α c at_ as tt ts : ℝ}
    (hα : 0 ≤ α) (hc : 0 ≤ c)
    (has : 0 < α * as) (hts : 0 < c * ts) :
    (α * at_ + c * tt) / (α * as + c * ts) ≤ max (at_ / as) (tt / ts) := by
  have hαpos : 0 < α := by
    rcases hα.lt_or_eq with h | h
    · exact h
    · rw [← h, zero_mul] at has; exact absurd has (lt_irrefl 0)
  have hcpos : 0 < c := by
    rcases hc.lt_or_eq with h | h
    · exact h
    · rw [← h, zero_mul] at hts; exact absurd hts (lt_irrefl 0)
  have key := mediant_le_max (a₁ := α * at_) (b₁ := c * tt) has hts
  rwa [mul_div_mul_left _ _ hαpos.ne', mul_div_mul_left _ _ hcpos.ne'] at key

/-- The axial part of a pass error is at most the whole error, pointwise: for a unit
`u`, `(1 - ⟪u, q⟫)² ≤ ‖u - q‖²`. -/
theorem axial_sq_le {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (u q : E) (hu : ‖u‖ = 1) :
    (1 - inner ℝ u q) ^ 2 ≤ ‖u - q‖ ^ 2 := by
  have h1 : 1 - inner ℝ u q = inner ℝ u (u - q) := by
    rw [inner_sub_right, real_inner_self_eq_norm_sq, hu, one_pow]
  rw [h1]
  have h2 : |inner ℝ u (u - q)| ≤ ‖u - q‖ := by
    calc |inner ℝ u (u - q)| ≤ ‖u‖ * ‖u - q‖ := abs_real_inner_le_norm _ _
      _ = ‖u - q‖ := by rw [hu, one_mul]
  nlinarith [abs_nonneg (inner ℝ u (u - q)), sq_abs (inner ℝ u (u - q))]

/-- Hence `α ≤ ρ²`: for any probability law of unit directions, the expected axial
part is at most the expected stage distortion, so the transverse weight
`(ρ² - α)/(d - 1)` of the pass covariance is nonnegative. -/
theorem alpha_le_rhoSq {E Ω : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω) (U : Ω → E) (q : E → E)
    (hU : ∀ ω, ‖U ω‖ = 1)
    (hint₁ : MeasureTheory.Integrable (fun ω => (1 - inner ℝ (U ω) (q (U ω))) ^ 2) μ)
    (hint₂ : MeasureTheory.Integrable (fun ω => ‖U ω - q (U ω)‖ ^ 2) μ) :
    ∫ ω, (1 - inner ℝ (U ω) (q (U ω))) ^ 2 ∂μ ≤ ∫ ω, ‖U ω - q (U ω)‖ ^ 2 ∂μ :=
  MeasureTheory.integral_mono hint₁ hint₂ (fun ω => axial_sq_le _ _ (hU ω))

end ApexQuant.Medical.PassShift
