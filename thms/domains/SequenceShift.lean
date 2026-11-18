import Mathlib

open Finset

namespace ApexQuant.DomainShift

/-- A finite autoregressive sequence shift is the sum of per-step shifts. -/
noncomputable def sequenceShift {T : Type*} [Fintype T]
    (stepShift : T → ℝ) (tokenWeight : T → ℝ) : ℝ :=
  ∑ t, tokenWeight t * stepShift t

/-- Nonnegative per-token bounds accumulate linearly over a sequence. -/
theorem sequence_shift_le_of_step_bounds {T : Type*} [Fintype T]
    {stepShift bound tokenWeight : T → ℝ}
    (hweight : ∀ t, 0 ≤ tokenWeight t)
    (hstep : ∀ t, stepShift t ≤ bound t) :
    sequenceShift stepShift tokenWeight ≤
      ∑ t, tokenWeight t * bound t := by
  unfold sequenceShift
  apply Finset.sum_le_sum
  intro t _
  exact mul_le_mul_of_nonneg_left (hstep t) (hweight t)

/-- A uniform per-token shift bound grows at most with total token mass. -/
theorem sequence_shift_le_of_uniform_bound {T : Type*} [Fintype T]
    {stepShift tokenWeight : T → ℝ} {ε : ℝ}
    (hweight : ∀ t, 0 ≤ tokenWeight t)
    (hstep : ∀ t, stepShift t ≤ ε) :
    sequenceShift stepShift tokenWeight ≤
      ∑ t, tokenWeight t * ε := by
  exact sequence_shift_le_of_step_bounds hweight hstep

/-- Any KL or JS bound can be transferred through the same sequence argument.
The divergence-specific analytic inequality is supplied separately. -/
theorem sequence_divergence_le_of_step_bound {T : Type*} [Fintype T]
    {divergence bound tokenWeight : T → ℝ}
    (hweight : ∀ t, 0 ≤ tokenWeight t)
    (hbound : ∀ t, divergence t ≤ bound t) :
    (∑ t, tokenWeight t * divergence t) ≤
      ∑ t, tokenWeight t * bound t := by
  apply Finset.sum_le_sum
  intro t _
  exact mul_le_mul_of_nonneg_left (hbound t) (hweight t)

end ApexQuant.DomainShift
