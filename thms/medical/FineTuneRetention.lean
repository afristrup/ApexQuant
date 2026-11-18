import Mathlib

open Finset

namespace ApexQuant.Medical.Retention

/-- Weighted absolute drift of a scalar response on a finite input distribution. -/
def weightedShift {X : Type*} [Fintype X]
    (μ : X → ℝ) (reference candidate : X → ℝ) : ℝ :=
  ∑ x, μ x * |candidate x - reference x|

/-- Pointwise retention: the deployed change from the base model is at least the
fine-tune's change minus the quantization error. -/
theorem retention_pointwise {V W : Type*} [SeminormedAddCommGroup W]
    (Δ E : V → W) (v : V) :
    ‖Δ v‖ - ‖E v‖ ≤ ‖Δ v - E v‖ :=
  norm_sub_norm_le _ _

/-- Retention under a pointwise envelope, on every input distribution. If the
quantization error satisfies `‖E v‖ ≤ U ‖v‖`, then on any nonnegative weighting
of inputs the deployed model keeps the fine-tune's average effect up to
`U` times the average input norm. -/
theorem retention_of_envelope {I V W : Type*} [Fintype I]
    [SeminormedAddCommGroup V] [SeminormedAddCommGroup W]
    (μ : I → ℝ) (hμ : ∀ i, 0 ≤ μ i) (x : I → V) (Δ E : V → W) (U : ℝ)
    (hE : ∀ v, ‖E v‖ ≤ U * ‖v‖) :
    ∑ i, μ i * ‖Δ (x i)‖ - U * ∑ i, μ i * ‖x i‖ ≤
      ∑ i, μ i * ‖Δ (x i) - E (x i)‖ := by
  rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
  apply Finset.sum_le_sum
  intro i _
  have h1 : ‖Δ (x i)‖ - U * ‖x i‖ ≤ ‖Δ (x i) - E (x i)‖ := by
    have := norm_sub_norm_le (Δ (x i)) (E (x i))
    linarith [hE (x i)]
  calc μ i * ‖Δ (x i)‖ - U * (μ i * ‖x i‖)
      = μ i * (‖Δ (x i)‖ - U * ‖x i‖) := by ring
    _ ≤ μ i * ‖Δ (x i) - E (x i)‖ := mul_le_mul_of_nonneg_left h1 (hμ i)

/-- Reverting to the base model is invisible on the calibration support: if the
adapter vanishes wherever the source distribution has mass, returning the base
model has zero source drift and removes the whole fine-tune on the target. -/
theorem revert_invisible_on_source {X : Type*} [Fintype X]
    (base adapter μs μt : X → ℝ)
    (hsupp : ∀ x, μs x ≠ 0 → adapter x = 0) :
    weightedShift μs (fun x => base x + adapter x) base = 0 ∧
      weightedShift μt (fun x => base x + adapter x) base =
        ∑ x, μt x * |adapter x| := by
  constructor
  · unfold weightedShift
    apply Finset.sum_eq_zero
    intro x _
    by_cases h : μs x = 0
    · simp [h]
    · simp [hsupp x h]
  · unfold weightedShift
    apply Finset.sum_congr rfl
    intro x _
    congr 1
    rw [show base x - (base x + adapter x) = -adapter x by ring, abs_neg]

/-- The same invisibility for vector outputs and squared error, the calibration
proxy on an empirical source distribution: if the adapter vanishes on the source
support, reverting it costs nothing on the source while the target loses the
adapter's full squared effect. -/
theorem revert_invisible_sq {X W : Type*} [Fintype X] [SeminormedAddCommGroup W]
    (base adapter : X → W) (μs μt : X → ℝ)
    (hsupp : ∀ x, μs x ≠ 0 → adapter x = 0) :
    ∑ x, μs x * ‖base x - (base x + adapter x)‖ ^ 2 = 0 ∧
      ∑ x, μt x * ‖base x - (base x + adapter x)‖ ^ 2 =
        ∑ x, μt x * ‖adapter x‖ ^ 2 := by
  have hsub : ∀ x, base x - (base x + adapter x) = -adapter x := fun x => by abel
  constructor
  · apply Finset.sum_eq_zero
    intro x _
    by_cases h : μs x = 0
    · simp [h]
    · simp [hsupp x h]
  · apply Finset.sum_congr rfl
    intro x _
    rw [hsub, norm_neg]

/-- A source-calibrated choice can prefer reverting the fine-tune. With strictly
positive source and target distributions on two inputs, the candidate that
returns the base model has smaller source drift than a candidate with a uniform
error, and larger target drift. -/
theorem calibrated_choice_reverts :
    ∃ (μs μt base adapter keep : Bool → ℝ),
      (∀ x, 0 < μs x) ∧ (∀ x, 0 < μt x) ∧
      (∑ x, μs x = 1) ∧ (∑ x, μt x = 1) ∧
      weightedShift μs (fun x => base x + adapter x) base <
        weightedShift μs (fun x => base x + adapter x) keep ∧
      weightedShift μt (fun x => base x + adapter x) keep <
        weightedShift μt (fun x => base x + adapter x) base := by
  let μs : Bool → ℝ := fun x => if x then 1 / 8 else 7 / 8
  let μt : Bool → ℝ := fun x => if x then 7 / 8 else 1 / 8
  let base : Bool → ℝ := fun _ => 0
  let adapter : Bool → ℝ := fun x => if x then 1 else 0
  let keep : Bool → ℝ := fun x => adapter x + 1 / 4
  refine ⟨μs, μt, base, adapter, keep, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro x; cases x <;> norm_num [μs]
  · intro x; cases x <;> norm_num [μt]
  · norm_num [μs]
  · norm_num [μt]
  · norm_num [weightedShift, μs, base, adapter, keep]
  · norm_num [weightedShift, μt, base, adapter, keep]

end ApexQuant.Medical.Retention
