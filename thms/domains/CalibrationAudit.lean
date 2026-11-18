import Mathlib

open Finset

namespace ApexQuant.DomainShift
def weightedShift {X : Type*} [Fintype X]
    (μ : X → ℝ) (reference candidate : X → ℝ) : ℝ :=
  ∑ x, μ x * |candidate x - reference x|


def SourceCalibrationPattern (sourceDrift targetDrift sourceControl targetControl tolerance : ℝ) : Prop :=
  targetControl < targetDrift ∧
    |sourceDrift - sourceControl| ≤ tolerance

/-- A measured target gap and a bounded source gap certify the empirical pattern
used in the calibration audit. -/
theorem source_calibration_pattern_of_measurements
    {sourceDrift targetDrift sourceControl targetControl tolerance : ℝ}
    (htarget : targetControl < targetDrift)
    (hsource : |sourceDrift - sourceControl| ≤ tolerance) :
    SourceCalibrationPattern sourceDrift targetDrift sourceControl targetControl tolerance :=
  ⟨htarget, hsource⟩

/-- The source and target distributions can overlap while target shift is larger.
This avoids relying only on disjoint point masses. -/
theorem overlapping_source_target_no_transfer :
    ∃ (μsource μtarget : Bool → ℝ) (reference candidate : Bool → ℝ),
      (∀ x, 0 < μsource x) ∧
      (∀ x, 0 < μtarget x) ∧
      (∑ x, μsource x = 1) ∧
      (∑ x, μtarget x = 1) ∧
      weightedShift μsource reference candidate = (1 / 4 : ℝ) ∧
      weightedShift μtarget reference candidate = (3 / 4 : ℝ) := by
  let μsource : Bool → ℝ := fun x => if x then 1 / 4 else 3 / 4
  let μtarget : Bool → ℝ := fun x => if x then 3 / 4 else 1 / 4
  let reference : Bool → ℝ := fun _ => 0
  let candidate : Bool → ℝ := fun x => if x then 1 else 0
  refine ⟨μsource, μtarget, reference, candidate, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro x
    cases x <;> norm_num [μsource]
  · intro x
    cases x <;> norm_num [μtarget]
  · norm_num [μsource]
  · norm_num [μtarget]
  · norm_num [weightedShift, μsource, reference, candidate]
  · norm_num [weightedShift, μtarget, reference, candidate]

end ApexQuant.DomainShift
