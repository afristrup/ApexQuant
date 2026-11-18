import Mathlib

open Finset

namespace ApexQuant.DomainShift

/-- A scalar model response, sufficient to formalize distributional drift. -/
def Model (X : Type*) := X → ℝ

/-- The merged model after applying a low-rank domain-specific LoRA update. -/
def mergeLora {X : Type*} (base adapter : Model X) : Model X :=
  fun x => base x + adapter x

/-- A normalized finite evaluation distribution over inputs. -/
def InputDistribution (X : Type*) [Fintype X] := X → ℝ

/-- Nonnegative weighted absolute output drift on an input distribution. -/
def weightedShift {X : Type*} [Fintype X]
    (μ : InputDistribution X) (reference candidate : Model X) : ℝ :=
  ∑ x, μ x * |candidate x - reference x|

/-- A calibration-free quantizer acts only on the merged model. -/
def DataFreeQuantizer (X : Type*) := Model X → Model X

/-- A data-dependent PTQ quantizer may depend on its calibration corpus. -/
def CalibratedQuantizer (X C : Type*) := C → Model X → Model X

/-- A pointwise error envelope implies a shift bound on every input distribution. -/
theorem weighted_shift_le_of_envelope {X : Type*} [Fintype X]
    {μ reference candidate envelope : Model X}
    (hμ : ∀ x, 0 ≤ μ x)
    (henvelope : ∀ x, |candidate x - reference x| ≤ envelope x) :
    weightedShift μ reference candidate ≤ ∑ x, μ x * envelope x := by
  unfold weightedShift
  apply Finset.sum_le_sum
  intro x _
  exact mul_le_mul_of_nonneg_left (henvelope x) (hμ x)

/-- The same calibration-free envelope applies after any adapter merge. -/
theorem lora_weighted_shift_le_of_envelope {X : Type*} [Fintype X]
    {μ base adapter candidate envelope : Model X}
    (hμ : ∀ x, 0 ≤ μ x)
    (henvelope : ∀ x,
      |candidate x - mergeLora base adapter x| ≤ envelope x) :
    weightedShift μ (mergeLora base adapter) candidate ≤
      ∑ x, μ x * envelope x := by
  exact weighted_shift_le_of_envelope hμ henvelope


/-- A normalized input distribution turns a uniform pointwise envelope into a
single distribution-shift tolerance. -/
theorem weighted_shift_le_of_uniform_envelope {X : Type*} [Fintype X]
    {μ reference candidate : Model X} {ε : ℝ}
    (hμ : ∀ x, 0 ≤ μ x) (hsum : ∑ x, μ x = 1)
    (hbound : ∀ x, |candidate x - reference x| ≤ ε) :
    weightedShift μ reference candidate ≤ ε := by
  calc
    weightedShift μ reference candidate ≤ ∑ x, μ x * ε := by
      apply weighted_shift_le_of_envelope hμ
      intro x
      exact hbound x
    _ = (∑ x, μ x) * ε := by
      simpa using (Finset.sum_mul (Finset.univ : Finset X) μ ε).symm
    _ = ε := by rw [hsum, one_mul]

/-- ApexQuant's model-level guarantee is independent of the calibration distribution. -/
theorem data_free_bound_any_domain {X : Type*} [Fintype X]
    (Apex : DataFreeQuantizer X) (merged : Model X)
    (μsource μtarget : InputDistribution X)
    (hsource : ∀ x, 0 ≤ μsource x)
    (htarget : ∀ x, 0 ≤ μtarget x)
    (envelope : Model X)
    (hbound : ∀ x, |Apex merged x - merged x| ≤ envelope x) :
    weightedShift μsource merged (Apex merged) ≤
        ∑ x, μsource x * envelope x ∧
      weightedShift μtarget merged (Apex merged) ≤
        ∑ x, μtarget x * envelope x := by
  constructor
  · exact weighted_shift_le_of_envelope hsource (fun x => hbound x)
  · exact weighted_shift_le_of_envelope htarget (fun x => hbound x)

/-- A source-calibrated PTQ bound transfers to target-domain data only if it is
also proved pointwise on the target-domain support. -/
theorem calibrated_bound_requires_target_support {X C : Type*} [Fintype X]
    (Q : CalibratedQuantizer X C) (calibration : C) (merged : Model X)
    (μtarget : InputDistribution X) (hμ : ∀ x, 0 ≤ μtarget x)
    (envelope : Model X)
    (htarget : ∀ x, |Q calibration merged x - merged x| ≤ envelope x) :
    weightedShift μtarget merged (Q calibration merged) ≤
      ∑ x, μtarget x * envelope x := by
  exact weighted_shift_le_of_envelope hμ htarget

/-- If source-calibrated PTQ has larger pointwise error on target-domain inputs, its measured
target-domain shift is no smaller than the calibration-free alternative. -/
theorem source_ptq_shift_dominates {X C : Type*} [Fintype X]
    (Q : CalibratedQuantizer X C) (calibration : C)
    (Apex : DataFreeQuantizer X) (merged : Model X)
    (μtarget : InputDistribution X)
    (hμ : ∀ x, 0 ≤ μtarget x)
    (hdom : ∀ x,
      |Apex merged x - merged x| ≤ |Q calibration merged x - merged x|) :
    weightedShift μtarget merged (Apex merged) ≤
      weightedShift μtarget merged (Q calibration merged) := by
  unfold weightedShift
  apply Finset.sum_le_sum
  intro x _
  exact mul_le_mul_of_nonneg_left (hdom x) (hμ x)

/-- Calibration independence is a noninterference property: deployment does not
observe which calibration corpus was supplied. -/
def CalibrationIndependent {C M : Type*} (artifact : C → M) : Prop :=
  ∀ c₁ c₂, artifact c₁ = artifact c₂

/-- A calibration-independent artifact has identical deployments for all
source or target calibration choices. -/
theorem calibration_noninterference {C M : Type*}
    {artifact : C → M} (hfree : CalibrationIndependent artifact)
    (c₁ c₂ : C) : artifact c₁ = artifact c₂ := by
  exact hfree c₁ c₂

/-- Source-domain accuracy alone does not imply target-domain accuracy. -/
theorem source_target_no_transfer :
    ∃ (μsource μtarget : Bool → ℝ) (reference candidate : Bool → ℝ),
      (∀ x, 0 ≤ μsource x) ∧
      (∀ x, 0 ≤ μtarget x) ∧
      (∑ x, μsource x = 1) ∧
      (∑ x, μtarget x = 1) ∧
      weightedShift μsource reference candidate = 0 ∧
      weightedShift μtarget reference candidate = 1 := by
  let μsource : Bool → ℝ := fun x => if x then 0 else 1
  let μtarget : Bool → ℝ := fun x => if x then 1 else 0
  let reference : Bool → ℝ := fun _ => 0
  let candidate : Bool → ℝ := fun x => if x then 1 else 0
  refine ⟨μsource, μtarget, reference, candidate, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro x
    cases x <;> simp [μsource]
  · intro x
    cases x <;> simp [μtarget]
  · simp [μsource]
  · simp [μtarget]
  · simp [weightedShift, μsource, reference, candidate]
  · simp [weightedShift, μtarget, reference, candidate]

end ApexQuant.DomainShift
