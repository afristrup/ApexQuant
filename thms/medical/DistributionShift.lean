import Mathlib

open Finset

namespace ApexQuant.Medical

/-- A scalar model response, sufficient to formalize distributional drift. -/
def Model (X : Type*) := X → ℝ

/-- The merged model after applying a low-rank medical LoRA update. -/
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

/-- The same calibration-free envelope applies after any LoRA merge. -/
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

/-- ApexQuant's model-level guarantee is independent of the calibration domain. -/
theorem data_free_bound_any_domain {X : Type*} [Fintype X]
    (Apex : DataFreeQuantizer X) (merged : Model X)
    (μgeneric μmedical : InputDistribution X)
    (hgeneric : ∀ x, 0 ≤ μgeneric x)
    (hmedical : ∀ x, 0 ≤ μmedical x)
    (envelope : Model X)
    (hbound : ∀ x, |Apex merged x - merged x| ≤ envelope x) :
    weightedShift μgeneric merged (Apex merged) ≤
        ∑ x, μgeneric x * envelope x ∧
      weightedShift μmedical merged (Apex merged) ≤
        ∑ x, μmedical x * envelope x := by
  constructor
  · exact weighted_shift_le_of_envelope hgeneric (fun x => hbound x)
  · exact weighted_shift_le_of_envelope hmedical (fun x => hbound x)

/-- A generic-calibrated PTQ bound transfers to medical data only if it is
also proved pointwise on the medical support. -/
theorem calibrated_bound_requires_medical_support {X C : Type*} [Fintype X]
    (Q : CalibratedQuantizer X C) (calibration : C) (merged : Model X)
    (μmedical : InputDistribution X) (hμ : ∀ x, 0 ≤ μmedical x)
    (envelope : Model X)
    (hmedical : ∀ x, |Q calibration merged x - merged x| ≤ envelope x) :
    weightedShift μmedical merged (Q calibration merged) ≤
      ∑ x, μmedical x * envelope x := by
  exact weighted_shift_le_of_envelope hμ hmedical

/-- If generic PTQ has larger pointwise error on medical inputs, its measured
medical shift is no smaller than the calibration-free alternative. -/
theorem generic_ptq_shift_dominates {X C : Type*} [Fintype X]
    (Q : CalibratedQuantizer X C) (calibration : C)
    (Apex : DataFreeQuantizer X) (merged : Model X)
    (μmedical : InputDistribution X)
    (hμ : ∀ x, 0 ≤ μmedical x)
    (hdom : ∀ x,
      |Apex merged x - merged x| ≤ |Q calibration merged x - merged x|) :
    weightedShift μmedical merged (Apex merged) ≤
      weightedShift μmedical merged (Q calibration merged) := by
  unfold weightedShift
  apply Finset.sum_le_sum
  intro x _
  exact mul_le_mul_of_nonneg_left (hdom x) (hμ x)

end ApexQuant.Medical
