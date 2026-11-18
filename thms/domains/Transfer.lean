import Mathlib

open MeasureTheory
open scoped NNReal ENNReal

namespace ApexQuant.Transfer

variable {X : Type*} [MeasurableSpace X]

/-! ### 1. Pointwise envelopes transfer to every measure -/

/-- A pointwise error envelope integrates to an expected-error bound under ANY
measure `μ`. Nothing is assumed about `μ`: it need not be the calibration
distribution, a probability measure, or even finite. The envelope is a
property of the map, so it holds on whatever domain the inputs come from. -/
theorem envelope_transfer {F : Type*} [NormedAddCommGroup F]
    (μ : Measure X) {f fh : X → F} {env : X → ℝ}
    (henv : ∀ x, ‖f x - fh x‖ ≤ env x)
    (hnorm : Integrable (fun x => ‖f x - fh x‖) μ) (hint : Integrable env μ) :
    ∫ x, ‖f x - fh x‖ ∂μ ≤ ∫ x, env x ∂μ :=
  integral_mono hnorm hint henv

/-- Envelope transfer from measurability alone: if `f` and `fh` are (almost
everywhere strongly) measurable and the envelope is integrable, the error norm
is integrable and the bound holds. Again `μ` is an arbitrary measure. -/
theorem envelope_transfer_of_measurable {F : Type*} [NormedAddCommGroup F]
    (μ : Measure X) {f fh : X → F} {env : X → ℝ}
    (hf : AEStronglyMeasurable f μ) (hfh : AEStronglyMeasurable fh μ)
    (henv : ∀ x, ‖f x - fh x‖ ≤ env x) (hint : Integrable env μ) :
    Integrable (fun x => ‖f x - fh x‖) μ ∧
      ∫ x, ‖f x - fh x‖ ∂μ ≤ ∫ x, env x ∂μ := by
  have hnorm : Integrable (fun x => ‖f x - fh x‖) μ :=
    hint.mono' (hf.sub hfh).norm
      (Filter.Eventually.of_forall fun x => by rw [norm_norm]; exact henv x)
  exact ⟨hnorm, envelope_transfer μ henv hnorm hint⟩

/-- A uniform envelope `ε` on a probability measure bounds the expected error
by `ε`, whatever the probability measure is. -/
theorem uniform_envelope_transfer {F : Type*} [NormedAddCommGroup F]
    (μ : Measure X) [IsProbabilityMeasure μ] {f fh : X → F} {ε : ℝ}
    (henv : ∀ x, ‖f x - fh x‖ ≤ ε)
    (hnorm : Integrable (fun x => ‖f x - fh x‖) μ) :
    ∫ x, ‖f x - fh x‖ ∂μ ≤ ε := by
  have h := envelope_transfer μ henv hnorm (integrable_const ε)
  simpa using h

/-! ### 2. Density-ratio transfer -/

/-- If the target measure is dominated by `M` times the source measure, the
target expectation of a nonnegative source-integrable `h` is at most `M` times
its source expectation. -/
theorem density_ratio_transfer {μs μt : Measure X} {M : ℝ≥0}
    (hdom : μt ≤ M • μs) {h : X → ℝ} (hnn : ∀ x, 0 ≤ h x)
    (hint : Integrable h μs) :
    ∫ x, h x ∂μt ≤ M * ∫ x, h x ∂μs := by
  have h1 : ∫ x, h x ∂μt ≤ ∫ x, h x ∂(M • μs) :=
    integral_mono_measure hdom (Filter.Eventually.of_forall hnn) hint.smul_measure_nnreal
  rw [integral_smul_nnreal_measure] at h1
  simpa [NNReal.smul_def] using h1

/-- A density bounded by `M` almost everywhere gives measure domination. -/
theorem withDensity_le_smul {μs : Measure X} {ρ : X → ℝ≥0∞} {M : ℝ≥0}
    (hρ : ∀ᵐ x ∂μs, ρ x ≤ M) : μs.withDensity ρ ≤ M • μs := by
  have h := withDensity_mono (μ := μs) hρ
  rwa [withDensity_const] at h

/-- Density-ratio transfer when the target has a density `ρ ≤ M` with respect
to the source. -/
theorem density_ratio_transfer_withDensity {μs : Measure X} {ρ : X → ℝ≥0∞} {M : ℝ≥0}
    (hρ : ∀ᵐ x ∂μs, ρ x ≤ M) {h : X → ℝ} (hnn : ∀ x, 0 ≤ h x)
    (hint : Integrable h μs) :
    ∫ x, h x ∂(μs.withDensity ρ) ≤ M * ∫ x, h x ∂μs :=
  density_ratio_transfer (withDensity_le_smul hρ) hnn hint

/-- Density-ratio transfer of an error envelope: the target expected error is
at most `M` times the source expectation of the envelope. -/
theorem envelope_density_ratio_transfer {F : Type*} [NormedAddCommGroup F]
    {μs μt : Measure X} {M : ℝ≥0} (hdom : μt ≤ M • μs)
    {f fh : X → F} {env : X → ℝ} (henv : ∀ x, ‖f x - fh x‖ ≤ env x)
    (hnorm : Integrable (fun x => ‖f x - fh x‖) μt)
    (hint : Integrable env μs) :
    ∫ x, ‖f x - fh x‖ ∂μt ≤ M * ∫ x, env x ∂μs := by
  have hnn : ∀ x, 0 ≤ env x := fun x => (norm_nonneg _).trans (henv x)
  have hint_t : Integrable env μt := hint.smul_measure_nnreal.mono_measure hdom
  exact (envelope_transfer μt henv hnorm hint_t).trans
    (density_ratio_transfer hdom hnn hint)

end ApexQuant.Transfer
