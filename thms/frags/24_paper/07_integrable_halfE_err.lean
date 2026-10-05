
theorem integrable_halfE_err [MeasurableSpace E] [BorelSpace E] {σ : Measure E}
    [IsProbabilityMeasure σ] (hσ : ∀ᵐ u ∂σ, ‖u‖ = 1) :
    Integrable (fun u : E => ‖u - halfE u‖ ^ 2) σ := by
  have h : ∀ᵐ u ∂σ, ‖u - halfE u‖ ^ 2 = 1 / 4 := by
    filter_upwards [hσ] with u hu
    rw [norm_sub_halfE, hu]; norm_num
  exact (integrable_const (1 / 4 : ℝ)).congr (h.mono fun u hu => hu.symm)

end Halving

section PaperStatement
open Orthogonal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
