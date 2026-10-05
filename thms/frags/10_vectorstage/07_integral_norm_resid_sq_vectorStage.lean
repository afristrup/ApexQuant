
/-- **The decay law is quantiser-agnostic.**  A restatement of `integral_norm_resid_sq` whose only
purpose is to be citable at an arbitrary stage quantiser: `Scheme.q` is an unrestricted map, and
neither `IsotropicPasses` nor the proof of §3 mentions coordinates, so a trellis-coded or lattice
stage quantiser inherits the same `ρ^{2P}` envelope with no change.  (This is definitionally the
same theorem; it is named separately because the paper's prose ties `ρ` to a scalar codebook and
the independence is worth making explicit.) -/
theorem integral_norm_resid_sq_vectorStage {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {S : Ω → Scheme E} {W : Ω → E} {ρ : ℝ}
    (h : IsotropicPasses S W ρ μ) (P : ℕ) :
    ∫ ω, ‖resid (S ω) (W ω) P‖ ^ 2 ∂μ = ρ ^ (2 * P) * ∫ ω, ‖W ω‖ ^ 2 ∂μ :=
  integral_norm_resid_sq h P

