private theorem aux_orbit (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    ∀ᵐ ω ∂μ, residDirPast q p (W ω, fun i : Fin p => R ω i) ∈ MulAction.orbit G x₀ := by
  filter_upwards [h.exact_never p] with ω hω
  exact h.transitive _ (norm_residDirPast h.implements hω)

omit [BorelSpace E] [SecondCountableTopology E] [MeasurableSMul₂ G E] [MeasurableInv G] in
