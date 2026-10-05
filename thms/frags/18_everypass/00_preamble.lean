/-! ## §18  Every pass, for every stage

§16 and §17 compare stage distortions.  This section pushes the comparison through §11 to the
quantity the paper measures, the expected squared residual after `P` passes, for any two stages
run through the same rotation stream from the same weight rows.  Nothing is assumed about the
stages beyond what `RotationStream` already asks of them (measurable, never exact), so the
statements cover scalar, `E₈` and trellis stages alike; `integral_norm_resid_sq_blockStage` is
Proposition 2 for a block stage with `ρ²` supplied by §17's block quadrature. -/

section EveryPass

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G] [MeasurableMul G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G}
  [κ.IsMulRightInvariant] {x₀ : E}

omit [BorelSpace E] [SecondCountableTopology E] [MeasurableInv G] [MeasurableMul G]
  [IsProbabilityMeasure μ] [κ.IsMulRightInvariant] in
