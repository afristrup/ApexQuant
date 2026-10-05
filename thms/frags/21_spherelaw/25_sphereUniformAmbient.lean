
/-- The uniform direction law as a measure on the ambient space `E`, supported on the sphere:
the measure §11 calls `orbitLaw κ x₀` when the rotation is Haar, and the `σ` against which
§9 and §17 evaluate `ρ²`. -/
noncomputable def sphereUniformAmbient (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [MeasurableSpace E] [BorelSpace E] [FiniteDimensional ℝ E] : Measure E :=
  (sphereUniform E).map Subtype.val
