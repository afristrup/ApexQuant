
/-- The law of a `U`-block of a uniform direction: the density `(1 − ‖x‖²)^{(m−2)/2}` on the
unit ball, normalised. -/
noncomputable def blockLaw (U : Type*) [NormedAddCommGroup U] [InnerProductSpace ℝ U]
    [MeasurableSpace U] [BorelSpace U] [FiniteDimensional ℝ U] (m : ℕ) : Measure U :=
  (blockMeasure U m Set.univ)⁻¹ • blockMeasure U m

variable {U V : Type*} [NormedAddCommGroup U] [InnerProductSpace ℝ U] [MeasurableSpace U]
  [BorelSpace U] [FiniteDimensional ℝ U] [Nontrivial U]
  [NormedAddCommGroup V] [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V]
  [FiniteDimensional ℝ V] [Nontrivial V]
