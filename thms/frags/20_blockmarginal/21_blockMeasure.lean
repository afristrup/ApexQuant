
/-- The unnormalised density measure of the `U`-block: `(1 − ‖x‖²)^{(m−2)/2}` on the unit ball. -/
noncomputable def blockMeasure (U : Type*) [NormedAddCommGroup U] [InnerProductSpace ℝ U]
    [MeasurableSpace U] [BorelSpace U] [FiniteDimensional ℝ U] (m : ℕ) : Measure U :=
  (volume.restrict (ball (0 : U) 1)).withDensity (blockDensity m)
