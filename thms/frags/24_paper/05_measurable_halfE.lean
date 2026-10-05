
theorem measurable_halfE [MeasurableSpace E] [BorelSpace E] : Measurable (halfE : E → E) :=
  (continuous_const_smul _).measurable
