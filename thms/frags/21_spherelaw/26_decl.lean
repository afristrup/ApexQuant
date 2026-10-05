
instance [Nontrivial E] : IsProbabilityMeasure (sphereUniformAmbient E) :=
  Measure.isProbabilityMeasure_map measurable_subtype_coe.aemeasurable
