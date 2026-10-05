
private theorem aux_isProbabilityMeasure_orbitLaw (κ : Measure G) [IsProbabilityMeasure κ] (x : X) :
    IsProbabilityMeasure (orbitLaw κ x) :=
  Measure.isProbabilityMeasure_map (measurable_smul_const x).aemeasurable
