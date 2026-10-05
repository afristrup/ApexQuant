private theorem aux_isProb (h : RotationStream ι q W R κ x₀ μ) [IsProbabilityMeasure μ] :
    IsProbabilityMeasure κ := by
  rw [← h.law_R 0]
  exact Measure.isProbabilityMeasure_map (h.meas_R 0).aemeasurable
