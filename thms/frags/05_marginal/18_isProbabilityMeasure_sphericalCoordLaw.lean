
/-- `f_d` really is a probability density. -/
theorem isProbabilityMeasure_sphericalCoordLaw (hd : 2 ≤ d) :
    IsProbabilityMeasure (sphericalCoordLaw d) := by
  constructor
  have hnn : 0 ≤ᵐ[volume.restrict (Set.Icc (-1:ℝ) 1)] sphericalPdf d := by
    rw [Filter.EventuallyLE, ae_restrict_iff' measurableSet_Icc]
    exact Filter.Eventually.of_forall fun t ht => sphericalPdf_nonneg ht hd
  rw [sphericalCoordLaw, withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ,
    ← ofReal_integral_eq_lintegral_ofReal (aux_integrable_pdf hd) hnn,
    aux_integral_pdf hd, ENNReal.ofReal_one]
