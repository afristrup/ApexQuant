
instance sphereUniform_isProbabilityMeasure [Nontrivial E] :
    IsProbabilityMeasure (sphereUniform E) := by
  constructor
  rw [sphereUniform, Measure.smul_apply, smul_eq_mul,
    ENNReal.inv_mul_cancel (toSphere_univ_ne_zero E) (measure_ne_top _ _)]

variable {E}
