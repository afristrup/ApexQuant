
/-! The bridge from the interval integrals above to the measure `sphericalCoordLaw` is pure
bookkeeping: `Set.Icc (-1) 1` and `Set.Ioc (-1) 1` differ by a Lebesgue-null set, and
`∫ x in a..b` *is* the integral against `volume.restrict (Set.Ioc a b)` when `a ≤ b`. -/

private theorem aux_restrict_Icc_eq (a b : ℝ) :
    volume.restrict (Set.Icc a b) = volume.restrict (Set.Ioc a b) :=
  (Measure.restrict_congr_set Ioc_ae_eq_Icc).symm
