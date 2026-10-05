
instance : IsProbabilityMeasure (haarOd d) :=
  ⟨by rw [haarOd, ← TopologicalSpace.PositiveCompacts.coe_top]; exact haarMeasure_self⟩
