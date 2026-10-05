
instance streamSpaceMeasurableSpace (E G : Type u) [MeasurableSpace E] [MeasurableSpace G] :
    ∀ i : Option ℕ, MeasurableSpace (streamSpace E G i)
  | none => inferInstanceAs (MeasurableSpace E)
  | some _ => inferInstanceAs (MeasurableSpace G)
