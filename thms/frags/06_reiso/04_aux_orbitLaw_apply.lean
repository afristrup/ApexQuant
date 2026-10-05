
/-! ### Auxiliary machinery for the three probabilistic statements

Everything past the two lemmas just proved needs the action to be **jointly** measurable,
`MeasurableSMul₂ G X` (i.e. `(g, x) ↦ g • x` measurable on `G × X`), and not merely separately
measurable in each variable, which is all that `MeasurableSMul G X` provides.  The `aux_`
lemmas below record the complete proofs under that extra instance; see the `FIXME`s for why the
three `theorem`s *as stated* cannot be discharged from `MeasurableSMul` alone. -/

private theorem aux_orbitLaw_apply (κ : Measure G) (x : X) {s : Set X} (hs : MeasurableSet s) :
    orbitLaw κ x s = κ {g | g • x ∈ s} :=
  Measure.map_apply (measurable_smul_const x) hs
