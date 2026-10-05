
private theorem aux_meas_unitVec (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    Measurable (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p) := by
  simp only [unitVec_rotScheme_eq_smul h.implements]
  exact (h.meas_R p).smul (aux_meas_dir h p)

omit [BorelSpace E] [SecondCountableTopology E] [MeasurableSMul₂ G E] [MeasurableInv G] in
