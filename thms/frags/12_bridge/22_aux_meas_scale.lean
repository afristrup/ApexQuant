
private theorem aux_meas_scale (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    Measurable (fun ω => ‖resid (rotScheme ι q (R ω)) (W ω) p‖ ^ 2) := by
  simp only [resid_rotScheme_eq_residPast h.implements]
  exact ((measurable_residPast h.meas_q p).comp (aux_meas_past h p)).norm.pow_const 2

variable [MeasurableMul G] [IsProbabilityMeasure μ] [κ.IsMulRightInvariant]
