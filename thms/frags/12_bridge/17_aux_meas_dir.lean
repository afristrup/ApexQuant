
private theorem aux_meas_dir (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    Measurable (fun ω => residDirPast q p (W ω, fun i : Fin p => R ω i)) :=
  (measurable_residDirPast h.meas_q p).comp (aux_meas_past h p)
