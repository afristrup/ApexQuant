private theorem aux_meas_past (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    Measurable (fun ω => (W ω, fun i : Fin p => R ω i)) :=
  h.meas_W.prodMk (measurable_pi_lambda _ fun i => h.meas_R i)
