
/-- The fresh rotation is independent of the pair *(normalised residual, squared scale)*, the
exact hypothesis of `indepFun_smul_of_indepPair`.  This is `RotationStream.fresh` pushed through
the measurable map `resid_rotScheme_eq_residPast` supplies. -/
theorem indepFun_rot_dirScale (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    IndepFun (fun ω => R ω p)
      (fun ω => (residDirPast q p (W ω, fun i : Fin p => R ω i),
        ‖resid (rotScheme ι q (R ω)) (W ω) p‖ ^ 2)) μ := by
  have hΨ : Measurable (fun x : E × (Fin p → G) =>
      (residDirPast (G := G) q p x, ‖residPast (G := G) q p x‖ ^ 2)) :=
    (measurable_residDirPast h.meas_q p).prodMk
      ((measurable_residPast h.meas_q p).norm.pow_const 2)
  have hc := (h.fresh p).comp (measurable_id (α := G)) hΨ
  simpa [Function.comp_def, resid_rotScheme_eq_residPast h.implements] using hc
