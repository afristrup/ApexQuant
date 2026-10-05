
/-- **Re-isotropisation, closed**: the unit direction `u_p` is uniform on the sphere at every
pass. -/
theorem law_unitVec_eq_sphereUniform (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1)
    (p : ℕ) :
    μ.map (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p) = sphereUniformAmbient E := by
  rw [law_unitVec_eq_orbitLaw h p, h.orbitLaw_eq hx₀]
