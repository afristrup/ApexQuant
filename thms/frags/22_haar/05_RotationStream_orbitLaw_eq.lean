
/-- For a rotation stream with a unit base point, the direction law of §11 *is* the uniform
law on the sphere: header item 2, discharged from the stream's own hypotheses. -/
theorem RotationStream.orbitLaw_eq (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1) :
    orbitLaw κ x₀ = sphereUniformAmbient E := by
  haveI : IsProbabilityMeasure κ := by
    rw [← h.law_R 0]; exact Measure.isProbabilityMeasure_map (h.meas_R 0).aemeasurable
  haveI : Nontrivial E := ⟨⟨x₀, 0, by rintro rfl; simp at hx₀⟩⟩
  exact orbitLaw_eq_sphereUniformAmbient h.implements κ h.transitive
