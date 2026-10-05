
/-- Hence the unit sphere is a single orbit. -/
theorem orbit_eq_sphere {u : E} (hu : ‖u‖ = 1) :
    MulAction.orbit (E ≃ₗᵢ[ℝ] E) u = {v : E | ‖v‖ = 1} := by
  ext v
  constructor
  · rintro ⟨L, rfl⟩
    simpa using hu
  · intro hv
    obtain ⟨L, hL⟩ := exists_isometry_map_eq hu hv
    exact ⟨L, by simpa using hL⟩

end Sphere

