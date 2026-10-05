
/-- Consequently the orbit law does not depend on the base point chosen within an orbit. -/
theorem orbitLaw_eq_of_mem_orbit (κ : Measure G) [κ.IsMulRightInvariant] [MeasurableMul G]
    {x₀ x : X} (hx : x ∈ MulAction.orbit G x₀) : orbitLaw κ x = orbitLaw κ x₀ := by
  obtain ⟨g, rfl⟩ := hx
  exact orbitLaw_smul κ g x₀
