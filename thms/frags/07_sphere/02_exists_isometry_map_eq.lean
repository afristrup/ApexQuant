
/-- Transitivity: the isometry group acts transitively on the unit sphere. -/
theorem exists_isometry_map_eq {u v : E} (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) :
    ∃ L : E ≃ₗᵢ[ℝ] E, L u = v := by
  -- a unit vector is nonzero, so the space has positive rank and `Fin (finrank ℝ E)` is inhabited
  have hune : u ≠ 0 := by
    intro h; rw [h, norm_zero] at hu; norm_num at hu
  have hpos : 0 < Module.finrank ℝ E := Module.finrank_pos_iff_exists_ne_zero.2 ⟨u, hune⟩
  obtain ⟨bu, hbu⟩ := aux_exists_orthonormalBasis_apply_eq hu ⟨0, hpos⟩
  obtain ⟨bv, hbv⟩ := aux_exists_orthonormalBasis_apply_eq hv ⟨0, hpos⟩
  refine ⟨bu.equiv bv (Equiv.refl _), ?_⟩
  rw [← hbu, bu.equiv_apply_basis, Equiv.refl_apply, hbv]
