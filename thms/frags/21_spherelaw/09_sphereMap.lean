
/-- A linear isometry equivalence restricted to the unit spheres. -/
def sphereMap (L : E ≃ₗᵢ[ℝ] F) (u : sphere (0 : E) 1) : sphere (0 : F) 1 :=
  ⟨L u, by rw [mem_sphere_zero_iff_norm, L.norm_map]; exact mem_sphere_zero_iff_norm.mp u.2⟩
