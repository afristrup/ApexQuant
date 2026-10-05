
/-- The cone over `s` pulls back along `L` to the cone over the pullback of `s`. -/
theorem preimage_cone (L : E ≃ₗᵢ[ℝ] F) (s : Set (sphere (0 : F) 1)) :
    ⇑L ⁻¹' (Set.Ioo (0 : ℝ) 1 • (Subtype.val '' s))
      = Set.Ioo (0 : ℝ) 1 • (Subtype.val '' (sphereMap L ⁻¹' s)) := by
  ext x
  simp only [Set.mem_preimage, Set.mem_smul, Set.mem_image]
  constructor
  · rintro ⟨r, hr, y, ⟨v, hv, rfl⟩, hx⟩
    refine ⟨r, hr, L.symm v, ⟨sphereMap L.symm v, ?_, rfl⟩, ?_⟩
    · show sphereMap L (sphereMap L.symm v) ∈ s
      convert hv using 1
      ext
      simp
    · apply L.injective
      rw [← hx, map_smul, L.apply_symm_apply]
  · rintro ⟨r, hr, y, ⟨w, hw, rfl⟩, rfl⟩
    exact ⟨r, hr, L w, ⟨sphereMap L w, hw, rfl⟩, by rw [map_smul]⟩
