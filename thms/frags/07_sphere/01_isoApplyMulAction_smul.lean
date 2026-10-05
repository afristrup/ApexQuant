
@[simp] theorem isoApplyMulAction_smul (L : E ≃ₗᵢ[ℝ] E) (x : E) : L • x = L x := rfl

/-- A unit vector occurs as a prescribed member of some orthonormal basis indexed by
`Fin (finrank ℝ E)`. -/
private theorem aux_exists_orthonormalBasis_apply_eq {u : E} (hu : ‖u‖ = 1)
    (i₀ : Fin (Module.finrank ℝ E)) :
    ∃ b : OrthonormalBasis (Fin (Module.finrank ℝ E)) ℝ E, b i₀ = u := by
  have ho : Orthonormal ℝ (Set.restrict ({i₀} : Set (Fin (Module.finrank ℝ E))) (fun _ => u)) := by
    rw [orthonormal_subsingleton_iff]
    intro i; simpa using hu
  obtain ⟨b, hb⟩ := ho.exists_orthonormalBasis_extension_of_card_eq (by simp)
  exact ⟨b, hb i₀ rfl⟩
