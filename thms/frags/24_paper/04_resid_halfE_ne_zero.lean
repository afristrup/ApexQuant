
theorem resid_halfE_ne_zero (rot : ℕ → (E ≃ₗᵢ[ℝ] E)) {w : E} (hw : w ≠ 0) (p : ℕ) :
    resid ⟨rot, halfE⟩ w p ≠ 0 := by
  rw [← norm_ne_zero_iff, norm_resid_halfE rot hw p]
  have := norm_pos_iff.mpr hw
  positivity
