
theorem norm_resid_halfE (rot : ℕ → (E ≃ₗᵢ[ℝ] E)) {w : E} (hw : w ≠ 0) (p : ℕ) :
    ‖resid ⟨rot, halfE⟩ w p‖ = ‖w‖ / 2 ^ p := by
  induction p with
  | zero => simp
  | succ p ih =>
      have hpos : (0 : ℝ) < ‖w‖ / 2 ^ p := by
        have := norm_pos_iff.mpr hw
        positivity
      have hne : resid ⟨rot, halfE⟩ w p ≠ 0 := by
        rw [← norm_ne_zero_iff, ih]
        exact ne_of_gt hpos
      rw [norm_resid_succ]
      have hq : (⟨rot, halfE⟩ : Scheme E).q = halfE := rfl
      rw [hq, norm_sub_halfE, norm_unitVec _ _ _ hne, ih]
      ring
