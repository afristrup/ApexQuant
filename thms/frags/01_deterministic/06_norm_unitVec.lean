
/-- `‖u_p‖ = 1` whenever the pass is not already exact. -/
theorem norm_unitVec (p : ℕ) (h : resid S w p ≠ 0) : ‖unitVec S w p‖ = 1 := by
  rw [unitVec, norm_smul, LinearIsometryEquiv.norm_map, norm_inv, norm_norm,
    inv_mul_cancel₀ (norm_ne_zero_iff.mpr h)]
