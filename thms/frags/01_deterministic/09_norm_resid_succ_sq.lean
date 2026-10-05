
/-- The squared form, which is what Proposition 2 iterates. -/
theorem norm_resid_succ_sq (p : ℕ) :
    ‖resid S w (p + 1)‖ ^ 2
      = ‖resid S w p‖ ^ 2 * ‖unitVec S w p - S.q (unitVec S w p)‖ ^ 2 := by
  rw [norm_resid_succ, mul_pow]
