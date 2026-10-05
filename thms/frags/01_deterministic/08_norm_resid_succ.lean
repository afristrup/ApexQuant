
/-- The norm form of `resid_succ_eq_smul`: `‖r_{p+1}‖ = ‖r_p‖ · ‖u_p - q(u_p)‖`. -/
theorem norm_resid_succ (p : ℕ) :
    ‖resid S w (p + 1)‖ = ‖resid S w p‖ * ‖unitVec S w p - S.q (unitVec S w p)‖ := by
  rw [resid_succ_eq_smul, norm_smul, LinearIsometryEquiv.norm_map, norm_norm]
