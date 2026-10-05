
/-- **The per-pass identity, exactly.**  The residual of pass `p+1` is the pass-`p` scale times
the pulled-back quantisation error of the unit direction.  This is the step the paper writes as
"because `R_p` is orthogonal, `‖r_{p+1}‖² = s_p² ‖u_p - q(u_p)‖²`", and it holds *identically* —
no probability, no hypothesis on `q`, and including the degenerate case `r_p = 0`. -/
theorem resid_succ_eq_smul (p : ℕ) :
    resid S w (p + 1)
      = ‖resid S w p‖ • (S.rot p).symm (unitVec S w p - S.q (unitVec S w p)) := by
  rw [resid_succ, map_sub, smul_sub]
  congr 1
  rw [unitVec, map_smul, LinearIsometryEquiv.symm_apply_apply, smul_smul]
  rcases eq_or_ne (resid S w p) 0 with h | h
  · simp [h]
  · rw [mul_inv_cancel₀ (norm_ne_zero_iff.mpr h), one_smul]
