
/-- `unitVec S w p` is `u_p = R_p r_p / ‖r_p‖` of eq. (2.2).

At `r_p = 0` this evaluates to `0` by Lean's `(0 : ℝ)⁻¹ = 0` convention; every statement below
that mentions `u_p` is either guarded by `r_p ≠ 0` or is trivially true at `r_p = 0`. -/
noncomputable def unitVec (p : ℕ) : E := ‖resid S w p‖⁻¹ • S.rot p (resid S w p)
