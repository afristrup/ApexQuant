
/-- `resid S w p` is `r_p` of eq. (2.2). -/
noncomputable def resid : ℕ → E
  | 0 => w
  | p + 1 =>
      resid p - ‖resid p‖ • (S.rot p).symm (S.q (‖resid p‖⁻¹ • S.rot p (resid p)))
