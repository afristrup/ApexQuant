
@[simp] theorem resid_zero : resid S w 0 = w := rfl

theorem resid_succ (p : ℕ) :
    resid S w (p + 1) = resid S w p - ‖resid S w p‖ • (S.rot p).symm (S.q (unitVec S w p)) :=
  rfl
