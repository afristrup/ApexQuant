
@[simp] theorem recon_zero : recon S w 0 = 0 := by simp [recon]

/-- Auxiliary: the telescoping identity `w - ŵ_P = r_P`, from which `sub_recon` (§2) follows. -/
private theorem aux_sub_recon (P : ℕ) : w - recon S w P = resid S w P := by
  induction P with
  | zero => simp
  | succ P ih =>
      have hr : recon S w (P + 1)
          = recon S w P + ‖resid S w P‖ • (S.rot P).symm (S.q (unitVec S w P)) := by
        simp [recon, Finset.sum_range_succ]
      rw [hr, ← sub_sub, ih, ← resid_succ]
