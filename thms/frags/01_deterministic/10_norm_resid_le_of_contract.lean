
/-- The residual norms are non-increasing along passes **iff** each pass does not increase the
normalised error; the paper's `ρ < 1` regime.  Stated as a hypothesis-carrying lemma rather
than assumed. -/
theorem norm_resid_le_of_contract {c : ℝ} (hc : 0 ≤ c)
    (h : ∀ p, ‖unitVec S w p - S.q (unitVec S w p)‖ ≤ c) (P : ℕ) :
    ‖resid S w P‖ ≤ c ^ P * ‖w‖ := by
  induction P with
  | zero => simp
  | succ P ih =>
      rw [norm_resid_succ]
      calc ‖resid S w P‖ * ‖unitVec S w P - S.q (unitVec S w P)‖
          ≤ (c ^ P * ‖w‖) * c :=
            mul_le_mul ih (h P) (norm_nonneg _) (mul_nonneg (pow_nonneg hc P) (norm_nonneg _))
        _ = c ^ (P + 1) * ‖w‖ := by ring
