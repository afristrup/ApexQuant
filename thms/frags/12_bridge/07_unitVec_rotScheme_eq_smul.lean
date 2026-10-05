
/-- **The unit direction is the fresh rotation applied to a function of the past.**  This is the
exact shape `indepFun_smul_of_indepPair` (§5) consumes. -/
theorem unitVec_rotScheme_eq_smul (hι : ∀ (g : G) (x : E), ι g x = g • x) (q : E → E) (w : E)
    (g : ℕ → G) (p : ℕ) :
    unitVec (rotScheme ι q g) w p = g p • residDirPast q p (w, fun i : Fin p => g i) := by
  rw [unitVec, residDirPast, smul_comm_real hι, resid_rotScheme_eq_residPast hι,
    show (rotScheme ι q g).rot p = ι (g p) from rfl, hι]
