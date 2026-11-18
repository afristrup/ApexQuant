
/-- `residPast q p` computes `r_p` from the pair `(w, (g_0,…,g_{p-1}))`, the *past* alone. -/
noncomputable def residPast (q : E → E) : ∀ p : ℕ, E × (Fin p → G) → E
  | 0, x => x.1
  | (p + 1), x =>
      let r := residPast q p (x.1, fun i => x.2 i.castSucc)
      r - ‖r‖ • ((x.2 (Fin.last p))⁻¹ • q (‖r‖⁻¹ • ((x.2 (Fin.last p)) • r)))
