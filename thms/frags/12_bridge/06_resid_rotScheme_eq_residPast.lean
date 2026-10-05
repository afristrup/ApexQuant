
/-- **The residual factors through the past.**  `r_p` is the value at `(w, g_0,…,g_{p-1})` of a
fixed map: it does not see `g_p, g_{p+1}, …`.  This is the inductive fact the paper argues in
prose after eq. (2.2), and it is what makes the pass-`p` rotation *fresh*. -/
theorem resid_rotScheme_eq_residPast (hι : ∀ (g : G) (x : E), ι g x = g • x) (q : E → E) (w : E)
    (g : ℕ → G) (p : ℕ) :
    resid (rotScheme ι q g) w p = residPast q p (w, fun i : Fin p => g i) := by
  induction p with
  | zero => rfl
  | succ p ih =>
      have hcast : (fun i : Fin p => (fun j : Fin (p + 1) => g j) i.castSucc)
          = fun i : Fin p => g i := by
        funext i; simp
      rw [resid_succ, unitVec, ih, residPast]
      simp only [hcast, Fin.val_last]
      rw [show (rotScheme ι q g).rot p = ι (g p) from rfl, show (rotScheme ι q g).q = q from rfl,
        symm_apply_eq_inv_smul hι, hι]
