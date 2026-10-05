
theorem measurable_residPast {q : E → E} (hq : Measurable q) (p : ℕ) :
    Measurable (residPast (G := G) q p) := by
  induction p with
  | zero => exact measurable_fst
  | succ p ih =>
      have h1 : Measurable (fun x : E × (Fin (p + 1) → G) =>
          (x.1, fun i : Fin p => x.2 i.castSucc)) :=
        measurable_fst.prodMk (measurable_pi_lambda _ fun i =>
          (measurable_pi_apply (i.castSucc)).comp measurable_snd)
      have hr : Measurable (fun x : E × (Fin (p + 1) → G) =>
          residPast (G := G) q p (x.1, fun i : Fin p => x.2 i.castSucc)) := ih.comp h1
      have hg : Measurable (fun x : E × (Fin (p + 1) → G) => x.2 (Fin.last p)) :=
        (measurable_pi_apply _).comp measurable_snd
      exact hr.sub ((hr.norm).smul
        ((hg.inv).smul (hq.comp (((hr.norm).inv).smul (hg.smul hr)))))
