
/-- **`IsotropicPasses.integrable`, derived** by induction along the passes, using
`IndepFun.integrable_mul` at each step. -/
theorem integrable_normSq_resid (h : RotationStream ι q W R κ x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀)) (p : ℕ) :
    Integrable (fun ω => ‖resid (rotScheme ι q (R ω)) (W ω) p‖ ^ 2) μ := by
  induction p with
  | zero => simpa using hW
  | succ p ih =>
      have hmul := (indepFun_scale_stageErr h p).integrable_mul ih (integrable_stageErr h hint p)
      have heq : (fun ω => ‖resid (rotScheme ι q (R ω)) (W ω) (p + 1)‖ ^ 2)
          = (fun ω => ‖resid (rotScheme ι q (R ω)) (W ω) p‖ ^ 2) *
            (fun ω => ‖unitVec (rotScheme ι q (R ω)) (W ω) p
              - q (unitVec (rotScheme ι q (R ω)) (W ω) p)‖ ^ 2) := by
        funext ω; exact norm_resid_succ_sq _ _ p
      rw [heq]; exact hmul
