
/-- **`IsotropicPasses.integrable_err`, derived** from finiteness of the stage distortion. -/
theorem integrable_stageErr (h : RotationStream ι q W R κ x₀ μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀)) (p : ℕ) :
    Integrable (fun ω => ‖unitVec (rotScheme ι q (R ω)) (W ω) p
        - q (unitVec (rotScheme ι q (R ω)) (W ω) p)‖ ^ 2) μ := by
  rw [← law_unitVec_eq_orbitLaw h p] at hint
  have hasm : AEStronglyMeasurable (fun u : E => ‖u - q u‖ ^ 2)
      (μ.map (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p)) :=
    ((measurable_id.sub h.meas_q).norm.pow_const 2).aestronglyMeasurable
  exact (integrable_map_measure hasm (aux_meas_unitVec h p).aemeasurable).mp hint
