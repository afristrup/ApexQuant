
/-- **`IsotropicPasses.distortion`, derived.**  The expected squared normalised quantisation
error is the *same number* at every pass, namely the stage distortion of `q` against the
direction law.  The paper asserts this; here it is a consequence of `law_unitVec_eq_orbitLaw`. -/
theorem integral_stageErr_eq_stageDistortion (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    ∫ ω, ‖unitVec (rotScheme ι q (R ω)) (W ω) p
        - q (unitVec (rotScheme ι q (R ω)) (W ω) p)‖ ^ 2 ∂μ
      = stageDistortion q (orbitLaw κ x₀) := by
  have hasm : AEStronglyMeasurable (fun u : E => ‖u - q u‖ ^ 2)
      (μ.map (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p)) :=
    ((measurable_id.sub h.meas_q).norm.pow_const 2).aestronglyMeasurable
  rw [stageDistortion, ← law_unitVec_eq_orbitLaw h p,
    integral_map (aux_meas_unitVec h p).aemeasurable hasm]
