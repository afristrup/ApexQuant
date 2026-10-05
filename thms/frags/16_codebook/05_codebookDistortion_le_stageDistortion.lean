
/-- Any quantiser with values in `C'` errs at least the distance to `C'`, in expectation. -/
theorem codebookDistortion_le_stageDistortion (hQ' : ∀ x, Q' x ∈ C')
    (hint : Integrable (fun u => ‖u - Q' u‖ ^ 2) σ) :
    codebookDistortion C' σ ≤ stageDistortion Q' σ :=
  integral_mono_of_nonneg (Filter.Eventually.of_forall fun _ => by positivity) hint
    (Filter.Eventually.of_forall fun x => by
      show Metric.infDist x C' ^ 2 ≤ ‖x - Q' x‖ ^ 2
      rw [← dist_eq_norm]
      exact pow_le_pow_left₀ Metric.infDist_nonneg (Metric.infDist_le_dist_of_mem (hQ' x)) 2)
