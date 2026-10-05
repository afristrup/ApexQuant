
theorem stageDistortion_nonneg (Q : E → E) (σ : Measure E) : 0 ≤ stageDistortion Q σ :=
  integral_nonneg fun _ => by positivity
