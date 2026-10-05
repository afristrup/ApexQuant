
theorem measurable_testFun {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) : Measurable (testFun (V := V) ψ) := by
  unfold testFun
  exact (hψ.comp (by fun_prop)).mul (measurable_const.indicator measurableSet_ball)
