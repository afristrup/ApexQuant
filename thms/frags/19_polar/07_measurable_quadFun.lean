
theorem measurable_quadFun (a b : ℕ) {g : ℝ → ℝ≥0∞} (hg : Measurable g) :
    Measurable (quadFun a b g) := by
  unfold quadFun
  exact (by fun_prop : Measurable fun p : ℝ × ℝ => ENNReal.ofReal (p.1 ^ a)).mul
    ((by fun_prop : Measurable fun p : ℝ × ℝ => ENNReal.ofReal (p.2 ^ b)).mul
      ((hg.comp (by fun_prop)).mul
        ((measurable_const.indicator measurableSet_Iio).comp (by fun_prop))))
