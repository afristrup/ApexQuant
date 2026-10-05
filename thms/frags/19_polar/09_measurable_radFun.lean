
theorem measurable_radFun (n : ℕ) : Measurable (radFun n) :=
  (by fun_prop : Measurable fun R : ℝ => ENNReal.ofReal (R ^ n)).mul
    (measurable_const.indicator measurableSet_Iio)
