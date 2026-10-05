
theorem measurable_angFun (a b : ℕ) {g : ℝ → ℝ≥0∞} (hg : Measurable g) :
    Measurable (angFun a b g) :=
  ((measurable_angWeight a b).mul (hg.comp Real.measurable_cos)).indicator measurableSet_Ioo
