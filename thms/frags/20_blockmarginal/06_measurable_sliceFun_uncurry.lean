theorem measurable_sliceFun_uncurry {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) :
    Measurable (fun p : U × ℝ => sliceFun ψ p.1 p.2) := by
  unfold sliceFun
  exact (hψ.comp (by fun_prop)).mul
    ((measurable_const.indicator measurableSet_Iio).comp (by fun_prop))

omit [Nontrivial U] [Nontrivial V] in
