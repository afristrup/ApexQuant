theorem measurable_sliceIntegral {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) :
    Measurable fun x : U =>
      ∫⁻ s in Set.Ioi (0 : ℝ), ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1)) * sliceFun ψ x s :=
  Measurable.lintegral_prod_right
    ((by fun_prop : Measurable fun p : U × ℝ =>
      ENNReal.ofReal (p.2 ^ (Module.finrank ℝ V - 1))).mul (measurable_sliceFun_uncurry hψ))
