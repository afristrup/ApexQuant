
/-- **Left side.**  The polar formula on `E` evaluates the test integral to the sphere
marginal times the radial constant. -/
theorem lintegral_testFun_eq_sphere {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) :
    ∫⁻ z, testFun (V := V) ψ z
      = (∫⁻ u : sphere (0 : WithLp 2 (U × V)) 1, ψ (WithLp.ofLp (u : WithLp 2 (U × V))).1
          ∂volume.toSphere) * radialConst (Module.finrank ℝ (WithLp 2 (U × V)) - 1) := by
  rw [lintegral_eq_polar _ (measurable_testFun hψ), ← lintegral_mul_const _ (by fun_prop)]
  refine lintegral_congr fun u => ?_
  have hm : Measurable (fun r : ℝ => ENNReal.ofReal
      (r ^ (Module.finrank ℝ (WithLp 2 (U × V)) - 1)) * (Set.Iio (1 : ℝ)).indicator 1 r) :=
    (by fun_prop : Measurable fun r : ℝ => ENNReal.ofReal
      (r ^ (Module.finrank ℝ (WithLp 2 (U × V)) - 1))).mul
      (measurable_const.indicator measurableSet_Iio)
  rw [radialConst, ← lintegral_const_mul _ hm]
  refine setLIntegral_congr_fun measurableSet_Ioi fun r hr => ?_
  rw [testFun_smul_sphere ψ u hr]
  ring
