
/-- **Right side, first two steps.**  Split `E = U × V`, then apply the polar formula on `V`. -/
theorem lintegral_testFun_eq_slices {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) :
    ∫⁻ z, testFun (V := V) ψ z
      = ∫⁻ x, (∫⁻ s in Set.Ioi (0 : ℝ), ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1)) * sliceFun ψ x s)
          * volume.toSphere (Set.univ : Set (sphere (0 : V) 1)) := by
  rw [← (WithLp.volume_preserving_toLp U V).lintegral_comp (measurable_testFun hψ),
    volume_eq_prod, lintegral_prod (fun a : U × V => testFun ψ (WithLp.toLp 2 a))
      ((measurable_testFun hψ).comp (WithLp.measurable_toLp 2 (U × V))).aemeasurable]
  refine lintegral_congr fun x => ?_
  simp_rw [testFun_toLp]
  have hm : Measurable fun y : V => sliceFun ψ x ‖y‖ :=
    (measurable_sliceFun_uncurry hψ).comp (measurable_const.prodMk measurable_norm)
  rw [lintegral_eq_polar _ hm]
  have hw : ∀ w : sphere (0 : V) 1,
      (∫⁻ s in Set.Ioi (0 : ℝ), ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1))
        * sliceFun ψ x ‖s • (w : V)‖)
      = ∫⁻ s in Set.Ioi (0 : ℝ), ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1)) * sliceFun ψ x s := by
    intro w
    refine setLIntegral_congr_fun measurableSet_Ioi fun s hs => ?_
    rw [norm_smul, mem_sphere_zero_iff_norm.mp w.2, mul_one, Real.norm_eq_abs, abs_of_pos hs]
  simp_rw [hw]
  rw [lintegral_const]

omit [Nontrivial U] [Nontrivial V] in
