
theorem testFun_smul_sphere (ψ : U → ℝ≥0∞) (u : sphere (0 : WithLp 2 (U × V)) 1) {r : ℝ}
    (hr : 0 < r) :
    testFun ψ (r • (u : WithLp 2 (U × V))) = ψ (WithLp.ofLp (u : WithLp 2 (U × V))).1
      * (Set.Iio (1 : ℝ)).indicator 1 r := by
  have hu : ‖(u : WithLp 2 (U × V))‖ = 1 := mem_sphere_zero_iff_norm.mp u.2
  have hn : ‖r • (u : WithLp 2 (U × V))‖ = r := by
    rw [norm_smul, hu, mul_one, Real.norm_eq_abs, abs_of_pos hr]
  unfold testFun
  rw [hn]
  congr 1
  · rw [WithLp.ofLp_smul, Prod.smul_fst, inv_smul_smul₀ hr.ne']
  · by_cases h : r < 1
    · rw [Set.indicator_of_mem (by rwa [mem_ball_zero_iff, hn]),
        Set.indicator_of_mem (Set.mem_Iio.mpr h)]
      rfl
    · rw [Set.indicator_of_notMem (by rwa [mem_ball_zero_iff, hn]),
        Set.indicator_of_notMem (fun h' => h (Set.mem_Iio.mp h'))]
