theorem testFun_toLp (ψ : U → ℝ≥0∞) (x : U) (y : V) :
    testFun ψ (WithLp.toLp 2 (x, y)) = sliceFun ψ x ‖y‖ := by
  unfold testFun sliceFun
  rw [WithLp.prod_norm_eq_of_L2]
  simp only [WithLp.fst, WithLp.snd]
  congr 1
  have hmem : WithLp.toLp 2 (x, y) ∈ ball (0 : WithLp 2 (U × V)) 1 ↔ ‖x‖ ^ 2 + ‖y‖ ^ 2 < 1 := by
    rw [mem_ball_zero_iff, WithLp.prod_norm_eq_of_L2, Real.sqrt_lt' one_pos, one_pow]
    simp only [WithLp.fst, WithLp.snd]
  by_cases h : ‖x‖ ^ 2 + ‖y‖ ^ 2 < 1
  · rw [Set.indicator_of_mem (hmem.mpr h), Set.indicator_of_mem (Set.mem_Iio.mpr h)]
    rfl
  · rw [Set.indicator_of_notMem (fun h' => h (hmem.mp h')),
      Set.indicator_of_notMem (fun h' => h (Set.mem_Iio.mp h'))]
