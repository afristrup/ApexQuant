
theorem norm_modelVec (n : ℕ) : ‖modelVec n‖ = 1 := by
  rw [modelVec, WithLp.norm_toLp_fst, norm_one]
