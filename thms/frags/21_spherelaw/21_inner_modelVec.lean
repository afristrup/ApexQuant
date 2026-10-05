
theorem inner_modelVec (n : ℕ) (x : modelSpace n) : ⟪modelVec n, x⟫_ℝ = (WithLp.ofLp x).1 := by
  rw [modelVec, WithLp.prod_inner_apply]
  simp [RCLike.inner_apply]

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E]
