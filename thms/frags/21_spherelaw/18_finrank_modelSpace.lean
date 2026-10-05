
theorem finrank_modelSpace (n : ℕ) : Module.finrank ℝ (modelSpace n) = n + 1 := by
  rw [finrank_withLp_prod, Module.finrank_self, finrank_euclideanSpace_fin]
  ring
