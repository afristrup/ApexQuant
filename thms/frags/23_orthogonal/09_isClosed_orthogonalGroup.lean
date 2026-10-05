
theorem isClosed_orthogonalGroup :
    IsClosed ((Matrix.orthogonalGroup (Fin d) ℝ : Submonoid _) : Set (Matrix (Fin d) (Fin d) ℝ)) := by
  have : ((Matrix.orthogonalGroup (Fin d) ℝ : Submonoid _) : Set (Matrix (Fin d) (Fin d) ℝ))
      = {A | star A * A = 1} := by
    ext A; exact Matrix.mem_unitaryGroup_iff'
  rw [this]
  exact isClosed_eq (continuous_star.mul continuous_id) continuous_const
