
/-! ### The action on `ℝ^d` -/

/-- The action of `O(d)` on `ℝ^d` Mathlib already provides is matrix-vector multiplication. -/
theorem smul_def (g : Od d) (x : Ed d) :
    g • x = Matrix.toEuclideanLin (g : Matrix (Fin d) (Fin d) ℝ) x := by
  rw [Matrix.toEuclideanLin_apply]; rfl
