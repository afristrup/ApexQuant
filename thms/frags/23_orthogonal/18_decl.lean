
instance : ContinuousSMul (Od d) (Ed d) := by
  constructor
  simp only [smul_def, Matrix.toEuclideanLin_apply]
  fun_prop
