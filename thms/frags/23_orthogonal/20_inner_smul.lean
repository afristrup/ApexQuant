
theorem inner_smul (g : Od d) (x y : Ed d) : ⟪g • x, g • y⟫_ℝ = ⟪x, y⟫_ℝ := by
  rw [smul_def, smul_def, ← LinearMap.adjoint_inner_right,
    ← Matrix.toEuclideanLin_conjTranspose_eq_adjoint, Matrix.toEuclideanLin_apply,
    Matrix.toEuclideanLin_apply, Matrix.mulVec_mulVec, ← Matrix.star_eq_conjTranspose,
    Matrix.mem_unitaryGroup_iff'.mp g.2, Matrix.one_mulVec, WithLp.toLp_ofLp]
