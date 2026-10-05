
/-- The rotation `g` as a linear isometry equivalence of `ℝ^d`. -/
noncomputable def iso (g : Od d) : Ed d ≃ₗᵢ[ℝ] Ed d :=
  LinearIsometry.toLinearIsometryEquiv
    ⟨Matrix.toEuclideanLin (g : Matrix (Fin d) (Fin d) ℝ), norm_smul' d g⟩ rfl
