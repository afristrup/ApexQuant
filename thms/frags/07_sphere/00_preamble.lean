/-! ### §5b  Instantiating at the sphere

The orbit of a unit vector under the linear isometry group is the whole unit sphere: this is the
transitivity the paper invokes when it writes "complete `â` to an orthonormal basis". -/

section Sphere

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- The tautological action of the isometry group on the space.  (Mathlib has
`LinearEquiv.applyDistribMulAction` but no isometric analogue, so we supply it.) -/
scoped instance isoApplyMulAction : MulAction (E ≃ₗᵢ[ℝ] E) E where
  smul L x := L x
  one_smul _ := rfl
  mul_smul _ _ _ := rfl
