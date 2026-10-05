
/-! ### Transitivity on the unit sphere -/

theorem transitive {x₀ : Ed d} (h₀ : ‖x₀‖ = 1) (x : Ed d) (hx : ‖x‖ = 1) :
    x ∈ MulAction.orbit (Od d) x₀ := by
  obtain ⟨L, hL⟩ := exists_isometry_map_eq h₀ hx
  let b := EuclideanSpace.basisFun (Fin d) ℝ
  refine ⟨⟨LinearMap.toMatrix b.toBasis b.toBasis L.toLinearEquiv,
    L.toMatrix_mem_unitaryGroup b b⟩, ?_⟩
  show Matrix.toEuclideanLin (LinearMap.toMatrix b.toBasis b.toBasis L.toLinearEquiv) x₀ = x
  rw [Matrix.toEuclideanLin_eq_toLin, ← EuclideanSpace.basisFun_toBasis, Matrix.toLin_toMatrix]
  exact hL
