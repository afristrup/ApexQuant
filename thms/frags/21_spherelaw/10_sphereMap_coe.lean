
@[simp] theorem sphereMap_coe (L : E ≃ₗᵢ[ℝ] F) (u : sphere (0 : E) 1) :
    ((sphereMap L u : sphere (0 : F) 1) : F) = L u := rfl
