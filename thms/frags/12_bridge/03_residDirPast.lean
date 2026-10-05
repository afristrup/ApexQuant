
/-- The normalised residual `r̂_p = r_p/‖r_p‖`, as a function of the past. -/
noncomputable def residDirPast (q : E → E) (p : ℕ) (x : E × (Fin p → G)) : E :=
  ‖residPast q p x‖⁻¹ • residPast q p x

variable {ι : G → (E ≃ₗᵢ[ℝ] E)}
