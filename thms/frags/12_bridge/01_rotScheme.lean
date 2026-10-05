
/-- The scheme whose pass-`p` rotation is `ι (g p)` and whose (single, fixed) quantiser is `q`. -/
def rotScheme (ι : G → (E ≃ₗᵢ[ℝ] E)) (q : E → E) (g : ℕ → G) : Scheme E where
  rot p := ι (g p)
  q := q
