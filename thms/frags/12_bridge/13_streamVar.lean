
/-- The joint randomness of the scheme, as one family of random variables. -/
def streamVar (W : Ω → E) (R : Ω → ℕ → G) : ∀ i : Option ℕ, Ω → streamSpace E G i
  | none => W
  | some p => fun ω => R ω p
