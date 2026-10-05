
/-- The state space of the joint randomness: the weight row at `none`, a rotation at `some p`. -/
def streamSpace (E G : Type u) : Option ℕ → Type u
  | none => E
  | some _ => G
