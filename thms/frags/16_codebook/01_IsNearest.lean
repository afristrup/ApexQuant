
/-- `Q` implements the codebook `C`: it takes values in `C` and always picks a nearest point. -/
def IsNearest (C : Set E) (Q : E → E) : Prop :=
  ∀ x, Q x ∈ C ∧ ∀ c ∈ C, ‖x - Q x‖ ≤ ‖x - c‖
