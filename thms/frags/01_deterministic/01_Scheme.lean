
/-- The data of a progressive (multistage) quantiser: a stream of rotations `R_p`, and a
**single fixed** quantiser `q` reused at every pass.  That `q` does not depend on `p` is the
paper's central design claim; it is expressed here by the type. -/
structure Scheme (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E] where
  /-- `rot p` is the rotation `R_p` applied at pass `p`. -/
  rot : ℕ → (E ≃ₗᵢ[ℝ] E)
  /-- the codebook rounding map; in the paper, nearest-neighbour rounding applied
  coordinate-wise to a `2^b`-level Lloyd–Max codebook for `f_d`. -/
  q : E → E

variable (S : Scheme E) (w : E)
