/-! ## §16  Corollary 1 for every codebook

Corollary 1 (§6) is stated for scalar codebooks applied coordinate-wise, and §9 records that its
optimality is optimality *within a class*.  This section removes the word *scalar*.  A *codebook*
is any set `C ⊆ E` of reproduction points, and a stage quantiser *implements* `C` if it always
returns a nearest codeword.  The scalar Lloyd–Max stage (`C` a product of `2^b`-point sets),
QuIP#'s E8P stage (`C` a finite subset of the `E₈` lattice, one copy per block of eight) and
QTIP's trellis stage (`C` the codeword set of the trellis code, which the Viterbi search rounds
to exactly) are all of this form; §17 checks the per-block case.

The objective is `codebookDistortion C σ = E_σ dist(u, C)²` against the direction law
`σ = orbitLaw κ x₀`, and §11's `law_unitVec_eq_orbitLaw` makes `σ` the same measure at every
pass.  So whichever codebook minimises it over a class is optimal at every pass, and no per-pass
re-fitting can improve on it in expectation (§18 states this at the residual level).  Nothing
here uses coordinates, block structure, or finiteness of `C`; finiteness enters only in
`exists_isNearest`, which supplies a nearest-codeword map for every finite codebook. -/

section Codebook

variable {E : Type*} [NormedAddCommGroup E] [MeasurableSpace E]
variable {C C' : Set E} {Q Q' : E → E} {σ : Measure E}
