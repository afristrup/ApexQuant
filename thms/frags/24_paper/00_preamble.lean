/-! ## §24  Proposition 2 as the paper states it, and a witness in every dimension

`integral_norm_resid_sq_paper` is Proposition 2 with nothing abstract left: Haar rotations on
`O(d)`, a scalar codebook applied coordinate-wise, and `ρ²` the quadrature `d·E_{f_d}[(t−q t)²]`
over the spherical marginal of Lemma 1.  Every hypothesis is about the inputs.

`exists_rotationStream_Od` shows the `RotationStream` bundle is satisfiable at `O(d)` for every
`d`: an i.i.d. Haar stream from `Measure.exists_iid`, the halving stage `x ↦ x/2`, and a unit
base point discharge every field.  `exists_witness_decay_Od` reads `err(P) = (1/2)^P` off the
general theorem, with `ρ² = 1/4` computed against the uniform law.  §12's `d = 1` witness is
now the special case it always was. -/

section Halving
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
