/-! ## §21  Lemma 1, closed: the uniform direction law and its marginals

`sphereUniform E` is the normalised surface measure, the paper's "uniform on `S^{d−1}`".  Its
block marginal is `blockLaw U (dim V)` (`sphereUniform_map_fst`), which at `U = ℝ` is exactly
the paper's `sphericalCoordLaw d` of §4b (`blockLaw_real`); the constant in `f_d` is forced by
both sides being probability measures, so §4's Beta integral is what identifies it.  The sphere
measure is natural under linear isometries (`toSphere_map_sphereMap`), which transports the
statement to any unit vector `e` of any finite-dimensional inner product space of dimension at
least two (`sphereUniform_map_inner`), and to any orthogonal decomposition `E ≃ U × V`
(`sphereUniform_map_block`).  `sphereUniformAmbient E` is the same law as a measure on `E`, the
`σ` of §9 and §17; `sphereUniformAmbient_map_block` and `sphereUniformAmbient_map_inner` are the
`hmarg` hypotheses of §17 and §3b, discharged.

What this does not touch is header item 2: that the *rotated* direction `R r/‖R r‖` has this law
still needs Haar measure on `O(d)`. -/

section Normalise

variable (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E]
