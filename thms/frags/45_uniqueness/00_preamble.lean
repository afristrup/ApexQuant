/-!
# §45  Uniqueness of the rotation-invariant law on the sphere

Header items 1 and 2 said that the uniform law on the sphere is rotation-invariant (§21) and
that the orbit law of a Haar rotation is the uniform law (§22, §23), but that the uniform law
being the *only* rotation-invariant probability measure on the sphere was not proved.  It is
the same argument as §22 run in the other direction: if `μ` is a `G`-invariant probability
measure carried by the unit sphere, then `μ(s) = ∫ μ((g•·)⁻¹ s) dκ(g)` for a right-invariant
probability `κ` on `G`; swapping the integrals turns the right side into
`∫ (orbitLaw κ x)(s) dμ(x)`, and the orbit law of every point of the sphere is the uniform law.

* `eq_sphereUniformAmbient_of_invariant`: the abstract statement, for a group acting by
  isometries and transitively on the unit sphere with a right-invariant probability measure.
* `Orthogonal.eq_sphereUniformAmbient_of_invariant`: at `O(d)` with its Haar measure, every
  `O(d)`-invariant probability measure on `ℝ^d` carried by the unit sphere is the uniform law.
-/

open scoped InnerProductSpace

section Uniqueness

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableMul G]
