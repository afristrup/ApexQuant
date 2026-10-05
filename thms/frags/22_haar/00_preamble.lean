/-! ## §22  The Haar identification: the direction law *is* the uniform law

Header item 2 was the last gap between §11 and the paper: `orbitLaw κ x₀`, the law of the
rotated direction, was not identified with the uniform law on the sphere.  The identification
needs no construction of Haar measure at all.  If `κ` is a right-invariant probability measure
on a group acting on `E` by isometries and transitively on the unit sphere, the hypotheses
`RotationStream` already carries, then `orbitLaw κ x₀ = sphereUniformAmbient E`
(`orbitLaw_eq_sphereUniformAmbient`).  The proof is the uniqueness-of-invariant-measure step of
Proposition 1 done by hand: the uniform law is `G`-invariant (§21), so integrating a test
function against it and averaging over `κ` changes nothing; swapping the integrals and using
that the orbit law is constant along the orbit (§5) returns the orbit law.

Consequences: `u_p` is uniform on the sphere at every pass (`law_unitVec_eq_sphereUniform`),
`ρ²` is the stage distortion against the uniform law (`integral_norm_resid_sq_uniform`), and
every coordinate of the uniform direction on `ℝ^d` has the law `sphericalCoordLaw d`
(`sphereUniformAmbient_map_coord`), which is the `hmarg` hypothesis of §3b. -/

open scoped InnerProductSpace

section OrbitLawSphere

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableMul G]
