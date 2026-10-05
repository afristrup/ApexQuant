/-! ## §23  `O(d)` with its Haar measure: the paper's setting, instantiated

§22 identifies the direction law for any group and any right-invariant probability measure with
the right properties.  This section exhibits the paper's own: `O(d)` as Mathlib's
`Matrix.orthogonalGroup`, made a Borel space, shown compact (a closed subset of a box, since the
entries of an orthogonal matrix lie in `[-1, 1]`), given its normalised Haar measure `haarOd d`,
shown right-invariant by `isMulRightInvariant_of_compactSpace`, a general fact about compact
groups proved from uniqueness of Haar measure.  Mathlib's existing action of `O(d)` on `ℝ^d` is
matrix-vector multiplication; it is continuous, hence jointly measurable, by isometries
(`inner_smul`), and transitive on the unit sphere (`transitive`, from §5b and the fact that the
matrix of an isometry in an orthonormal basis is orthogonal).  `orbitLaw_haar` is then
Proposition 1 in full: the law of `g • x₀` for `g` Haar on `O(d)` is the uniform law. -/

/-! ### Right-invariance of Haar measure on a compact group -/

section Unimodular
variable {G : Type*} [Group G] [TopologicalSpace G] [IsTopologicalGroup G] [CompactSpace G]
  [MeasurableSpace G] [BorelSpace G] [SecondCountableTopology G]
