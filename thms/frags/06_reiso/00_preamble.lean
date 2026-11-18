/-! ## §5  Theorem 1 (re-isotropisation)

The paper: *"Let `r ∈ ℝ^d \ {0}` be a random vector and `R` Haar on `O(d)`, independent of `r`.
Then `u := Rr/‖Rr‖` is uniform on `S^{d-1}` and independent of `‖r‖`, regardless of the
distribution of `r`."*

The proof given in the paper is a right-invariance argument, and right-invariance is genuinely
all it needs.  Formalising it abstractly makes that visible: the statement is about a compact
group acting transitively on a set, and has nothing to do with spheres, rotations, or `ℝ^d`.

`orbitLaw κ x₀` is the law of `g • x₀` for `g ∼ κ`.

**Read the scope of this section carefully.**  `orbitLaw κ x₀` is *defined* as a pushforward.
Nothing below identifies it with the normalised surface measure on `S^{d-1}`, so the paper's
phrase "`u` is uniform on the sphere" is **not** established here: that identification needs
(a) Haar measure on the orthogonal group, (b) uniqueness of the rotation-invariant probability
measure on the sphere, and (c) a Mathlib handle on the uniform measure on `Metric.sphere`.
Mathlib currently supplies none of the three, there is no `MeasurableSpace` instance on
`E ≃ₗᵢ[ℝ] E` and no Haar measure on the orthogonal group, so the abstract results below cannot
presently even be *instantiated* at the paper's setting.  What is proved is the
distributional-invariance half of Theorem 1: the law of `R • x` is the same for every `x` in the
orbit, whatever the law of `x`.  That is the paper's right-invariance argument, and it is the
part Proposition 2 consumes; the "uniform" wording is stronger than the proof. -/

section Reisotropisation

variable {G X : Type*} [Group G] [MeasurableSpace G] [MeasurableSpace X]
variable [MulAction G X] [MeasurableSMul G X]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
