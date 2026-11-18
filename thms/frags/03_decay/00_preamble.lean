/-! ## §3  Proposition 2 (geometric decay)

The paper's Proposition 2 asserts `E‖r_{p+1}‖² = ρ² E‖r_p‖²`, hence `√(E‖r_P‖²)/‖w‖ = ρ^P`.

Its proof uses exactly two probabilistic inputs, isolated here as `IsotropicPasses`:

* `indep p`: the normalised direction's squared quantisation error is independent of the
  squared scale.  Theorem 1 (§5) supplies this, `u_p ⟂ r_p` gives `u_p ⟂ s_p`.
* `distortion p`: every pass sees the *same* expected normalised distortion `ρ²`.
  Corollary 1 (§6) plus Lemma 1 (§4) supply this, the coordinate marginals are `f_d` at
  every pass, so the expectation is the same number `ρ²` at every pass.

Note what is *absent*: the closed form of `f_d`, the optimality of the codebook, and any
concentration statement.  Proposition 2 is an identity in expectation and nothing more. -/

section Decay

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
