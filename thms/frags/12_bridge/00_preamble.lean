/-! ## §11  The bridge: a rotation stream *derives* `IsotropicPasses`

Header item 3 records that §3's Proposition 2 is conditional on `IsotropicPasses`, and that
nothing connected §5's re-isotropisation machinery to it.  This section builds that connection.

The setting is the one §5 can actually be instantiated in.  Mathlib has no `MeasurableSpace` on
`E ≃ₗᵢ[ℝ] E`, so the rotations are drawn from an abstract measurable group `G` acting on `E`,
together with a map `ι : G → (E ≃ₗᵢ[ℝ] E)` that *implements* the action (`ι g x = g • x`).  For
the paper's `O(d)` one takes `G = O(d)` with its Borel structure; for the deployed randomised
Hadamard transform, `G` is the finite sign-flip group, the results below are indifferent to
which, exactly as §5 is.

The chain is:

* `resid_rotScheme_eq_residPast`, `r_p` is a fixed function of `(w, g_0, …, g_{p-1})` only;
* `measurable_residPast`, that function is measurable;
* `unitVec_rotScheme_eq_smul`, `u_p = g_p • r̂_p` with `r̂_p` a function of the past;
* `indepFun_rot_past`, mutual independence of `(W, R_0, R_1, …)` makes the fresh `R_p`
  independent of the whole past, hence of `(r̂_p, ‖r_p‖²)`;
* `law_unitVec_eq_orbitLaw`, therefore `u_p` has the *same* law `σ = orbitLaw κ x₀` at every
  pass.  This is the statement re-isotropisation is for, and it is what makes the per-pass
  distortion the same number at every pass;
* `isotropicPasses_of_rotationStream`, assembly.

What is thereby removed from the hypothesis list of Proposition 2: `IsotropicPasses.indep` and
`IsotropicPasses.distortion`, the two substantive fields, and both integrability fields. -/

section BridgeAlgebra

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {G : Type*} [Group G] [MulAction G E]
