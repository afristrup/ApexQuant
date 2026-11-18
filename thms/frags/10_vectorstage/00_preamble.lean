/-! ## §9  Beyond scalar codebooks: where Corollary 1's optimality actually stops

Corollary 1 (§6) says the density-matched Lloyd–Max codebook is MSE-optimal *among `2^b`-level
scalar quantisers applied coordinate-wise*.  That qualifier is not a footnote.  Trellis-coded
quantisation (QTIP) and lattice codebooks (QuIP#'s E8P) exist precisely to leave that class, and
on a Gaussian source they recover a shaping gain that no scalar quantiser can reach at any rate.
A reader who takes "one codebook is optimal at every pass" at face value will be misled about
exactly the comparison the low-bit literature cares about.

Two facts, both proved below, place the boundary precisely:

* **The decay law never used the coordinate structure.**  `Scheme.q` is an arbitrary map
  `E → E`, and `IsotropicPasses.distortion` is stated as `∫‖u_p − q u_p‖² = ρ²` with no
  reference to coordinates.  So §3's geometric envelope applies verbatim to a trellis or lattice
  stage quantiser (`integral_norm_resid_sq_vectorStage`).  The progressive construction is *not*
  tied to scalar codebooks; only the evaluation of `ρ` by 1-D quadrature is.

* **Enlarging the class can only help** (`sInf_stageDistortion_le_of_subset`).  So "optimal at
  every pass" is optimality within a strictly smaller class whenever the containment is strict,
  and the attainable decay constant over a richer class is no larger.

Nothing here quantifies the gap, that is an information-theoretic statement about the source,
not a consequence of this recursion.  What is established is that the gap is on the right side of
the inequality, and that the framework does not have to pay it. -/

section VectorStage

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
