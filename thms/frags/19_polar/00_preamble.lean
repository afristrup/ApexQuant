/-! ## §19  The uniform measure on the sphere, and its polar decomposition

Header item 1 records that Lemma 1's content, that a coordinate of a uniform point on `S^{d−1}`
has density `f_d`, was absent because "Mathlib has no usable handle on the uniform measure on a
sphere".  It does: `MeasureTheory.Measure.toSphere` is the surface measure obtained from
Lebesgue measure by the cone construction, and `measurePreserving_homeomorphUnitSphereProd` is
the polar decomposition `volume ≅ toSphere ⊗ r^{d−1} dr` on `E ∖ {0}`.  §19 to §21 use it to
prove Lemma 1, in the block form the vector stages of §17 need.

This section has the tools: the polar formula for a Lebesgue integral in `lintegral` form
(`lintegral_eq_polar`), the radial constant `∫₀¹ rⁿ dr`, and the two-dimensional polar change
of variables on a quadrant, which is where the two radii of a block decomposition `E = U × V`
meet.  Everything is done in `ℝ≥0∞`, so no integrability side condition ever arises. -/

open Measure Set Metric

section Polar
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]
