/-! ## §20  Lemma 1, block form: the marginal of a block of coordinates

For `E = U × V` (Euclidean product) and `u` uniform on the sphere of `E`, the `U`-block of `u`
has density proportional to `(1 − ‖x‖²)^{(dim V − 2)/2}` on the unit ball of `U`.  The proof is
the classical one, made exact: integrate a test function `ψ(u_U)` cut off at the unit ball, once
by the polar formula on `E`, once by the product structure and the polar formulas on `U` and `V`
followed by a polar change on the two radii.  The radial factors agree and cancel, leaving an
angular integral on the sphere of `U` (`lintegral_sphere_fst`); the same angular integral is what
the candidate density produces after the substitution `ρ = cos θ` (`lintegral_ball_density`).
`toSphere_map_fst` states the result as an equality of measures. -/

section Block
variable {U V : Type*} [NormedAddCommGroup U] [InnerProductSpace ℝ U] [MeasurableSpace U]
  [BorelSpace U] [FiniteDimensional ℝ U] [Nontrivial U]
  [NormedAddCommGroup V] [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V]
  [FiniteDimensional ℝ V] [Nontrivial V]
