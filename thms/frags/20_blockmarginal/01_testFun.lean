
/-- The test function: `ψ` of the `U`-block of the direction, cut off at the unit ball. -/
noncomputable def testFun (ψ : U → ℝ≥0∞) (z : WithLp 2 (U × V)) : ℝ≥0∞ :=
  ψ (‖z‖⁻¹ • (WithLp.ofLp z).1) * (ball (0 : WithLp 2 (U × V)) 1).indicator 1 z
