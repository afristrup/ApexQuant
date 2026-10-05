
/-- The integrand of the `V`-slice at a fixed `x : U`, as a function of `‖y‖`. -/
noncomputable def sliceFun (ψ : U → ℝ≥0∞) (x : U) (s : ℝ) : ℝ≥0∞ :=
  ψ ((√(‖x‖ ^ 2 + s ^ 2))⁻¹ • x) * (Set.Iio (1 : ℝ)).indicator 1 (‖x‖ ^ 2 + s ^ 2)

omit [Nontrivial U] [Nontrivial V] in
