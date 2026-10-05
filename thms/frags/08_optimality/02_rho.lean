
/-- `ρ = √(d · D(cb))`. -/
noncomputable def rho (d : ℕ) (cb : ℝ → ℝ) (ν : Measure ℝ) : ℝ :=
  Real.sqrt (rhoSq d cb ν)
