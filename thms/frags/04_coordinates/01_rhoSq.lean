
/-- `ρ²` of eq. (2.3): `d · E_{t∼ν}[(t - cb t)²]`. -/
noncomputable def rhoSq (d : ℕ) (cb : ℝ → ℝ) (ν : Measure ℝ) : ℝ :=
  (d : ℝ) * ∫ t, (t - cb t) ^ 2 ∂ν
