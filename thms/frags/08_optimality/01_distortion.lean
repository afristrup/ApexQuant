
/-- Per-coordinate distortion of a scalar quantiser under `ν`. -/
noncomputable def distortion (cb : ℝ → ℝ) (ν : Measure ℝ) : ℝ := ∫ t, (t - cb t) ^ 2 ∂ν
