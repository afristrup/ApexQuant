
/-- The stage distortion of an **arbitrary** quantiser `Q` against the direction law `σ` — the
measure §5 calls `orbitLaw`, i.e. the law of `u_p` on the unit sphere.  This is `ρ²` of
eq. (2.3) with the coordinate structure removed. -/
noncomputable def stageDistortion (Q : E → E) (σ : Measure E) : ℝ := ∫ u, ‖u - Q u‖ ^ 2 ∂σ
