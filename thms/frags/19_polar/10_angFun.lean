
/-- The angular factor, cut off to the first quadrant. -/
noncomputable def angFun (a b : ℕ) (g : ℝ → ℝ≥0∞) (θ : ℝ) : ℝ≥0∞ :=
  (Set.Ioo 0 (Real.pi / 2)).indicator (fun θ => angWeight a b θ * g (Real.cos θ)) θ
