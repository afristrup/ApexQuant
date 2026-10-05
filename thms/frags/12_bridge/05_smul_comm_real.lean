
/-- If `ι` implements the `G`-action, that action is `ℝ`-linear. -/
theorem smul_comm_real (hι : ∀ (g : G) (x : E), ι g x = g • x) (g : G) (c : ℝ) (x : E) :
    g • (c • x) = c • (g • x) := by
  rw [← hι, ← hι, map_smul]
