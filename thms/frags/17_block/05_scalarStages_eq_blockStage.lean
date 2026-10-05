
/-- The scalar stages of §9 are the block stages at `F = ℝ`, definitionally. -/
theorem scalarStages_eq_blockStage (d : ℕ) (C : Set (ℝ → ℝ)) :
    scalarStages d C = {Q | ∃ cb ∈ C, Q = blockStage cb} := rfl

omit [MeasurableSpace F] in
