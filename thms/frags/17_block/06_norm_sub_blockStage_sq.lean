/-- The squared stage error splits over blocks, exactly and pointwise. -/
theorem norm_sub_blockStage_sq (qB : F → F) (v : BlockSpace m F) :
    ‖v - blockStage qB v‖ ^ 2 = ∑ j, ‖v j - qB (v j)‖ ^ 2 := by
  rw [PiLp.norm_sq_eq_of_L2]; simp [blockStage]

omit [MeasurableSpace F] in
