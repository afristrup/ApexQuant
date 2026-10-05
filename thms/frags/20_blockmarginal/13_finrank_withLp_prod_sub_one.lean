
theorem finrank_withLp_prod_sub_one :
    Module.finrank ℝ (WithLp 2 (U × V)) - 1
      = (Module.finrank ℝ U - 1) + (Module.finrank ℝ V - 1) + 1 := by
  have hU : 0 < Module.finrank ℝ U := Module.finrank_pos
  have hV : 0 < Module.finrank ℝ V := Module.finrank_pos
  rw [finrank_withLp_prod]
  omega

set_option maxHeartbeats 1000000 in
