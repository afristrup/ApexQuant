theorem finrank_withLp_prod :
    Module.finrank ℝ (WithLp 2 (U × V)) = Module.finrank ℝ U + Module.finrank ℝ V := by
  rw [(WithLp.linearEquiv 2 ℝ (U × V)).finrank_eq, Module.finrank_prod]
