
theorem norm_smul' (g : Od d) (x : Ed d) : ‖g • x‖ = ‖x‖ := by
  rw [norm_eq_sqrt_real_inner, norm_eq_sqrt_real_inner, inner_smul]
