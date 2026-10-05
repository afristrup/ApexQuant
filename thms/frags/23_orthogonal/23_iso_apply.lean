
theorem iso_apply (g : Od d) (x : Ed d) : iso d g x = g • x := by
  rw [iso, LinearIsometry.coe_toLinearIsometryEquiv]; rfl
