
theorem norm_sub_halfE (u : E) : ‖u - halfE u‖ = ‖u‖ / 2 := by
  have : u - halfE u = (2 : ℝ)⁻¹ • u := by
    rw [halfE, sub_eq_iff_eq_add, ← add_smul]; norm_num
  rw [this, norm_smul, norm_inv, Real.norm_ofNat, div_eq_inv_mul]
