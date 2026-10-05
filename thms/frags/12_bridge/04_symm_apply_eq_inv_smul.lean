
/-- If `ι` implements the `G`-action, the inverse isometry implements the inverse action. -/
theorem symm_apply_eq_inv_smul (hι : ∀ (g : G) (x : E), ι g x = g • x) (g : G) (y : E) :
    (ι g).symm y = g⁻¹ • y := by
  have h : ι g (g⁻¹ • y) = y := by rw [hι, smul_inv_smul]
  conv_lhs => rw [← h]
  rw [LinearIsometryEquiv.symm_apply_apply]
