
/-- **The block quadrature.**  If every block of the direction has the same law `ν`, the stage
distortion of the block stage is `m · E_ν‖x − q_B x‖²`.  This is `integral_norm_sub_cb_sq` (§3b)
with the coordinate replaced by a block; no independence between blocks is used. -/
theorem stageDistortion_blockStage (qB : F → F) (hmarg : ∀ j, σ.map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) :
    stageDistortion (blockStage qB) σ = m * ∫ x, ‖x - qB x‖ ^ 2 ∂ν := by
  unfold stageDistortion
  simp_rw [norm_sub_blockStage_sq]
  rw [integral_finsetSum _ (fun j _ => aux_integrable_block qB hmarg hint j)]
  simp_rw [aux_integral_block qB hmarg hint]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
