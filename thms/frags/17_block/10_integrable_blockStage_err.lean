
/-- Integrability of the stage error, from integrability of one block's error. -/
theorem integrable_blockStage_err (qB : F → F) (hmarg : ∀ j, σ.map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) :
    Integrable (fun v : BlockSpace m F => ‖v - blockStage qB v‖ ^ 2) σ := by
  simp_rw [norm_sub_blockStage_sq]
  exact integrable_finsetSum _ fun j _ => aux_integrable_block qB hmarg hint j
