
private theorem aux_integral_block (qB : F → F) (hmarg : ∀ j, σ.map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) (j : Fin m) :
    ∫ v, ‖v j - qB (v j)‖ ^ 2 ∂σ = ∫ x, ‖x - qB x‖ ^ 2 ∂ν := by
  have hasm : AEStronglyMeasurable (fun x => ‖x - qB x‖ ^ 2) (σ.map (fun v => v j)) := by
    rw [hmarg j]; exact hint.aestronglyMeasurable
  rw [← hmarg j, integral_map (measurable_block j).aemeasurable hasm]
