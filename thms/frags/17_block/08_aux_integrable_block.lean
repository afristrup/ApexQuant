
private theorem aux_integrable_block (qB : F → F) (hmarg : ∀ j, σ.map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) (j : Fin m) :
    Integrable (fun v : BlockSpace m F => ‖v j - qB (v j)‖ ^ 2) σ := by
  have hasm : AEStronglyMeasurable (fun x => ‖x - qB x‖ ^ 2) (σ.map (fun v => v j)) := by
    rw [hmarg j]; exact hint.aestronglyMeasurable
  exact (integrable_map_measure hasm (measurable_block j).aemeasurable).mp
    (by rw [hmarg j]; exact hint)
