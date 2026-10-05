
/-- **Eq. (2.3) for a block stage**: `ρ² = E_ν‖x − q_B x‖² / E_ν‖x‖²`, the normalised block
distortion, with no reference to `m` or `d`. -/
theorem stageDistortion_blockStage_eq_ratio [IsProbabilityMeasure σ] (hm : m ≠ 0) (qB : F → F)
    (hmarg : ∀ j, σ.map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) (hint₂ : Integrable (fun x : F => ‖x‖ ^ 2) ν)
    (hunit : ∀ᵐ v ∂σ, ‖v‖ = 1) :
    stageDistortion (blockStage qB) σ = (∫ x, ‖x - qB x‖ ^ 2 ∂ν) / ∫ x, ‖x‖ ^ 2 ∂ν := by
  rw [stageDistortion_blockStage qB hmarg hint, integral_norm_sq_block hm hmarg hint₂ hunit]
  have hm' : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hm
  field_simp
