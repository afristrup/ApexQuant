
/-- The stage distortion of the halving stage against any probability law on the unit sphere
is exactly `1/4`. -/
theorem stageDistortion_halfE [MeasurableSpace E] {σ : Measure E} [IsProbabilityMeasure σ]
    (hσ : ∀ᵐ u ∂σ, ‖u‖ = 1) : stageDistortion halfE σ = 1 / 4 := by
  unfold stageDistortion
  have h : ∀ᵐ u ∂σ, ‖u - halfE u‖ ^ 2 = 1 / 4 := by
    filter_upwards [hσ] with u hu
    rw [norm_sub_halfE, hu]; norm_num
  rw [integral_congr_ae h, integral_const, probReal_univ, one_smul]
