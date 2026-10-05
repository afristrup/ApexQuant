
theorem image_cos_Ioo : Real.cos '' Set.Ioo 0 (Real.pi / 2) = Set.Ioo 0 1 := by
  ext ρ
  constructor
  · rintro ⟨θ, hθ, rfl⟩
    refine ⟨Real.cos_pos_of_mem_Ioo ⟨by linarith [hθ.1, Real.pi_pos], hθ.2⟩, ?_⟩
    have := Real.cos_lt_cos_of_nonneg_of_le_pi_div_two le_rfl hθ.2.le hθ.1
    rwa [Real.cos_zero] at this
  · intro hρ
    exact ⟨Real.arccos ρ, ⟨Real.arccos_pos.mpr hρ.2, Real.arccos_lt_pi_div_two.mpr hρ.1⟩,
      Real.cos_arccos (by linarith [hρ.1]) hρ.2.le⟩
