
/-- **Lemma 1, block form, with its constant**: the law of a `B`-block of a uniform direction on
the sphere of dimension `B + m` has density `Γ((B+m)/2)/(√π^B Γ(m/2)) · (1 − ‖x‖²)^{(m−2)/2}` on
the unit ball. -/
theorem blockLaw_eq_withDensity (m : ℕ) (hm : 1 ≤ m) :
    blockLaw U m = (volume.restrict (ball (0 : U) 1)).withDensity fun x =>
      ENNReal.ofReal ((blockMass (Module.finrank ℝ U) m)⁻¹ * (1 - ‖x‖ ^ 2) ^ (((m : ℝ) - 2) / 2)) := by
  rw [blockLaw, blockMeasure_univ U m hm, ← ENNReal.ofReal_inv_of_pos (blockMass_pos hm), blockMeasure,
    ← withDensity_smul _ (measurable_blockDensity m)]
  congr 1
  funext x
  rw [Pi.smul_apply, smul_eq_mul, blockDensity, ← ENNReal.ofReal_mul (inv_nonneg.mpr (blockMass_pos hm).le)]

end BlockConstant

