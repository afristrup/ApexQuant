
theorem lintegral_orbitLaw (κ : Measure G) (x : E) {f : E → ℝ≥0∞} (hf : Measurable f) :
    ∫⁻ y, f y ∂(orbitLaw κ x) = ∫⁻ g, f (g • x) ∂κ := by
  rw [orbitLaw, lintegral_map hf (measurable_smul_const x)]
