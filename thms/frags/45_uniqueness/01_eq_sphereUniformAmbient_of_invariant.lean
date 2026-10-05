
/-- **Uniqueness of the invariant law on the sphere.**  If `G` acts on `E` by isometries and
transitively on the unit sphere, and carries a right-invariant probability measure `κ`, then
every `G`-invariant probability measure on `E` carried by the unit sphere is the uniform law:
average the invariance `μ = (g • ·)_* μ` over `g ∼ κ`, swap the integrals, and use that the
orbit law of every point of the sphere is the uniform law (§22). -/
theorem eq_sphereUniformAmbient_of_invariant [Nontrivial E] {ι : G → (E ≃ₗᵢ[ℝ] E)}
    (hι : ∀ (g : G) (x : E), ι g x = g • x)
    (κ : Measure G) [IsProbabilityMeasure κ] [κ.IsMulRightInvariant]
    (htrans : ∀ x y : E, ‖x‖ = 1 → ‖y‖ = 1 → y ∈ MulAction.orbit G x)
    (μ : Measure E) [IsProbabilityMeasure μ] (hμ : ∀ᵐ x ∂μ, ‖x‖ = 1)
    (hinv : ∀ g : G, μ.map (fun x => g • x) = μ) :
    μ = sphereUniformAmbient E := by
  ext s hs
  have hmeas : Measurable fun p : G × E => s.indicator (1 : E → ℝ≥0∞) (p.1 • p.2) :=
    (measurable_const.indicator hs).comp measurable_smul
  have h1 : μ s = ∫⁻ g, ∫⁻ x, s.indicator (1 : E → ℝ≥0∞) (g • x) ∂μ ∂κ := by
    have : ∀ g : G, ∫⁻ x, s.indicator (1 : E → ℝ≥0∞) (g • x) ∂μ = μ s := by
      intro g
      have hg : Measurable fun x : E => g • x := measurable_const_smul g
      calc ∫⁻ x, s.indicator (1 : E → ℝ≥0∞) (g • x) ∂μ
          = ∫⁻ x, ((fun x : E => g • x) ⁻¹' s).indicator (1 : E → ℝ≥0∞) x ∂μ := by
            refine lintegral_congr fun x => ?_
            by_cases hx : g • x ∈ s
            · rw [Set.indicator_of_mem hx, Set.indicator_of_mem (by exact hx)]; rfl
            · rw [Set.indicator_of_notMem hx, Set.indicator_of_notMem (by exact hx)]
        _ = μ ((fun x : E => g • x) ⁻¹' s) := lintegral_indicator_one (hg hs)
        _ = (μ.map fun x : E => g • x) s := (Measure.map_apply hg hs).symm
        _ = μ s := by rw [hinv g]
    simp_rw [this]
    rw [lintegral_const, measure_univ, mul_one]
  have h2 : ∫⁻ g, ∫⁻ x, s.indicator (1 : E → ℝ≥0∞) (g • x) ∂μ ∂κ
      = ∫⁻ x, ∫⁻ g, s.indicator (1 : E → ℝ≥0∞) (g • x) ∂κ ∂μ :=
    lintegral_lintegral_swap (f := fun g x => s.indicator (1 : E → ℝ≥0∞) (g • x))
      hmeas.aemeasurable
  have h3 : ∀ x : E, ∫⁻ g, s.indicator (1 : E → ℝ≥0∞) (g • x) ∂κ = (orbitLaw κ x) s := by
    intro x
    have := lintegral_indicator_one (μ := orbitLaw κ x) hs
    rw [← this, lintegral_orbitLaw κ x (f := s.indicator 1) (measurable_const.indicator hs)]
  have h4 : ∫⁻ x, (orbitLaw κ x) s ∂μ = ∫⁻ x, (sphereUniformAmbient E) s ∂μ := by
    refine lintegral_congr_ae ?_
    filter_upwards [hμ] with x hx
    rw [orbitLaw_eq_sphereUniformAmbient hι κ (fun y hy => htrans x y hx hy)]
  rw [h1, h2]
  simp_rw [h3]
  rw [h4, lintegral_const, measure_univ, mul_one]

end Uniqueness

namespace Orthogonal
variable (d : ℕ)
