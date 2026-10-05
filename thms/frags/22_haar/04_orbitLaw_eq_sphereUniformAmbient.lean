
/-- **The Haar identification.**  If `κ` is a right-invariant probability measure on a group `G`
acting on `E` by isometries and transitively on the unit sphere, then the law of `g • x₀` for
`g ∼ κ` is the uniform law on the sphere.  This is the uniqueness-of-invariant-measure step of
Proposition 1, proved directly: the uniform law is `G`-invariant (§21), so integrating a test
function against it, then averaging over `κ`, swapping the integrals, and using that the orbit
law is constant along the orbit (§5) returns the orbit law. -/
theorem orbitLaw_eq_sphereUniformAmbient [Nontrivial E] {ι : G → (E ≃ₗᵢ[ℝ] E)}
    (hι : ∀ (g : G) (x : E), ι g x = g • x)
    (κ : Measure G) [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] {x₀ : E}
    (htrans : ∀ x : E, ‖x‖ = 1 → x ∈ MulAction.orbit G x₀) :
    orbitLaw κ x₀ = sphereUniformAmbient E := by
  have key : ∀ f : E → ℝ≥0∞, Measurable f →
      ∫⁻ y, f y ∂(orbitLaw κ x₀) = ∫⁻ y, f y ∂(sphereUniformAmbient E) := by
    intro f hf
    have h1 : ∀ g : G, ∫⁻ x, f (g • x) ∂(sphereUniformAmbient E)
        = ∫⁻ y, f y ∂(sphereUniformAmbient E) := by
      intro g
      rw [← lintegral_map hf (measurable_const_smul g), sphereUniformAmbient_map_smul hι g]
    have h2 : ∫⁻ y, f y ∂(sphereUniformAmbient E)
        = ∫⁻ g, ∫⁻ x, f (g • x) ∂(sphereUniformAmbient E) ∂κ := by
      simp_rw [h1]; rw [lintegral_const, measure_univ, mul_one]
    have hmeas : Measurable (fun p : G × E => f (p.1 • p.2)) := hf.comp measurable_smul
    have h3 : ∫⁻ g, ∫⁻ x, f (g • x) ∂(sphereUniformAmbient E) ∂κ
        = ∫⁻ x, ∫⁻ g, f (g • x) ∂κ ∂(sphereUniformAmbient E) :=
      lintegral_lintegral_swap hmeas.aemeasurable
    have h4 : ∀ᵐ x ∂(sphereUniformAmbient E),
        ∫⁻ g, f (g • x) ∂κ = ∫⁻ y, f y ∂(orbitLaw κ x₀) := by
      filter_upwards [sphereUniformAmbient_ae_norm (E := E)] with x hx
      rw [← lintegral_orbitLaw κ x hf, orbitLaw_eq_of_mem_orbit κ (htrans x hx)]
    rw [h2, h3, lintegral_congr_ae h4, lintegral_const, measure_univ, mul_one]
  refine Measure.ext fun s hs => ?_
  have := key (s.indicator 1) (measurable_const.indicator hs)
  rwa [lintegral_indicator_one hs, lintegral_indicator_one hs] at this

end OrbitLawSphere

section RotationStreamUniform

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] [FiniteDimensional ℝ E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G] [MeasurableMul G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {q : E → E} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G}
  [κ.IsMulRightInvariant] {x₀ : E}
