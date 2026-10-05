
/-- The block marginal of the ambient uniform law: the `hmarg` hypothesis of §17's block
quadrature, discharged for the uniform direction. -/
theorem sphereUniformAmbient_map_block {U V : Type*} [NormedAddCommGroup U]
    [InnerProductSpace ℝ U] [MeasurableSpace U] [BorelSpace U] [FiniteDimensional ℝ U]
    [Nontrivial U] [NormedAddCommGroup V] [InnerProductSpace ℝ V] [MeasurableSpace V]
    [BorelSpace V] [FiniteDimensional ℝ V] [Nontrivial V] (L : E ≃ₗᵢ[ℝ] WithLp 2 (U × V)) :
    (sphereUniformAmbient E).map (fun x : E => (WithLp.ofLp (L x)).1)
      = blockLaw U (Module.finrank ℝ V) := by
  have hg : Measurable fun x : E => (WithLp.ofLp (L x)).1 := by
    exact measurable_fst.comp ((WithLp.measurable_ofLp 2 (U × V)).comp L.continuous.measurable)
  rw [sphereUniformAmbient, Measure.map_map hg measurable_subtype_coe]
  exact sphereUniform_map_block L
