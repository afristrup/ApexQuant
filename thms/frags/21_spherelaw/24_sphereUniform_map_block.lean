
/-- **Lemma 1, block form, for any orthogonal decomposition.**  If `L : E ≃ U × V` is a linear
isometry, the `U`-component of a uniform direction has law `blockLaw U (dim V)`.  For
`E = ℝ^d`, `U` the coordinates of one block of eight and `V` the rest, this is the law `f_{d,8}`
that the `E₈` stage's `ρ` is a quadrature over; for a trellis block of length `T` it is
`f_{d,T}`. -/
theorem sphereUniform_map_block {U V : Type*} [NormedAddCommGroup U] [InnerProductSpace ℝ U]
    [MeasurableSpace U] [BorelSpace U] [FiniteDimensional ℝ U] [Nontrivial U]
    [NormedAddCommGroup V] [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V]
    [FiniteDimensional ℝ V] [Nontrivial V] (L : E ≃ₗᵢ[ℝ] WithLp 2 (U × V)) :
    (sphereUniform E).map (fun u : sphere (0 : E) 1 => (WithLp.ofLp (L u)).1)
      = blockLaw U (Module.finrank ℝ V) := by
  have hcomp : (fun u : sphere (0 : E) 1 => (WithLp.ofLp (L u)).1)
      = (fun u : sphere (0 : WithLp 2 (U × V)) 1 => (WithLp.ofLp (u : WithLp 2 (U × V))).1)
          ∘ sphereMap L := by
    funext u; rfl
  rw [hcomp, ← Measure.map_map measurable_sphere_fst (measurable_sphereMap L),
    sphereUniform_map_sphereMap L, sphereUniform_map_fst]
