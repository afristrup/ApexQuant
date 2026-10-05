
/-- **Lemma 1, block form.**  Under the uniform law on the sphere of `U × V`, the `U`-block has
law `blockLaw U (dim V)`, i.e. density proportional to `(1 − ‖x‖²)^{(dim V − 2)/2}` on the unit
ball of `U`. -/
theorem sphereUniform_map_fst :
    (sphereUniform (WithLp 2 (U × V))).map
        (fun u : sphere (0 : WithLp 2 (U × V)) 1 => (WithLp.ofLp (u : WithLp 2 (U × V))).1)
      = blockLaw U (Module.finrank ℝ V) := by
  have h1 : ((sphereUniform (WithLp 2 (U × V))).map
      (fun u : sphere (0 : WithLp 2 (U × V)) 1 => (WithLp.ofLp (u : WithLp 2 (U × V))).1)) Set.univ
      = 1 := by
    rw [Measure.map_apply measurable_sphere_fst MeasurableSet.univ, Set.preimage_univ,
      measure_univ]
  rw [sphereUniform, Measure.map_smul, toSphere_map_fst, smul_smul] at h1 ⊢
  rw [Measure.smul_apply, smul_eq_mul] at h1
  rw [blockLaw, ENNReal.eq_inv_of_mul_eq_one_left h1]
