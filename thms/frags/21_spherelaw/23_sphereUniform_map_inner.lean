
/-- **Lemma 1.**  For `u` uniform on the unit sphere of `E` (`dim E ≥ 2`) and any unit vector
`e`, the coordinate `⟪e, u⟫` has the law `sphericalCoordLaw (dim E)`, i.e. the density `f_d`. -/
theorem sphereUniform_map_inner (hd : 2 ≤ Module.finrank ℝ E) {e : E} (he : ‖e‖ = 1) :
    (sphereUniform E).map (fun u : sphere (0 : E) 1 => ⟪e, (u : E)⟫_ℝ)
      = sphericalCoordLaw (Module.finrank ℝ E) := by
  haveI : Nonempty (Fin (Module.finrank ℝ E - 1)) := ⟨⟨0, by omega⟩⟩
  haveI : Nontrivial (EuclideanSpace ℝ (Fin (Module.finrank ℝ E - 1))) :=
    (WithLp.equiv 2 _).symm.injective.nontrivial
  haveI : Nontrivial E := Module.nontrivial_of_finrank_pos (R := ℝ) (by omega)
  let L₀ := toModel E (by omega)
  obtain ⟨M, hM⟩ := exists_isometry_map_eq (u := L₀ e) (v := modelVec _)
    (by rw [L₀.norm_map, he]) (norm_modelVec _)
  let L := L₀.trans M
  have hL : L e = modelVec _ := by simp [L, hM]
  have hcoord : ∀ u : E, ⟪e, u⟫_ℝ = (WithLp.ofLp (L u)).1 := by
    intro u
    rw [← L.inner_map_map e u, hL, inner_modelVec]
  have hcomp : (fun u : sphere (0 : E) 1 => ⟪e, (u : E)⟫_ℝ)
      = (fun u : sphere (0 : modelSpace (Module.finrank ℝ E - 1)) 1 =>
          (WithLp.ofLp (u : modelSpace (Module.finrank ℝ E - 1))).1) ∘ sphereMap L := by
    funext u
    simp [hcoord]
  rw [hcomp, ← Measure.map_map measurable_sphere_fst (measurable_sphereMap L),
    sphereUniform_map_sphereMap L, sphereUniform_map_fst, finrank_euclideanSpace_fin,
    blockLaw_real _ (by omega)]
  congr 1
  omega
