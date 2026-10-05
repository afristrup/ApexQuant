
/-- Polar decomposition of a Lebesgue integral on a finite-dimensional inner product space. -/
theorem lintegral_eq_polar (F : E → ℝ≥0∞) (hF : Measurable F) :
    ∫⁻ x, F x = ∫⁻ u : sphere (0 : E) 1, (∫⁻ r in Set.Ioi (0 : ℝ),
      ENNReal.ofReal (r ^ (Module.finrank ℝ E - 1)) * F (r • (u : E))) ∂volume.toSphere := by
  have h1 : ∫⁻ x, F x = ∫⁻ x in ({0}ᶜ : Set E), F x := by rw [restrict_compl_singleton]
  have h2 : ∫⁻ x in ({0}ᶜ : Set E), F x
      = ∫⁻ x : ({0}ᶜ : Set E), F x ∂(volume.comap Subtype.val) := by
    rw [← setLIntegral_univ (μ := volume.comap Subtype.val),
      setLIntegral_subtype (measurableSet_singleton _).compl Set.univ F, Set.image_univ, Subtype.range_coe]
  have h3 := (measurePreserving_homeomorphUnitSphereProd (volume : Measure E)).lintegral_comp_emb
    (Homeomorph.measurableEmbedding _)
    (fun p : sphere (0 : E) 1 × Set.Ioi (0 : ℝ) => F ((p.2 : ℝ) • (p.1 : E)))
  have h4 : ∀ a : ({0}ᶜ : Set E),
      F (((homeomorphUnitSphereProd E a).2 : ℝ) • ((homeomorphUnitSphereProd E a).1 : E)) = F a := by
    intro a
    rw [homeomorphUnitSphereProd_apply_snd_coe, homeomorphUnitSphereProd_apply_fst_coe, smul_smul,
      mul_inv_cancel₀ (norm_ne_zero_iff.mpr (Set.mem_compl_singleton_iff.mp a.2)), one_smul]
  simp only [h4] at h3
  have hmeas : Measurable (fun p : sphere (0 : E) 1 × Set.Ioi (0 : ℝ) => F ((p.2 : ℝ) • (p.1 : E))) := by
    fun_prop
  rw [h1, h2, h3, lintegral_prod _ hmeas.aemeasurable]
  refine lintegral_congr fun u => ?_
  rw [volumeIoiPow, lintegral_withDensity_eq_lintegral_mul _ (by fun_prop) (by fun_prop)]
  have hpt : ((fun r : Set.Ioi (0 : ℝ) => ENNReal.ofReal ((r : ℝ) ^ (Module.finrank ℝ E - 1)))
      * fun y : Set.Ioi (0 : ℝ) => F (((u, y).2 : ℝ) • ((u, y).1 : E)))
      = fun r : Set.Ioi (0 : ℝ) =>
        (fun t : ℝ => ENNReal.ofReal (t ^ (Module.finrank ℝ E - 1)) * F (t • (u : E))) r := by
    funext r; rfl
  rw [hpt, ← setLIntegral_univ (μ := volume.comap Subtype.val),
    setLIntegral_subtype measurableSet_Ioi Set.univ
      (fun t : ℝ => ENNReal.ofReal (t ^ (Module.finrank ℝ E - 1)) * F (t • (u : E))),
    Set.image_univ, Subtype.range_coe]

end Polar

section Radial
