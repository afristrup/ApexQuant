
/-- The ambient uniform law is invariant under every element of a group acting by isometries. -/
theorem sphereUniformAmbient_map_smul {ι : G → (E ≃ₗᵢ[ℝ] E)}
    (hι : ∀ (g : G) (x : E), ι g x = g • x) (g : G) :
    (sphereUniformAmbient E).map (fun x => g • x) = sphereUniformAmbient E := by
  have h1 : (fun x : E => g • x) ∘ (Subtype.val : sphere (0 : E) 1 → E)
      = Subtype.val ∘ sphereMap (ι g) := by
    funext u; simp [hι]
  rw [sphereUniformAmbient, Measure.map_map (measurable_const_smul g) measurable_subtype_coe, h1,
    ← Measure.map_map measurable_subtype_coe (measurable_sphereMap _), sphereUniform_map_sphereMap]
