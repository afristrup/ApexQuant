
/-- **Corollary 1's optimality is optimality within a class.**  If `Q₁ ⊆ Q₂`, for instance `Q₁`
the coordinate-wise scalar quantisers and `Q₂` those together with the lattice or trellis
quantisers of the same rate, then the best attainable stage distortion over `Q₂` is no worse
than over `Q₁`.  Corollary 1 is therefore silent about, not evidence against, the richer class. -/
theorem sInf_stageDistortion_le_of_subset {Q₁ Q₂ : Set (E → E)} (hsub : Q₁ ⊆ Q₂)
    (hne : Q₁.Nonempty) (σ : Measure E) :
    sInf ((fun Q => stageDistortion Q σ) '' Q₂) ≤ sInf ((fun Q => stageDistortion Q σ) '' Q₁) :=
  csInf_le_csInf ⟨0, fun _ hx => by obtain ⟨Q, _, rfl⟩ := hx; exact stageDistortion_nonneg Q σ⟩
    (hne.image _) (Set.image_mono hsub)
