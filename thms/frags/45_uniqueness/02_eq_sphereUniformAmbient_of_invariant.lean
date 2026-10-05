
/-- **Uniqueness at `O(d)`.**  Every `O(d)`-invariant probability measure on `ℝ^d` carried by
the unit sphere is the uniform law on the sphere.  With `orbitLaw_haar` this is Proposition 1
in the form the paper's proof invokes: the uniform law is *the* rotation-invariant law. -/
theorem eq_sphereUniformAmbient_of_invariant (μ : Measure (Ed d)) [IsProbabilityMeasure μ]
    (hμ : ∀ᵐ x ∂μ, ‖x‖ = 1) (hinv : ∀ g : Od d, μ.map (fun x => g • x) = μ) :
    μ = sphereUniformAmbient (Ed d) := by
  haveI : (ae μ).NeBot := ae_neBot.mpr (IsProbabilityMeasure.ne_zero μ)
  obtain ⟨x₀, h₀⟩ := hμ.exists
  haveI : Nontrivial (Ed d) := ⟨⟨x₀, 0, by rintro rfl; simp at h₀⟩⟩
  exact Progressive.eq_sphereUniformAmbient_of_invariant (ι := iso d) (fun g x => iso_apply d g x)
    (haarOd d) (fun x y hx hy => transitive d hx y hy) μ hμ hinv

end Orthogonal

