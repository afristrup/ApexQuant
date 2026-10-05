
/-! ### The identification -/

/-- **Proposition 1 at `O(d)`, in full.**  The law of `g • x₀` for `g` Haar-distributed on
`O(d)` and `x₀` a unit vector is the uniform law on the sphere. -/
theorem orbitLaw_haar (x₀ : Ed d) (h₀ : ‖x₀‖ = 1) :
    orbitLaw (haarOd d) x₀ = sphereUniformAmbient (Ed d) := by
  haveI : Nontrivial (Ed d) := ⟨⟨x₀, 0, by rintro rfl; simp at h₀⟩⟩
  exact orbitLaw_eq_sphereUniformAmbient (ι := iso d) (fun g x => iso_apply d g x) (haarOd d)
    (fun x hx => transitive d h₀ x hx)

end Orthogonal
