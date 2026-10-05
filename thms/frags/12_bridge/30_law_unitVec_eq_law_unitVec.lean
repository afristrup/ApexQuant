
/-- The direction law is *the same at every pair of passes* — the citable form of
"re-isotropisation restores isotropy at every stage". -/
theorem law_unitVec_eq_law_unitVec (h : RotationStream ι q W R κ x₀ μ) (p p' : ℕ) :
    μ.map (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p)
      = μ.map (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p') := by
  rw [law_unitVec_eq_orbitLaw h p, law_unitVec_eq_orbitLaw h p']
