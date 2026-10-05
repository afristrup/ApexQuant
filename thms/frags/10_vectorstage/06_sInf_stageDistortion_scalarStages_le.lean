
/-- Instantiating the previous lemma: whatever richer class `Qs` a trellis or lattice stage
quantiser lives in, as long as it also contains the scalar stages, its optimum is at least as
good.  This is the formal content of "Corollary 1's scope is scalar quantisers". -/
theorem sInf_stageDistortion_scalarStages_le {d : ℕ} {C : Set (ℝ → ℝ)}
    {Qs : Set (EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))}
    (hsub : scalarStages d C ⊆ Qs) (hne : (scalarStages d C).Nonempty)
    (σ : Measure (EuclideanSpace ℝ (Fin d))) :
    sInf ((fun Q => stageDistortion Q σ) '' Qs)
      ≤ sInf ((fun Q => stageDistortion Q σ) '' scalarStages d C) :=
  sInf_stageDistortion_le_of_subset hsub hne σ

end VectorStage
