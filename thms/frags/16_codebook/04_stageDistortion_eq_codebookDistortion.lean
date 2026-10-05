
/-- Hence its stage distortion is the codebook distortion. -/
theorem stageDistortion_eq_codebookDistortion (hQ : IsNearest C Q) (σ : Measure E) :
    stageDistortion Q σ = codebookDistortion C σ := by
  unfold stageDistortion codebookDistortion
  simp_rw [norm_sub_eq_infDist hQ]
