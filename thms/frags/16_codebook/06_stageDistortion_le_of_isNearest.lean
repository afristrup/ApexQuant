
/-- Nearest-codeword rounding is optimal among all maps into the same codebook. -/
theorem stageDistortion_le_of_isNearest (hQ : IsNearest C Q) (hQ' : ∀ x, Q' x ∈ C)
    (hint : Integrable (fun u => ‖u - Q' u‖ ^ 2) σ) :
    stageDistortion Q σ ≤ stageDistortion Q' σ := by
  rw [stageDistortion_eq_codebookDistortion hQ]
  exact codebookDistortion_le_stageDistortion hQ' hint
