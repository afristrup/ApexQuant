
/-- **Corollary 1, for every codebook class.**  If `C` minimises the codebook distortion over a
class `𝒞` of codebooks, all `2^b`-level product codebooks, all E8P copies, all trellis codes of
a given rate, or all of them together, then a nearest-codeword stage for `C` has stage
distortion no larger than any stage with values in any member of `𝒞`.  By
`stage_rho_pow_le_of_le` (§9) its `P`-pass envelope is then no larger for every `P`. -/
theorem stageDistortion_le_of_codebook_optimal {𝒞 : Set (Set E)}
    (hmin : ∀ C'' ∈ 𝒞, codebookDistortion C σ ≤ codebookDistortion C'' σ)
    (hQ : IsNearest C Q) (hC' : C' ∈ 𝒞) (hQ' : ∀ x, Q' x ∈ C')
    (hint : Integrable (fun u => ‖u - Q' u‖ ^ 2) σ) :
    stageDistortion Q σ ≤ stageDistortion Q' σ := by
  rw [stageDistortion_eq_codebookDistortion hQ]
  exact (hmin C' hC').trans (codebookDistortion_le_stageDistortion hQ' hint)

omit [MeasurableSpace E] in
