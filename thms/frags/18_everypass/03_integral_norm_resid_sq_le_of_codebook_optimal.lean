
/-- **Corollary 1 for every codebook, at the residual level.**  A nearest-codeword stage for the
codebook that minimises `codebookDistortion` over a class beats every stage with values in any
member of that class, at every pass count.  The paper's Corollary 1 is the case where the class
is the `2^b`-level product codebooks; the E8P and trellis classes are the same theorem. -/
theorem integral_norm_resid_sq_le_of_codebook_optimal {𝒞 : Set (Set E)} {C C' : Set E}
    {q q' : E → E}
    (hmin : ∀ C'' ∈ 𝒞, codebookDistortion C (orbitLaw κ x₀) ≤ codebookDistortion C'' (orbitLaw κ x₀))
    (hQ : IsNearest C q) (hC' : C' ∈ 𝒞) (hQ' : ∀ x, q' x ∈ C')
    (h : RotationStream ι q W R κ x₀ μ) (h' : RotationStream ι q' W R κ x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀))
    (hint' : Integrable (fun u : E => ‖u - q' u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι q (R ω)) (W ω) P‖ ^ 2 ∂μ
      ≤ ∫ ω, ‖resid (rotScheme ι q' (R ω)) (W ω) P‖ ^ 2 ∂μ :=
  integral_norm_resid_sq_le_of_stageDistortion_le h h' hW hint hint'
    (stageDistortion_le_of_codebook_optimal hmin hQ hC' hQ' hint') P

end EveryPass

section BlockEveryPass

variable {m : ℕ} {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
  [MeasurableSpace F] [BorelSpace F] [SecondCountableTopology F]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G (BlockSpace m F)]
  [MeasurableSMul₂ G (BlockSpace m F)] [MeasurableInv G] [MeasurableMul G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : G → (BlockSpace m F ≃ₗᵢ[ℝ] BlockSpace m F)} {W : Ω → BlockSpace m F}
  {R : Ω → ℕ → G} {κ : Measure G} [κ.IsMulRightInvariant] {x₀ : BlockSpace m F}
