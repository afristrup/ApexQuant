/-- Nearest-codeword per block is nearest-codeword for the product codebook.  So coordinate-wise
Lloyd–Max rounding implements the product of `2^b`-point sets and per-block E8P rounding
implements the product of E8P copies, in the sense of §16. -/
theorem isNearest_blockStage {qB : F → F} {CB : Set F} (h : IsNearest CB qB) :
    IsNearest {v : BlockSpace m F | ∀ j, v j ∈ CB} (blockStage qB) := by
  intro v
  refine ⟨fun j => by simpa [blockStage] using (h (v j)).1, fun c hc => ?_⟩
  rw [← pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero,
    norm_sub_blockStage_sq, PiLp.norm_sq_eq_of_L2]
  simp only [PiLp.sub_apply]
  exact Finset.sum_le_sum fun j _ => pow_le_pow_left₀ (norm_nonneg _) ((h (v j)).2 _ (hc j)) 2

variable {σ : Measure (BlockSpace m F)} {ν : Measure F}
