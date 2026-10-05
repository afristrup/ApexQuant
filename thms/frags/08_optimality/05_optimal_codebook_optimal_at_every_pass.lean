
/-- **Corollary 1 as stated**: if `cb★` minimises the per-coordinate distortion over a class `Q`
of `2^b`-level scalar quantisers, it minimises the `P`-pass error for every `P`, so no per-pass
re-fitting can improve on it. -/
theorem optimal_codebook_optimal_at_every_pass {Q : Set (ℝ → ℝ)} {cb : ℝ → ℝ}
    (hmem : cb ∈ Q) (hmin : ∀ c ∈ Q, distortion cb ν ≤ distortion c ν) (P : ℕ) :
    ∀ c ∈ Q, rho d cb ν ^ P ≤ rho d c ν ^ P :=
  fun c hc => rho_pow_le_of_distortion_le (hmin c hc) P
