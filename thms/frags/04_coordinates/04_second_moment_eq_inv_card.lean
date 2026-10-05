
/-- `E_ν[t²] = 1/d` is *forced*: it is `‖U‖ = 1` plus the coordinates sharing a law, not an
extra assumption about `f_d`.  (This is the step the paper's proof records parenthetically.) -/
theorem second_moment_eq_inv_card
    {U : Ω → EuclideanSpace ℝ (Fin d)} {ν : Measure ℝ} [IsProbabilityMeasure μ] (hd : d ≠ 0)
    (hmeas : ∀ i, Measurable fun ω => U ω i)
    (hmarg : ∀ i, μ.map (fun ω => U ω i) = ν)
    (hint : Integrable (fun t => t ^ 2) ν)
    (hunit : ∀ ω, ‖U ω‖ = 1) :
    ∫ t, t ^ 2 ∂ν = 1 / (d : ℝ) := by
  have hd' : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hd
  have hasm : ∀ i, AEStronglyMeasurable (fun t : ℝ => t ^ 2) (μ.map (fun ω => U ω i)) := by
    intro i; rw [hmarg i]; exact hint.aestronglyMeasurable
  have hmapint : ∀ i, Integrable (fun ω => (U ω i) ^ 2) μ := fun i =>
    (integrable_map_measure (hasm i) (hmeas i).aemeasurable).mp (by rw [hmarg i]; exact hint)
  have hcoord : ∀ i : Fin d, ∫ ω, (U ω i) ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν := by
    intro i; rw [← hmarg i, integral_map (hmeas i).aemeasurable (hasm i)]
  -- `‖U ω‖ = 1` pointwise, i.e. the coordinate squares sum to `1` on the nose.
  have hone : ∀ ω : Ω, ∑ i, (U ω i) ^ 2 = 1 := by
    intro ω
    rw [← EuclideanSpace.real_norm_sq_eq, hunit ω, one_pow]
  -- Integrate: `d · E_ν[t²] = E[‖U‖²] = 1`.  Again only linearity of expectation.
  have key : (d : ℝ) * ∫ t, t ^ 2 ∂ν = 1 := by
    have h1 : ∫ ω, ∑ i, (U ω i) ^ 2 ∂μ = 1 := by
      simp_rw [hone]; simp
    rw [integral_finsetSum _ (fun i _ => hmapint i)] at h1
    simp_rw [hcoord] at h1
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h1
    exact h1
  field_simp at key ⊢
  linarith [key]

end Coordinates

