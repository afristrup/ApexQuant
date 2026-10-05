
/-- **Lemma 1, scalar form**: the normalised block density at `U = ℝ`, `V` of dimension `d − 1`,
is the paper's coordinate law `sphericalCoordLaw d`. -/
theorem blockLaw_real (m : ℕ) (hm : 1 ≤ m) :
    blockLaw ℝ m = sphericalCoordLaw (m + 1) := by
  have hprob := isProbabilityMeasure_sphericalCoordLaw (d := m + 1) (by omega)
  have h1 := hprob.measure_univ
  rw [sphericalCoordLaw_eq_smul_blockMeasure m hm] at h1 ⊢
  rw [Measure.smul_apply, smul_eq_mul] at h1
  rw [blockLaw, ENNReal.eq_inv_of_mul_eq_one_left h1]

end Normalise

section Naturality
open scoped Pointwise

variable {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E]
  [NormedAddCommGroup F] [InnerProductSpace ℝ F] [MeasurableSpace F] [BorelSpace F]
  [FiniteDimensional ℝ F]
