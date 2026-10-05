
private theorem aux_ii (a : ℝ) (ha : -1 < a) :
    IntervalIntegrable (fun t : ℝ => (1 - t ^ 2) ^ a) volume (-1) 1 := by
  have hR := aux_ii_right a ha
  have hL : IntervalIntegrable (fun t : ℝ => (1 - t ^ 2) ^ a) volume (-1) 0 := by
    have := (IntervalIntegrable.iff_comp_neg (f := fun t : ℝ => (1 - t ^ 2) ^ a)
      (a := 0) (b := 1) (by finiteness)).mp hR
    simpa using this.symm
  exact hL.trans hR
