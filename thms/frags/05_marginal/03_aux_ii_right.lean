
private theorem aux_ii_right (a : ℝ) (ha : -1 < a) :
    IntervalIntegrable (fun t : ℝ => (1 - t ^ 2) ^ a) volume 0 1 := by
  have h1 : IntervalIntegrable (fun t : ℝ => (1 - t) ^ a) volume 0 1 := by
    have := (intervalIntegral.intervalIntegrable_rpow' (a := 0) (b := 1) ha).comp_sub_left 1
    simpa using this.symm
  have h2 : ContinuousOn (fun t : ℝ => (1 + t) ^ a) (Set.uIcc (0:ℝ) 1) := by
    apply ContinuousOn.rpow_const (by fun_prop)
    intro x hx
    rw [Set.uIcc_of_le (by norm_num : (0:ℝ) ≤ 1)] at hx
    left; nlinarith [hx.1]
  refine (h1.mul_continuousOn h2).congr ?_
  intro x hx
  rw [Set.uIoc_of_le (by norm_num : (0:ℝ) ≤ 1)] at hx
  have hx1 : (0:ℝ) ≤ 1 - x := by linarith [hx.2]
  have hx2 : (0:ℝ) ≤ 1 + x := by linarith [hx.1]
  simp only
  rw [← Real.mul_rpow hx1 hx2]
  ring_nf
