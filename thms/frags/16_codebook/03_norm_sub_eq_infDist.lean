/-- A nearest-codeword quantiser attains the distance to the codebook. -/
theorem norm_sub_eq_infDist (hQ : IsNearest C Q) (x : E) :
    ‖x - Q x‖ = Metric.infDist x C := by
  refine le_antisymm ?_ ?_
  · refine not_lt.mp fun hlt => ?_
    obtain ⟨c, hc, hlt'⟩ := (Metric.infDist_lt_iff ⟨Q x, (hQ x).1⟩).1 hlt
    rw [dist_eq_norm] at hlt'
    exact absurd ((hQ x).2 c hc) (not_le.mpr hlt')
  · rw [← dist_eq_norm]
    exact Metric.infDist_le_dist_of_mem (hQ x).1
