
theorem toSphere_univ_ne_zero [Nontrivial E] :
    volume.toSphere (Set.univ : Set (sphere (0 : E) 1)) ≠ 0 := by
  rw [Measure.toSphere_apply_univ]
  refine mul_ne_zero ?_ (measure_ball_pos volume (0 : E) one_pos).ne'
  exact_mod_cast Module.finrank_pos.ne'
