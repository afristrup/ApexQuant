
theorem blockMass_pos {B m : ℕ} (hm : 1 ≤ m) : 0 < blockMass B m := by
  unfold blockMass
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have h1 : 0 < Real.Gamma ((m : ℝ) / 2) := Real.Gamma_pos_of_pos (by positivity)
  have h2 : 0 < Real.Gamma (((B : ℝ) + m) / 2) := Real.Gamma_pos_of_pos (by positivity)
  positivity
