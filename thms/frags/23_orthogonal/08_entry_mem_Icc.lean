
/-! ### Compactness -/

theorem entry_mem_Icc (g : Od d) (i j : Fin d) :
    (g : Matrix (Fin d) (Fin d) ℝ) i j ∈ Set.Icc (-1 : ℝ) 1 := by
  have h : star (g : Matrix (Fin d) (Fin d) ℝ) * g = 1 := Matrix.mem_unitaryGroup_iff'.mp g.2
  have hjj : (star (g : Matrix (Fin d) (Fin d) ℝ) * g) j j = 1 := by rw [h]; simp
  rw [Matrix.mul_apply] at hjj
  simp only [Matrix.star_apply, star_trivial] at hjj
  have hle : (g : Matrix (Fin d) (Fin d) ℝ) i j * (g : Matrix (Fin d) (Fin d) ℝ) i j ≤ 1 := by
    rw [← hjj]
    exact Finset.single_le_sum (f := fun k => (g : Matrix (Fin d) (Fin d) ℝ) k j *
      (g : Matrix (Fin d) (Fin d) ℝ) k j) (fun k _ => mul_self_nonneg _) (Finset.mem_univ i)
  constructor <;> nlinarith [hle]
