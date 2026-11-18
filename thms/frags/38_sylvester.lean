/-! ## §38  The Sylvester construction: flat orthogonal matrices exist in every dimension `2^n`

§36 assumed a flat orthogonal matrix `H` (every entry `±1/√d`).  This section constructs one:
the Walsh–Hadamard matrix on the hypercube `{0,1}^n`, `H(a, b) = (−1)^{⟨a, b⟩}/√(2^n)`
(`walsh`), is orthogonal because its character sum factorises over the coordinates,
`∑_b ∏_j (−1)^{(a_j + a'_j) b_j} = ∏_j ∑_β (−1)^{(a_j + a'_j) β} = 2^n [a = a']`
(`walsh_mul_transpose`), and every entry squares to `1/2^n` (`walsh_apply_sq`).  Reindexing the
hypercube by `Fin (2^n)` gives `exists_isFlat : ∃ H : Od (2^n), IsFlat H`, so the hypothesis of
§36 is satisfiable exactly in the dimensions the deployed pipeline uses. -/

namespace Hadamard

open Orthogonal


/-! ### The Sylvester–Walsh matrix on the hypercube -/

/-- `(−1)^{αβ}` for bits `α, β`. -/
def sgnAnd (α β : Bool) : ℝ := if α && β then -1 else 1

theorem sgnAnd_sq (α β : Bool) : sgnAnd α β ^ 2 = 1 := by
  cases α <;> cases β <;> simp [sgnAnd]

theorem sgnAnd_false_right (α : Bool) : sgnAnd α false = 1 := by cases α <;> simp [sgnAnd]

/-- `∑_β (−1)^{αβ} (−1)^{α'β} = 2 [α = α']`. -/
theorem sum_sgnAnd_mul (α α' : Bool) :
    ∑ β : Bool, sgnAnd α β * sgnAnd α' β = if α = α' then 2 else 0 := by
  cases α <;> cases α' <;> norm_num [sgnAnd]

/-- The Walsh–Hadamard matrix on `{0,1}^n`, with entries `±1/√(2^n)`. -/
noncomputable def walsh (n : ℕ) : Matrix (Fin n → Bool) (Fin n → Bool) ℝ :=
  fun a b => (∏ j, sgnAnd (a j) (b j)) / Real.sqrt (2 ^ n)

theorem walsh_apply_sq (n : ℕ) (a b : Fin n → Bool) : walsh n a b ^ 2 = 1 / 2 ^ n := by
  unfold walsh
  rw [div_pow, Real.sq_sqrt (by positivity), ← Finset.prod_pow]
  simp [sgnAnd_sq]

/-- **Orthogonality of the Walsh matrix**: the character sum factorises over coordinates. -/
theorem walsh_mul_transpose (n : ℕ) : walsh n * Matrix.transpose (walsh n) = 1 := by
  ext a a'
  simp only [Matrix.mul_apply, Matrix.transpose_apply, walsh, Matrix.one_apply]
  have h2 : (0 : ℝ) < Real.sqrt (2 ^ n) := Real.sqrt_pos.mpr (by positivity)
  have hprod : ∀ b : Fin n → Bool,
      (∏ j, sgnAnd (a j) (b j)) / Real.sqrt (2 ^ n) * ((∏ j, sgnAnd (a' j) (b j)) / Real.sqrt (2 ^ n))
        = (1 / 2 ^ n) * ∏ j, (sgnAnd (a j) (b j) * sgnAnd (a' j) (b j)) := by
    intro b
    rw [div_mul_div_comm, Real.mul_self_sqrt (by positivity), ← Finset.prod_mul_distrib,
      div_eq_inv_mul, one_div]
  simp_rw [hprod]
  rw [← Finset.mul_sum]
  have hfac := Fintype.prod_sum (fun j (β : Bool) => sgnAnd (a j) β * sgnAnd (a' j) β)
  beta_reduce at hfac
  rw [← hfac]
  simp_rw [sum_sgnAnd_mul]
  by_cases h : a = a'
  · subst h
    simp only [if_true, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
    field_simp
  · rw [if_neg h]
    obtain ⟨j, hj⟩ : ∃ j, a j ≠ a' j := by
      by_contra hcon
      push_neg at hcon
      exact h (funext hcon)
    rw [Finset.prod_eq_zero (Finset.mem_univ j) (by rw [if_neg hj]), mul_zero]

/-- The reindexing `{0,1}^n ≃ Fin (2^n)`. -/
noncomputable def cubeEquiv (n : ℕ) : (Fin n → Bool) ≃ Fin (2 ^ n) :=
  (Fintype.equivFin _).trans (finCongr (by simp))

/-- The Walsh–Hadamard matrix on `Fin (2^n)`. -/
noncomputable def walshFin (n : ℕ) : Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℝ :=
  Matrix.reindex (cubeEquiv n) (cubeEquiv n) (walsh n)

theorem walshFin_mem (n : ℕ) : walshFin n ∈ Matrix.orthogonalGroup (Fin (2 ^ n)) ℝ := by
  rw [Matrix.mem_orthogonalGroup_iff, walshFin, Matrix.reindex_apply, Matrix.transpose_submatrix,
    Matrix.submatrix_mul_equiv, walsh_mul_transpose, Matrix.submatrix_one_equiv]

theorem walshFin_apply_sq (n : ℕ) (i j : Fin (2 ^ n)) : walshFin n i j ^ 2 = 1 / 2 ^ n := by
  rw [walshFin, Matrix.reindex_apply, Matrix.submatrix_apply, walsh_apply_sq]

/-- **The Sylvester construction**: in every dimension `2^n` there is a flat orthogonal matrix,
so the hypothesis `IsFlat` of §36 is satisfiable exactly where the deployed pipeline needs it. -/
theorem exists_isFlat (n : ℕ) : ∃ H : Od (2 ^ n), IsFlat H := by
  refine ⟨⟨walshFin n, walshFin_mem n⟩, fun i j => ?_⟩
  show walshFin n i j ^ 2 = 1 / ((2 ^ n : ℕ) : ℝ)
  rw [walshFin_apply_sq]; push_cast; rfl

end Hadamard
