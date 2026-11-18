
/-! ## §50  Summation by parts over the cells of a scalar codebook

The sharp distortion-gap identity turns a sum over the cells of a scalar codebook into a sum over
its thresholds.  That last step is pure algebra: no measure, no quantiser and no monotonicity
enter it.  It is isolated here so that it can be checked on its own, and so that the stage that
consumes it does not have to reindex anything.

The indexing convention, fixed once and used by everything below.  A codebook with `K + 1`
codewords has `K + 1` cells, indexed by `Fin (K + 1)`, and `K + 2` cell boundaries, indexed by
`Fin (K + 2)`: the two outer boundaries and the `K` interior thresholds.  Cell `j` runs from
boundary `j.castSucc` to boundary `j.succ`, the outer boundaries are `0` and `Fin.last (K + 1)`,
and the `k`-th threshold, for `k : Fin K`, is boundary `k.succ.castSucc`.

* `sum_fin_by_parts` is Abel summation in the `Fin` indexing just described, in the
  shifted-difference form a cell sum produces.
* `sum_cell_gap_eq_boundary_sub_threshold` is the one place where the midpoint property of the
  thresholds is used, and it is used purely as the algebraic identity
  `c k.castSucc + c k.succ = 2 * x k.succ.castSucc`: that identity makes the codeword-squared and
  the first-moment contributions cancel at every interior threshold, leaving the once-integrated
  term alone.  Nothing is assumed about the boundaries, about monotonicity, or about where the
  three boundary-indexed functions come from.
* `sum_fin_telescope` and `abs_sum_fin_weighted_le` are the elementary facts about codeword
  increments that the bound stage needs.

None of these statements mentions a measure, a codebook structure or a quantiser, and none of them
is specific to the real line beyond using `ℝ` as the scalar field. -/

namespace Threshold

/-- **Abel summation in the cell indexing.**  `f` is the cell weight on `Fin (K + 1)`, `D` the
boundary function on `Fin (K + 2)`, and cell `j` contributes `D j.succ - D j.castSucc`.  The
result is the two outer boundary terms minus a sum over the `K` interior thresholds, the `k`-th
of which is boundary `k.succ.castSucc`.  No hypothesis whatever on `f` or `D`. -/
theorem sum_fin_by_parts {K : ℕ} (f : Fin (K + 1) → ℝ) (D : Fin (K + 2) → ℝ) :
    ∑ j : Fin (K + 1), f j * (D j.succ - D j.castSucc)
      = f (Fin.last K) * D (Fin.last (K + 1)) - f 0 * D 0
        - ∑ k : Fin K, (f k.succ - f k.castSucc) * D k.succ.castSucc := by
  induction K with
  | zero =>
      rw [Fin.sum_univ_one, Finset.univ_eq_empty, Finset.sum_empty]
      norm_num [Fin.last, Fin.succ, Fin.castSucc, Fin.castAdd, Fin.castLE]
      ring
  | succ K ih =>
      rw [Fin.sum_univ_castSucc (f := fun j : Fin (K + 2) => f j * (D j.succ - D j.castSucc)),
        Fin.sum_univ_castSucc
          (f := fun k : Fin (K + 1) => (f k.succ - f k.castSucc) * D k.succ.castSucc)]
      have hih := ih (fun j : Fin (K + 1) => f j.castSucc) (fun i : Fin (K + 2) => D i.castSucc)
      simp only [Fin.succ_castSucc, Fin.succ_last, Fin.castSucc_zero] at hih ⊢
      rw [hih]
      ring

/-- **The cell-to-threshold identity, with the midpoint cancellation.**  Take cell weights `c` on
`Fin (K + 1)`, and three functions on the `K + 2` boundaries: the boundary points `x`, a jump
function `F`, and an accumulated function `G`.  The summand is exactly what a distortion gap
produces on cell `j`: the codeword square against the mass increment `F j.succ - F j.castSucc`,
minus twice the codeword against the first-moment increment, itself written as a boundary term
`x * F` minus the increment of `G`.

If every interior boundary is the midpoint of the two codewords flanking it, then all the `F`
contributions at interior thresholds cancel and only `G` survives there.  This is the entire
content of "the thresholds are midpoints"; it is used nowhere else.  Monotonicity of `c` and any
relation between the `x` are irrelevant here and are not assumed. -/
theorem sum_cell_gap_eq_boundary_sub_threshold {K : ℕ} (c : Fin (K + 1) → ℝ)
    (x F G : Fin (K + 2) → ℝ)
    (hmid : ∀ k : Fin K, c k.castSucc + c k.succ = 2 * x k.succ.castSucc) :
    ∑ j : Fin (K + 1),
        (c j ^ 2 * (F j.succ - F j.castSucc)
          - 2 * c j * (x j.succ * F j.succ - x j.castSucc * F j.castSucc
            - (G j.succ - G j.castSucc)))
      = c (Fin.last K) ^ 2 * F (Fin.last (K + 1)) - c 0 ^ 2 * F 0
        - 2 * (c (Fin.last K) * (x (Fin.last (K + 1)) * F (Fin.last (K + 1))) - c 0 * (x 0 * F 0))
        + 2 * (c (Fin.last K) * G (Fin.last (K + 1)) - c 0 * G 0)
        - 2 * ∑ k : Fin K, (c k.succ - c k.castSucc) * G k.succ.castSucc := by
  have h1 := sum_fin_by_parts (fun j : Fin (K + 1) => c j ^ 2) F
  have h2 := sum_fin_by_parts c (fun i : Fin (K + 2) => x i * F i)
  have h3 := sum_fin_by_parts c G
  have hsplit : ∑ j : Fin (K + 1),
      (c j ^ 2 * (F j.succ - F j.castSucc)
        - 2 * c j * (x j.succ * F j.succ - x j.castSucc * F j.castSucc
          - (G j.succ - G j.castSucc)))
      = (∑ j : Fin (K + 1), c j ^ 2 * (F j.succ - F j.castSucc))
        - 2 * (∑ j : Fin (K + 1), c j * (x j.succ * F j.succ - x j.castSucc * F j.castSucc))
        + 2 * ∑ j : Fin (K + 1), c j * (G j.succ - G j.castSucc) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  have hcancel : (∑ k : Fin K, (c k.succ ^ 2 - c k.castSucc ^ 2) * F k.succ.castSucc)
      = 2 * ∑ k : Fin K,
          (c k.succ - c k.castSucc) * (x k.succ.castSucc * F k.succ.castSucc) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    linear_combination (F k.succ.castSucc * (c k.succ - c k.castSucc)) * hmid k
  linear_combination hsplit + h1 - 2 * h2 + 2 * h3 - hcancel

/-- **The codeword increments telescope.**  A special case of `sum_fin_by_parts` at the constant
boundary function; the spread of the codebook is the sum of its increments. -/
theorem sum_fin_telescope {K : ℕ} (c : Fin (K + 1) → ℝ) :
    ∑ k : Fin K, (c k.succ - c k.castSucc) = c (Fin.last K) - c 0 := by
  have h := sum_fin_by_parts c (fun _ : Fin (K + 2) => (1 : ℝ))
  simp only [sub_self, mul_zero, Finset.sum_const_zero, mul_one] at h
  linarith

/-- Every increment of a monotone codeword sequence is nonnegative. -/
theorem sub_castSucc_nonneg {K : ℕ} {c : Fin (K + 1) → ℝ} (hc : Monotone c) (k : Fin K) :
    0 ≤ c k.succ - c k.castSucc := by
  have hle : k.castSucc ≤ k.succ := by
    rw [Fin.le_def, Fin.val_castSucc, Fin.val_succ]
    omega
  exact sub_nonneg.mpr (hc hle)

/-- **The bound the threshold form is for.**  A sum over thresholds weighted by the codeword
increments of a monotone sequence is bounded by the spread times any uniform bound on the
threshold values.  No sign or size hypothesis on `M` beyond `|G k| ≤ M` pointwise; at `K = 0` both
sides are zero. -/
theorem abs_sum_fin_weighted_le {K : ℕ} {c : Fin (K + 1) → ℝ} (hc : Monotone c)
    (G : Fin K → ℝ) {M : ℝ} (hG : ∀ k, |G k| ≤ M) :
    |∑ k : Fin K, (c k.succ - c k.castSucc) * G k| ≤ (c (Fin.last K) - c 0) * M := by
  calc |∑ k : Fin K, (c k.succ - c k.castSucc) * G k|
      ≤ ∑ k : Fin K, |(c k.succ - c k.castSucc) * G k| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k : Fin K, (c k.succ - c k.castSucc) * M := by
        refine Finset.sum_le_sum fun k _ => ?_
        rw [abs_mul, abs_of_nonneg (sub_castSucc_nonneg hc k)]
        exact mul_le_mul_of_nonneg_left (hG k) (sub_castSucc_nonneg hc k)
    _ = (c (Fin.last K) - c 0) * M := by rw [← Finset.sum_mul, sum_fin_telescope]

end Threshold
