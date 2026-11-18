
/-! ## §48  Every coordinate of the deployed rotation carries the same law

§36 records the second and fourth moments of a coordinate of `H D_ε x`.  That was enough to
reduce the discrepancy against the Haar rate to the spikiness `∑ⱼ xⱼ⁴`, but it says nothing
about the coordinate's *law*, and it leaves the per-pass rate of the deployed rotation bounded
only by its extreme values over the sphere.

This section closes that gap.  Writing `σᵢⱼ = √d Hᵢⱼ ∈ {±1}`, coordinate `i` is the Rademacher
sum `∑ⱼ σᵢⱼ xⱼ εⱼ / √d`, and the `εⱼ` are i.i.d. symmetric, so `σᵢⱼ εⱼ` has the law of `εⱼ`.
Every coordinate therefore carries the *same* law, that of `∑ⱼ xⱼ εⱼ / √d`, and it does not
depend on `i` or on which flat `H` was used.

The consequence for the paper is that `ρ_H²(x)`, the deployed per-pass rate, is not merely
sandwiched: for a stage that acts coordinatewise it is *exactly* the expectation of the stage's
error against one scalar law, which is what makes the fourth-moment comparison of §37 the whole
story rather than one term of it.

Nothing here is asymptotic and nothing is assumed about `x`. -/

namespace CoordLaw

open Hadamard

variable {n d : ℕ}

/-- Flipping a sign bit negates `sgn`. -/
theorem sgn_not (b : Bool) : sgn (!b) = -sgn b := by cases b <;> simp [sgn]

/-- Flipping a fixed set of bits is an involution of the sign cube. -/
def flipEquiv (τ : Fin n → Bool) : Signs n ≃ Signs n where
  toFun s := fun j => xor (τ j) (s j)
  invFun s := fun j => xor (τ j) (s j)
  left_inv s := by funext j; simp
  right_inv s := by funext j; simp

/-- **The law of a Rademacher sum depends only on `|aⱼ|`.**  Averaging any `f` over the sign
cube gives the same value for weight vectors that agree up to sign, because flipping the bits
where they disagree is a bijection of the cube. -/
theorem signAvg_radSum_congr_abs {a b : Fin n → ℝ} (hab : ∀ j, |a j| = |b j|) (f : ℝ → ℝ) :
    signAvg (fun s => f (radSum a s)) = signAvg (fun s => f (radSum b s)) := by
  set τ : Fin n → Bool := fun j => decide (a j ≠ b j) with hτ
  have key : ∀ s : Signs n, radSum b (flipEquiv τ s) = radSum a s := by
    intro s
    unfold radSum
    refine Finset.sum_congr rfl fun j _ => ?_
    show b j * sgn (xor (τ j) (s j)) = a j * sgn (s j)
    by_cases h : a j = b j
    · have hf : τ j = false := by simp [hτ, h]
      rw [hf, Bool.false_xor, h]
    · have ht : τ j = true := by simp [hτ, h]
      have hneg : a j = -b j := (abs_eq_abs.mp (hab j)).resolve_left h
      rw [ht, Bool.true_xor, sgn_not, hneg]
      ring
  unfold signAvg
  congr 1
  exact Fintype.sum_equiv (flipEquiv τ) (fun s => f (radSum a s)) (fun s => f (radSum b s))
    (fun s => (congrArg f (key s)).symm)

/-- **Every coordinate carries the same law**, namely that of `∑ⱼ xⱼ εⱼ / √d`.  The right-hand
side mentions neither `i` nor `H`. -/
theorem signAvg_coord_eq_signAvg_radSum {H : Od d} (hH : IsFlat H) (x : Ed d) (i : Fin d)
    (f : ℝ → ℝ) :
    signAvg (fun s => f ((hadRot H s • x) i))
      = signAvg (fun s => f (radSum (fun j => x j / Real.sqrt d) s)) := by
  simp_rw [hadRot_apply_eq_radSum]
  refine signAvg_radSum_congr_abs (fun j => ?_) f
  have hd : (0 : ℝ) ≤ (d : ℝ) := Nat.cast_nonneg d
  have habs : |(H : Matrix (Fin d) (Fin d) ℝ) i j| = 1 / Real.sqrt d := by
    have h := hH i j
    have h1 : |(H : Matrix (Fin d) (Fin d) ℝ) i j| = Real.sqrt (1 / (d : ℝ)) := by
      rw [← h, Real.sqrt_sq_eq_abs]
    rw [h1, one_div, Real.sqrt_inv, one_div]
  rw [abs_mul, habs, abs_div, abs_of_nonneg (Real.sqrt_nonneg ((d : ℝ)))]
  ring

/-- **The coordinate law is the same for every flat rotation.**  Two flat `H`, `H'` and any two
coordinates give the same average, so the deployed rate depends on the direction alone. -/
theorem signAvg_coord_indep {H H' : Od d} (hH : IsFlat H) (hH' : IsFlat H') (x : Ed d)
    (i i' : Fin d) (f : ℝ → ℝ) :
    signAvg (fun s => f ((hadRot H s • x) i)) = signAvg (fun s => f ((hadRot H' s • x) i')) := by
  rw [signAvg_coord_eq_signAvg_radSum hH x i f, signAvg_coord_eq_signAvg_radSum hH' x i' f]

end CoordLaw
