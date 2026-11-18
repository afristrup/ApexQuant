/-! ## §40  Theorem 1 for the deployed rotation, at the level of characteristic functions

Theorem 1 is false for `H D_ε`: the law of the direction is supported on `2^d` points.  What
survives is a quantitative Gaussian approximation of each coordinate, exact in its dependence
on the input.  The coordinate `(H D_ε v)ᵢ = ∑ⱼ aⱼ εⱼ` with `aⱼ = Hᵢⱼ vⱼ` is a Rademacher sum: its
odd moments vanish (`signAvg_radSum_odd_pow`) and its characteristic function factorises,
`E cos(t ∑ aⱼ εⱼ) = ∏ⱼ cos(t aⱼ)` (`signAvg_cos_radSum`).  Since `|cos x − e^{−x²/2}| ≤ x⁴/3`
for `|x| ≤ 1` (`abs_cos_sub_exp_le`) and products of numbers in `[−1, 1]` telescope
(`abs_prod_sub_prod_le`), the characteristic function of a coordinate of `H D_ε v` is within
`t⁴ ∑ⱼ vⱼ⁴/(3d²)` of the Gaussian `e^{−t²‖v‖²/(2d)}` for `|t| ≤ √d`
(`abs_signAvg_cos_coord_sub_exp_le`).  This is the same spikiness `∑ⱼ vⱼ⁴` that governs the
fourth-moment gap of §37: for a spread direction it is `O(1/d)` and the coordinate law is
Gaussian to order `t⁴/d³`; for a coordinate vector it is `1` and the bound is `t⁴/(3d²)`.
Passing from characteristic functions to Kolmogorov distance is Esseen's smoothing inequality,
which is not formalised here; and `f_d` itself is compared to the Gaussian only through its
moments (§5, §37).  The spikiness of real residual directions is measured in
`experiments/hadamard_spikiness.py`. -/

namespace Hadamard

open Orthogonal


/-! ### Symmetry: odd moments vanish -/

/-- Flipping every sign. -/
def flipAll {n : ℕ} : Signs n ≃ Signs n :=
  Function.Involutive.toPerm (fun s : Signs n => fun j => !s j) fun s => by
    funext j; simp

theorem radSum_flipAll {n : ℕ} (a : Fin n → ℝ) (s : Signs n) :
    radSum a (flipAll s) = -radSum a s := by
  unfold radSum
  rw [← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  show a j * sgn (!s j) = -(a j * sgn (s j))
  cases s j <;> simp [sgn]

theorem signAvg_odd {n : ℕ} (a : Fin n → ℝ) {F : ℝ → ℝ} (hF : ∀ x, F (-x) = -F x) :
    signAvg (fun s => F (radSum a s)) = 0 := by
  have h : ∑ s : Signs n, F (radSum a s) = -∑ s : Signs n, F (radSum a s) := by
    conv_lhs => rw [← Fintype.sum_equiv flipAll (fun s => F (radSum a (flipAll s)))
      (fun s => F (radSum a s)) (fun s => rfl)]
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun s _ => ?_
    rw [radSum_flipAll, hF]
  unfold signAvg
  have : ∑ s : Signs n, F (radSum a s) = 0 := by linarith
  rw [this, zero_div]

/-- **Odd moments of a Rademacher sum vanish.** -/
theorem signAvg_radSum_odd_pow {n : ℕ} (a : Fin n → ℝ) (k : ℕ) :
    signAvg (fun s => radSum a s ^ (2 * k + 1)) = 0 :=
  signAvg_odd a (F := fun x => x ^ (2 * k + 1)) fun x => by
    show (-x) ^ (2 * k + 1) = -x ^ (2 * k + 1)
    exact Odd.neg_pow ⟨k, rfl⟩ x

/-! ### The characteristic function factorises -/

/-- **The characteristic function of a Rademacher sum**: `E cos(t ∑ aⱼ εⱼ) = ∏ⱼ cos(t aⱼ)`. -/
theorem signAvg_cos_radSum {n : ℕ} (a : Fin n → ℝ) (t : ℝ) :
    signAvg (fun s => Real.cos (t * radSum a s)) = ∏ j, Real.cos (t * a j) := by
  induction n with
  | zero => simp [signAvg, radSum]
  | succ n ih =>
      unfold signAvg at ih ⊢
      rw [sum_signs_succ, Fin.prod_univ_succ]
      simp only [radSum_cons]
      rw [Fintype.sum_bool]
      simp only [sgn_true, sgn_false]
      have hcos : ∀ S : ℝ, Real.cos (t * (a 0 * 1 + S)) + Real.cos (t * (a 0 * -1 + S))
          = 2 * Real.cos (t * a 0) * Real.cos (t * S) := by
        intro S
        rw [show t * (a 0 * 1 + S) = t * a 0 + t * S by ring,
          show t * (a 0 * -1 + S) = t * S - t * a 0 by ring, Real.cos_add, Real.cos_sub]
        ring
      rw [← Finset.sum_add_distrib]
      simp_rw [hcos]
      have hprod : ∏ i : Fin n, Real.cos (t * a i.succ) = ∏ j, Real.cos (t * Fin.tail a j) := rfl
      rw [← Finset.mul_sum, hprod, ← ih (Fin.tail a), pow_succ]
      field_simp
      try ring

/-! ### Comparison with the Gaussian characteristic function -/

/-- `|cos x − e^{−x²/2}| ≤ x⁴/3` for `|x| ≤ 1`. -/
theorem abs_cos_sub_exp_le {x : ℝ} (hx : |x| ≤ 1) :
    |Real.cos x - Real.exp (-(x ^ 2) / 2)| ≤ x ^ 4 / 3 := by
  have h1 := Real.cos_bound hx
  have h2 : |Real.exp (-(x ^ 2) / 2) - 1 - (-(x ^ 2) / 2)| ≤ (-(x ^ 2) / 2) ^ 2 := by
    refine Real.abs_exp_sub_one_sub_id_le ?_
    rw [abs_le]
    have := sq_nonneg x
    have hx2 : x ^ 2 ≤ 1 := by
      have := abs_le.mp hx; nlinarith [abs_nonneg x, sq_abs x]
    constructor <;> nlinarith
  have hx4 : |x| ^ 4 = x ^ 4 := by
    rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, pow_mul, sq_abs]
  rw [hx4] at h1
  calc |Real.cos x - Real.exp (-(x ^ 2) / 2)|
      = |(Real.cos x - (1 - x ^ 2 / 2)) - (Real.exp (-(x ^ 2) / 2) - 1 - (-(x ^ 2) / 2))| := by
        congr 1; ring
    _ ≤ |Real.cos x - (1 - x ^ 2 / 2)| + |Real.exp (-(x ^ 2) / 2) - 1 - (-(x ^ 2) / 2)| :=
        abs_sub _ _
    _ ≤ x ^ 4 * (5 / 96) + (-(x ^ 2) / 2) ^ 2 := add_le_add h1 h2
    _ ≤ x ^ 4 / 3 := by nlinarith [pow_nonneg (sq_nonneg x) 2]

/-- A telescoping bound: products of numbers in `[−1, 1]` differ by at most the sum of the
factor differences. -/
theorem abs_prod_sub_prod_le {ι : Type*} (s : Finset ι) (p q : ι → ℝ) (hp : ∀ i ∈ s, |p i| ≤ 1)
    (hq : ∀ i ∈ s, |q i| ≤ 1) :
    |∏ i ∈ s, p i - ∏ i ∈ s, q i| ≤ ∑ i ∈ s, |p i - q i| := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
      rw [Finset.prod_insert ha, Finset.prod_insert ha, Finset.sum_insert ha]
      have hp' := fun i hi => hp i (Finset.mem_insert_of_mem hi)
      have hq' := fun i hi => hq i (Finset.mem_insert_of_mem hi)
      have hP : |∏ i ∈ s, p i| ≤ 1 := by
        rw [Finset.abs_prod]
        exact Finset.prod_le_one (fun i _ => abs_nonneg _) fun i hi => hp' i hi
      have hqa : |q a| ≤ 1 := hq a (Finset.mem_insert_self a s)
      calc |p a * ∏ i ∈ s, p i - q a * ∏ i ∈ s, q i|
          = |(p a - q a) * ∏ i ∈ s, p i + q a * (∏ i ∈ s, p i - ∏ i ∈ s, q i)| := by
            congr 1; ring
        _ ≤ |(p a - q a) * ∏ i ∈ s, p i| + |q a * (∏ i ∈ s, p i - ∏ i ∈ s, q i)| :=
            abs_add_le _ _
        _ = |p a - q a| * |∏ i ∈ s, p i| + |q a| * |∏ i ∈ s, p i - ∏ i ∈ s, q i| := by
            rw [abs_mul, abs_mul]
        _ ≤ |p a - q a| * 1 + 1 * ∑ i ∈ s, |p i - q i| :=
            add_le_add (mul_le_mul_of_nonneg_left hP (abs_nonneg _))
              (mul_le_mul hqa (ih hp' hq') (abs_nonneg _) zero_le_one)
        _ = |p a - q a| + ∑ i ∈ s, |p i - q i| := by ring

/-- **The characteristic function of a Rademacher sum is close to the Gaussian one**: for
`|t aⱼ| ≤ 1`, `|∏ cos(t aⱼ) − e^{−t² ∑ aⱼ²/2}| ≤ t⁴ ∑ aⱼ⁴ / 3`. -/
theorem abs_prod_cos_sub_exp_le {n : ℕ} (a : Fin n → ℝ) {t : ℝ} (ht : ∀ j, |t * a j| ≤ 1) :
    |∏ j, Real.cos (t * a j) - Real.exp (-(t ^ 2 * ∑ j, a j ^ 2) / 2)|
      ≤ t ^ 4 * (∑ j, a j ^ 4) / 3 := by
  have he : Real.exp (-(t ^ 2 * ∑ j, a j ^ 2) / 2) = ∏ j, Real.exp (-((t * a j) ^ 2) / 2) := by
    rw [← Real.exp_sum]
    congr 1
    rw [Finset.mul_sum, ← Finset.sum_neg_distrib, Finset.sum_div]
    refine Finset.sum_congr rfl fun j _ => ?_; ring
  rw [he]
  refine (abs_prod_sub_prod_le Finset.univ _ _ (fun j _ => Real.abs_cos_le_one _)
    (fun j _ => by
      rw [abs_of_pos (Real.exp_pos _)]
      exact Real.exp_le_one_iff.mpr (by nlinarith [sq_nonneg (t * a j)]))).trans ?_
  calc ∑ j, |Real.cos (t * a j) - Real.exp (-((t * a j) ^ 2) / 2)|
      ≤ ∑ j, (t * a j) ^ 4 / 3 := Finset.sum_le_sum fun j _ => abs_cos_sub_exp_le (ht j)
    _ = t ^ 4 * (∑ j, a j ^ 4) / 3 := by
        rw [Finset.mul_sum, Finset.sum_div]
        refine Finset.sum_congr rfl fun j _ => ?_; ring

/-! ### Application to the deployed rotation -/

variable {d : ℕ}

theorem abs_entry_of_isFlat {H : Od d} (hH : IsFlat H) (i j : Fin d) :
    |(H : Matrix (Fin d) (Fin d) ℝ) i j| = 1 / Real.sqrt d := by
  have h := hH i j
  have hd : (0 : ℝ) < d := by
    rcases Nat.eq_zero_or_pos d with h0 | h0
    · subst h0; exact Fin.elim0 i
    · exact_mod_cast h0
  rw [← sq_abs] at h
  have : |(H : Matrix (Fin d) (Fin d) ℝ) i j| = Real.sqrt (1 / d) := by
    rw [← Real.sqrt_sq (abs_nonneg _), h]
  rw [this, Real.sqrt_div' _ hd.le, Real.sqrt_one]

/-- **Theorem 1 for the deployed rotation, at the level of characteristic functions.**  For a
direction `v` with `‖v‖ ≤ 1` and `|t| ≤ √d`, the characteristic function of a coordinate of
`H D_ε v` is within `t⁴ ∑ⱼ vⱼ⁴/(3d²)` of the Gaussian characteristic function
`e^{−t²‖v‖²/(2d)}`: the coordinate law is Gaussian to the extent that the direction is spread,
and the deviation is governed by the same spikiness `∑ⱼ vⱼ⁴` as the fourth-moment gap. -/
theorem abs_signAvg_cos_coord_sub_exp_le {H : Od d} (hH : IsFlat H) {v : Ed d} (hv : ‖v‖ ≤ 1)
    (i : Fin d) {t : ℝ} (ht : |t| ≤ Real.sqrt d) :
    |signAvg (fun s => Real.cos (t * (hadRot H s • v) i))
        - Real.exp (-(t ^ 2 * ‖v‖ ^ 2 / d) / 2)|
      ≤ t ^ 4 * (∑ j, v j ^ 4) / (3 * (d : ℝ) ^ 2) := by
  have hd : (0 : ℝ) < d := by
    rcases Nat.eq_zero_or_pos d with h0 | h0
    · subst h0; exact Fin.elim0 i
    · exact_mod_cast h0
  set a : Fin d → ℝ := fun j => (H : Matrix (Fin d) (Fin d) ℝ) i j * v j with ha
  have hcoord : ∀ s, (hadRot H s • v) i = radSum a s := fun s => hadRot_apply_eq_radSum H s v i
  simp_rw [hcoord, signAvg_cos_radSum]
  have h2 : ∑ j, a j ^ 2 = ‖v‖ ^ 2 / d := by
    rw [norm_sq_eq_sum_sq]; simp_rw [ha, mul_pow, hH i]; rw [← Finset.mul_sum]; try ring
  have h4 : ∑ j, a j ^ 4 = (∑ j, v j ^ 4) / (d : ℝ) ^ 2 := by
    have : ∀ j, a j ^ 4 = (1 / (d : ℝ)) ^ 2 * v j ^ 4 := fun j => by
      rw [ha]; simp only
      rw [mul_pow, show (4 : ℕ) = 2 * 2 from rfl, pow_mul, hH i]
      try ring
    simp_rw [this]; rw [← Finset.mul_sum]; try ring
  have hta : ∀ j, |t * a j| ≤ 1 := by
    intro j
    rw [abs_mul, ha]; simp only
    rw [abs_mul, abs_entry_of_isFlat hH]
    have hvj : |v j| ≤ ‖v‖ := by
      have := norm_sq_eq_sum_sq v
      have h1 : v j ^ 2 ≤ ‖v‖ ^ 2 := by
        rw [this]
        exact Finset.single_le_sum (f := fun k => v k ^ 2) (fun k _ => sq_nonneg _)
          (Finset.mem_univ j)
      exact abs_le_of_sq_le_sq' h1 (norm_nonneg _) |>.2 |> fun h => (abs_le.mpr
        ⟨(abs_le_of_sq_le_sq' h1 (norm_nonneg _)).1, h⟩)
    have hs : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd
    calc |t| * (1 / Real.sqrt d * |v j|) ≤ Real.sqrt d * (1 / Real.sqrt d * 1) :=
          mul_le_mul ht (mul_le_mul_of_nonneg_left (hvj.trans hv) (by positivity))
            (by positivity) hs.le
      _ = 1 := by field_simp
  have := abs_prod_cos_sub_exp_le a hta
  rw [h2, h4] at this
  have e : -(t ^ 2 * (‖v‖ ^ 2 / d)) / 2 = -(t ^ 2 * ‖v‖ ^ 2 / d) / 2 := by ring
  rw [e] at this
  refine this.trans (le_of_eq ?_)
  try ring

end Hadamard
