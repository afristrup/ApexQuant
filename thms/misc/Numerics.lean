import Mathlib

/-!
# Decimal claims of the paper as real-number theorems

Every statement is a closed inequality or identity between explicit real numbers. Logarithms are
`Real.logb 2`, and each bound `p / q ≤ logb 2 x` is reduced to the rational inequality
`2 ^ p ≤ x ^ q`.
-/

namespace ApexQuant.Numerics

/-- `2 ^ p ≤ x ^ q` gives `p / q ≤ log2 x`. -/
theorem le_logb_two {x : ℝ} (hx : 0 < x) {p : ℤ} {q : ℕ} (hq : 0 < q)
    (h : (2 : ℝ) ^ p ≤ x ^ q) : (p : ℝ) / q ≤ Real.logb 2 x := by
  have hq' : (0 : ℝ) < q := by exact_mod_cast hq
  rw [div_le_iff₀ hq', mul_comm, ← Real.logb_pow,
    Real.le_logb_iff_rpow_le one_lt_two (pow_pos hx q), Real.rpow_intCast]
  exact h

/-- `x ^ q ≤ 2 ^ p` gives `log2 x ≤ p / q`. -/
theorem logb_two_le {x : ℝ} (hx : 0 < x) {p : ℤ} {q : ℕ} (hq : 0 < q)
    (h : x ^ q ≤ (2 : ℝ) ^ p) : Real.logb 2 x ≤ (p : ℝ) / q := by
  have hq' : (0 : ℝ) < q := by exact_mod_cast hq
  rw [le_div_iff₀ hq', mul_comm, ← Real.logb_pow,
    Real.logb_le_iff_le_rpow one_lt_two (pow_pos hx q), Real.rpow_intCast]
  exact h

/-- Rate gap to the Gaussian value `4^(-2)` of a relative distortion `ρ2`, in bits. -/
noncomputable def G (ρ2 : ℝ) : ℝ := 2 + (1 / 2) * Real.logb 2 ρ2

/-- Rate gap to the sphere floor `0.0615` of a relative distortion `ρ2`, in bits. -/
noncomputable def F (ρ2 : ℝ) : ℝ := (1 / 2) * Real.logb 2 (ρ2 / 0.0615)

/-! ## 1. Table gaps to the Gaussian value -/
/-- `G 0.1171` lies in `[0.45, 0.46]`; true value `0.4529`. -/
theorem G_1171 : 0.45 ≤ G 0.1171 ∧ G 0.1171 ≤ 0.46 := by
  have h1 := le_logb_two (x := 0.1171) (by norm_num) (p := -31) (q := 10) (by norm_num)
    (by norm_num)
  have h2 := logb_two_le (x := 0.1171) (by norm_num) (p := -77) (q := 25) (by norm_num)
    (by norm_num)
  push_cast at h1 h2
  unfold G
  constructor <;> linarith

/-- `G 0.0906` lies in `[0.26, 0.27]`; true value `0.2678`. -/
theorem G_0906 : 0.26 ≤ G 0.0906 ∧ G 0.0906 ≤ 0.27 := by
  have h1 := le_logb_two (x := 0.0906) (by norm_num) (p := -87) (q := 25) (by norm_num)
    (by norm_num)
  have h2 := logb_two_le (x := 0.0906) (by norm_num) (p := -173) (q := 50) (by norm_num)
    (by norm_num)
  push_cast at h1 h2
  unfold G
  constructor <;> linarith

/-- `G 0.0720` lies in `[0.10, 0.11]`; true value `0.1021`. -/
theorem G_0720 : 0.10 ≤ G 0.0720 ∧ G 0.0720 ≤ 0.11 := by
  have h1 := le_logb_two (x := 0.0720) (by norm_num) (p := -19) (q := 5) (by norm_num)
    (by norm_num)
  have h2 := logb_two_le (x := 0.0720) (by norm_num) (p := -189) (q := 50) (by norm_num)
    (by norm_num)
  push_cast at h1 h2
  unfold G
  constructor <;> linarith

/-- `G 0.0702` lies in `[0.08, 0.09]`; true value `0.0838`. -/
theorem G_0702 : 0.08 ≤ G 0.0702 ∧ G 0.0702 ≤ 0.09 := by
  have h1 := le_logb_two (x := 0.0702) (by norm_num) (p := -96) (q := 25) (by norm_num)
    (by norm_num)
  have h2 := logb_two_le (x := 0.0702) (by norm_num) (p := -191) (q := 50) (by norm_num)
    (by norm_num)
  push_cast at h1 h2
  unfold G
  constructor <;> linarith

/-! ## 2. Gaps to the floor -/
/-- `F 0.1171` lies in `[0.46, 0.47]`; true value `0.4645`. -/
theorem F_1171 : 0.46 ≤ F 0.1171 ∧ F 0.1171 ≤ 0.47 := by
  have h1 := le_logb_two (x := 0.1171 / 0.0615) (by norm_num) (p := 23) (q := 25) (by norm_num)
    (by norm_num)
  have h2 := logb_two_le (x := 0.1171 / 0.0615) (by norm_num) (p := 47) (q := 50) (by norm_num)
    (by norm_num)
  push_cast at h1 h2
  unfold F
  constructor <;> linarith

/-- `F 0.0906` lies in `[0.27, 0.28]`; true value `0.2795`. -/
theorem F_0906 : 0.27 ≤ F 0.0906 ∧ F 0.0906 ≤ 0.28 := by
  have h1 := le_logb_two (x := 0.0906 / 0.0615) (by norm_num) (p := 27) (q := 50) (by norm_num)
    (by norm_num)
  have h2 := logb_two_le (x := 0.0906 / 0.0615) (by norm_num) (p := 14) (q := 25) (by norm_num)
    (by norm_num)
  push_cast at h1 h2
  unfold F
  constructor <;> linarith

/-- `F 0.0720` lies in `[0.11, 0.12]`; true value `0.1137`. -/
theorem F_0720 : 0.11 ≤ F 0.0720 ∧ F 0.0720 ≤ 0.12 := by
  have h1 := le_logb_two (x := 0.0720 / 0.0615) (by norm_num) (p := 11) (q := 50) (by norm_num)
    (by norm_num)
  have h2 := logb_two_le (x := 0.0720 / 0.0615) (by norm_num) (p := 6) (q := 25) (by norm_num)
    (by norm_num)
  push_cast at h1 h2
  unfold F
  constructor <;> linarith

/-- `F 0.0702` lies in `[0.09, 0.10]`; true value `0.0954`. -/
theorem F_0702 : 0.09 ≤ F 0.0702 ∧ F 0.0702 ≤ 0.10 := by
  have h1 := le_logb_two (x := 0.0702 / 0.0615) (by norm_num) (p := 9) (q := 50) (by norm_num)
    (by norm_num)
  have h2 := logb_two_le (x := 0.0702 / 0.0615) (by norm_num) (p := 1) (q := 5) (by norm_num)
    (by norm_num)
  push_cast at h1 h2
  unfold F
  constructor <;> linarith

/-! ## 3. Relative distance of the trellis value to the floor -/

/-- `1 - 0.0615 / 0.0720` lies in `[0.14, 0.15]`; true value `0.1458`. -/
theorem floor_fraction :
    (0.14 : ℝ) ≤ 1 - 0.0615 / 0.0720 ∧ (1 : ℝ) - 0.0615 / 0.0720 ≤ 0.15 := by
  constructor <;> norm_num

/-! ## 4. Three-pass totals -/

/-- Three trellis passes lose at most `0.31` bits against the Gaussian value. -/
theorem three_pass_0720 : 3 * G 0.0720 ≤ 0.31 := by
  have h := logb_two_le (x := 0.0720) (by norm_num) (p := -129) (q := 34) (by norm_num)
    (by norm_num)
  push_cast at h
  unfold G
  linarith

/-- Three scalar passes lose between `1.35` and `1.37` bits; true value `1.3587`. -/
theorem three_pass_1171 : 1.35 ≤ 3 * G 0.1171 ∧ 3 * G 0.1171 ≤ 1.37 := by
  have h1 := G_1171.1
  have h2 := logb_two_le (x := 0.1171) (by norm_num) (p := -34) (q := 11) (by norm_num)
    (by norm_num)
  push_cast at h2
  unfold G at h1 ⊢
  constructor <;> linarith

/-! ## 5. Two trellis passes against the four-bit floor -/

/-- Two trellis passes against the four-bit floor `0.00378`: between `0.21` and `0.23` bits; true value `0.2278`. -/
theorem two_pass_vs_four_bit_floor : 0.21 ≤ (1 / 2) * Real.logb 2 ((0.0720 : ℝ) ^ 2 / 0.00378) ∧ (1 / 2) * Real.logb 2 ((0.0720 : ℝ) ^ 2 / 0.00378) ≤ 0.23 := by
  have h1 := le_logb_two (x := (0.0720 : ℝ) ^ 2 / 0.00378) (by norm_num) (p := 21) (q := 50) (by norm_num)
    (by norm_num)
  have h2 := logb_two_le (x := (0.0720 : ℝ) ^ 2 / 0.00378) (by norm_num) (p := 23) (q := 50) (by norm_num)
    (by norm_num)
  push_cast at h1 h2
  constructor <;> linarith

/-! ## 6. Axis obstruction numbers -/

/-- The squared distance from `1` to the level `1.510`. -/
theorem axis_sq : ((1 : ℝ) - 1.510) ^ 2 = 0.2601 := by norm_num

/-- The axis error exceeds the average two-bit distortion. -/
theorem axis_gt : (0.2601 : ℝ) > 0.1171 := by norm_num

/-- `1.510` is the nearer of the two positive two-bit Lloyd-Max levels to `1`. -/
theorem axis_nearest : |(1 : ℝ) - 1.510| < |(1 : ℝ) - 0.4528| := by
  rw [abs_of_neg (by norm_num), abs_of_pos (by norm_num)]
  norm_num

/-! ## 7. The Hadamard bound is loose at `d = 512` -/

/-- The bound evaluates to more than `0.62` (true value `0.8254`), far above `0.1171`. -/
theorem hadamard_bound_loose :
    (0.1171 : ℝ) < 3.02 * (3 * (1.59 / Real.sqrt 512) + Real.sqrt (2 / 512)) := by
  have h1 : Real.sqrt 512 ≤ 23 := by
    rw [Real.sqrt_le_left (by norm_num)]
    norm_num
  have h0 : (0 : ℝ) < Real.sqrt 512 := Real.sqrt_pos.mpr (by norm_num)
  have h2 : (1.59 : ℝ) / 23 ≤ 1.59 / Real.sqrt 512 :=
    div_le_div_of_nonneg_left (by norm_num) h0 h1
  have h3 : (0 : ℝ) ≤ Real.sqrt (2 / 512) := Real.sqrt_nonneg _
  norm_num at h2 ⊢
  linarith

/-! ## 8. Storage arithmetic -/

/-- Two bits plus sixteen side bits per 512 weights. -/
theorem bits_per_weight : (2 : ℝ) + 16 / 512 = 2.03125 := by norm_num

/-- Three passes at the rounded rate. -/
theorem three_pass_bits : (3 : ℝ) * 2.03 = 6.09 := by norm_num

/-- Two plus three passes at the rounded rate `2.03`. -/
theorem five_pass_bits_rounded : (2.03 : ℝ) * 2 + 2.03 * 3 = 10.15 := by norm_num

/-- Two plus three passes at the exact rate `2.03125`; this rounds to the printed `10.16`. -/
theorem five_pass_bits_exact : (4.0625 : ℝ) + 6.09375 = 10.15625 := by norm_num

/-- `70.6e9` weights at `2.03125` bits are between `17.9` and `18` gigabytes; true value
`17.926`. -/
theorem storage_gigabytes :
    (70.6e9 : ℝ) * 2.03125 / 8 / 1e9 ≤ 18 ∧ (17.9 : ℝ) ≤ 70.6e9 * 2.03125 / 8 / 1e9 := by
  constructor <;> norm_num

/-! ## 9. Powers -/

/-- `0.3422 ^ 2` is `0.1171` to four decimals. -/
theorem pow_two_3422 : |(0.3422 : ℝ) ^ 2 - 0.1171| ≤ 0.0001 := by
  rw [abs_le]
  constructor <;> norm_num

/-- `0.3422 ^ 3` is `0.0401` to four decimals. -/
theorem pow_three_3422 : |(0.3422 : ℝ) ^ 3 - 0.0401| ≤ 0.0001 := by
  rw [abs_le]
  constructor <;> norm_num

/-- `0.3422 ^ 4` is `0.0137` to four decimals. -/
theorem pow_four_3422 : |(0.3422 : ℝ) ^ 4 - 0.0137| ≤ 0.0001 := by
  rw [abs_le]
  constructor <;> norm_num

/-- `1 / 0.1171` lies in `[8.5, 8.6]`; true value `8.5397`. -/
theorem inv_1171 : (8.5 : ℝ) ≤ 1 / 0.1171 ∧ (1 : ℝ) / 0.1171 ≤ 8.6 := by
  constructor <;> norm_num

/-- `1 / 0.0720` lies in `[13.8, 13.9]`; true value `13.8889`. -/
theorem inv_0720 : (13.8 : ℝ) ≤ 1 / 0.0720 ∧ (1 : ℝ) / 0.0720 ≤ 13.9 := by
  constructor <;> norm_num

/-! ## 10. Rounded scales -/

/-- The scale rounding term is below `0.004`; true value `0.003506`. -/
theorem rounded_scale :
    2 * (2 : ℝ) ^ (-(11 : ℤ)) * Real.sqrt (1 - 0.0720) / Real.sqrt 0.0720 < 0.004 := by
  have h1 : Real.sqrt (1 - 0.0720) ≤ 0.97 := by
    rw [Real.sqrt_le_left (by norm_num)]
    norm_num
  have h2 : (0.26 : ℝ) ≤ Real.sqrt 0.0720 := Real.le_sqrt_of_sq_le (by norm_num)
  have e : (2 : ℝ) * 2 ^ (-(11 : ℤ)) = 1 / 1024 := by norm_num
  rw [div_lt_iff₀ (by linarith), e]
  linarith

end ApexQuant.Numerics
