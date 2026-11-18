/-! ## §31  The Dunkl–Williams inequality and the sharp slope of `h_c`

§29 bounded the derivative of `h_c(z) = ‖z/‖z‖ − c‖²` by `4‖z/‖z‖ − c‖/‖z‖`, twice the true
value, because it went through a two-point Lipschitz estimate with a symmetric slope.  This
section proves the sharp bound `2‖z/‖z‖ − c‖/‖z‖` (`norm_fderiv_hc_le_sharp`) from the
Dunkl–Williams inequality in an inner product space, `‖z/‖z‖ − w/‖w‖‖ ≤ 2‖z − w‖/(‖z‖ + ‖w‖)`
(`norm_dir_sub_dir_le_dunkl`), whose right side has the exact slope `1/‖z‖` at `w = z`. -/

section DunklWilliams

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem norm_dir_of_ne_zero {z : E} (hz : z ≠ 0) : ‖dir z‖ = 1 := by
  rw [dir, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr hz)]

theorem inner_dir_dir (z w : E) : ⟪dir z, dir w⟫_ℝ = ‖z‖⁻¹ * ‖w‖⁻¹ * ⟪z, w⟫_ℝ := by
  simp only [dir, real_inner_smul_left, real_inner_smul_right]; ring

/-- **The Dunkl–Williams inequality** in an inner product space:
`‖z/‖z‖ − w/‖w‖‖ ≤ 2‖z − w‖/(‖z‖ + ‖w‖)`. -/
theorem norm_dir_sub_dir_le_dunkl {z w : E} (hz : z ≠ 0) (hw : w ≠ 0) :
    ‖dir z - dir w‖ ≤ 2 * ‖z - w‖ / (‖z‖ + ‖w‖) := by
  have ha : 0 < ‖z‖ := norm_pos_iff.mpr hz
  have hb : 0 < ‖w‖ := norm_pos_iff.mpr hw
  set c := ⟪dir z, dir w⟫_ℝ with hc
  have hc1 : |c| ≤ 1 := by
    have := abs_real_inner_le_norm (dir z) (dir w)
    rwa [norm_dir_of_ne_zero hz, norm_dir_of_ne_zero hw, one_mul] at this
  have hl : ‖dir z - dir w‖ ^ 2 = 2 - 2 * c := by
    rw [norm_sub_sq_real, norm_dir_of_ne_zero hz, norm_dir_of_ne_zero hw]; ring
  have hr : ‖z - w‖ ^ 2 = ‖z‖ ^ 2 + ‖w‖ ^ 2 - 2 * (‖z‖ * ‖w‖ * c) := by
    rw [norm_sub_sq_real, hc, inner_dir_dir]; field_simp; ring
  have hsq : ‖dir z - dir w‖ ^ 2 ≤ (2 * ‖z - w‖ / (‖z‖ + ‖w‖)) ^ 2 := by
    rw [div_pow, mul_pow, hr, hl, le_div_iff₀ (by positivity)]
    have hc' := (abs_le.mp hc1).1
    nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ 1 + c) (sq_nonneg (‖z‖ - ‖w‖))]
  exact (pow_le_pow_iff_left₀ (norm_nonneg _) (by positivity) two_ne_zero).mp hsq

/-- `‖p‖² − ‖q‖² = ⟪p − q, p + q⟫`. -/
theorem norm_sq_sub_norm_sq (p q : E) : ‖p‖ ^ 2 - ‖q‖ ^ 2 = ⟪p - q, p + q⟫_ℝ := by
  rw [← real_inner_self_eq_norm_sq, ← real_inner_self_eq_norm_sq, inner_sub_left, inner_add_right,
    inner_add_right, real_inner_comm p q]; ring

/-- **The sharp two-point bound** for `h_c`: the slope between `z` and `w` is at most
`2(‖dir z − c‖ + ‖dir w − c‖)/(‖z‖ + ‖w‖)`. -/
theorem abs_hc_sub_le_sharp (c : E) {z w : E} (hz : z ≠ 0) (hw : w ≠ 0) :
    |hc c z - hc c w| ≤ ‖z - w‖ * (2 * (‖dir z - c‖ + ‖dir w - c‖) / (‖z‖ + ‖w‖)) := by
  have h1 : hc c z - hc c w = ⟪dir z - dir w, (dir z - c) + (dir w - c)⟫_ℝ := by
    unfold hc
    rw [norm_sq_sub_norm_sq]; congr 1; abel
  rw [h1]
  calc |⟪dir z - dir w, (dir z - c) + (dir w - c)⟫_ℝ|
      ≤ ‖dir z - dir w‖ * ‖(dir z - c) + (dir w - c)‖ := abs_real_inner_le_norm _ _
    _ ≤ (2 * ‖z - w‖ / (‖z‖ + ‖w‖)) * (‖dir z - c‖ + ‖dir w - c‖) :=
        mul_le_mul (norm_dir_sub_dir_le_dunkl hz hw) (norm_add_le _ _) (norm_nonneg _)
          (by positivity)
    _ = _ := by ring

/-- **The sharp derivative bound** `‖h_c'(z)‖ ≤ 2‖dir z − c‖/‖z‖`. -/
theorem norm_fderiv_hc_le_sharp [FiniteDimensional ℝ E] (c : E) {z : E} (hz : z ≠ 0) :
    ‖fderiv ℝ (hc c) z‖ ≤ 2 * ‖dir z - c‖ / ‖z‖ := by
  have hz' : 0 < ‖z‖ := norm_pos_iff.mpr hz
  refine le_of_forall_pos_le_add fun ε hε => ?_
  have hK : ContinuousAt (fun w : E => 2 * (‖dir w - c‖ + ‖dir z - c‖) / (‖w‖ + ‖z‖)) z := by
    have hd : ContinuousAt (dir : E → E) z := (contDiffAt_dir hz).continuousAt
    exact (continuousAt_const.mul ((hd.sub continuousAt_const).norm.add continuousAt_const)).div
      (continuous_norm.continuousAt.add continuousAt_const) (by positivity)
  have hKz : (fun w : E => 2 * (‖dir w - c‖ + ‖dir z - c‖) / (‖w‖ + ‖z‖)) z
      = 2 * ‖dir z - c‖ / ‖z‖ := by
    simp only; field_simp
  have hev : ∀ᶠ w in 𝓝 z, ‖hc c w - hc c z‖ ≤ (2 * ‖dir z - c‖ / ‖z‖ + ε) * ‖w - z‖ := by
    have hlt : ∀ᶠ w in 𝓝 z,
        2 * (‖dir w - c‖ + ‖dir z - c‖) / (‖w‖ + ‖z‖) < 2 * ‖dir z - c‖ / ‖z‖ + ε :=
      hK (Iio_mem_nhds (by rw [hKz]; linarith))
    filter_upwards [hlt, isOpen_ne.mem_nhds hz] with w hw hw0
    rw [Real.norm_eq_abs]
    calc |hc c w - hc c z| ≤ ‖w - z‖ * (2 * (‖dir w - c‖ + ‖dir z - c‖) / (‖w‖ + ‖z‖)) :=
          abs_hc_sub_le_sharp c hw0 hz
      _ ≤ ‖w - z‖ * (2 * ‖dir z - c‖ / ‖z‖ + ε) :=
          mul_le_mul_of_nonneg_left hw.le (norm_nonneg _)
      _ = _ := mul_comm _ _
  exact (differentiableAt_hc c hz).hasFDerivAt.le_of_lip' (by positivity) hev

end DunklWilliams
