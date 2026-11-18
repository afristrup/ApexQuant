/-!
# §44  The damped characteristic-function bound and the Kolmogorov–Smirnov rate

§40 bounds the characteristic-function gap of a coordinate of `H D_ε v` by `t⁴ ∑ⱼ vⱼ⁴/(3d²)`
for `|t| ≤ √d`; that bound has no Gaussian damping and, fed into the band-limited Esseen
inequality of §43, gives a constant.  This section proves the damped form and closes the
Kolmogorov–Smirnov corollary.

* `Hadamard.abs_cos_le_exp`: `|cos x| ≤ e^{−x²/4}` for `|x| ≤ 1`.
* `Hadamard.abs_prod_sub_prod_le_weighted`: the telescoping bound with weights,
  `|∏ pᵢ − ∏ qᵢ| ≤ (∏ wᵢ) ∑ᵢ |pᵢ − qᵢ|/wᵢ` when `|pᵢ|, |qᵢ| ≤ wᵢ`.
* `Hadamard.abs_prod_cos_sub_exp_le_damped`: for `|t aⱼ| ≤ 1`,
  `|∏ cos(t aⱼ) − e^{−t²∑aⱼ²/2}| ≤ (t⁴ ∑ aⱼ⁴/2) e^{−t²∑aⱼ²/4}`, using `e^{1/4} ≤ 3/2`.
* `Hadamard.norm_charFun_coordLaw_sub_gaussian_le_damped`: the same for the coordinate law of
  `H D_ε v` against `N(0, ‖v‖²/d)`, on the full range `|t vⱼ| ≤ √d`.
* The Gaussian distribution function `N(0, v)` is `(√(2πv))⁻¹`-Lipschitz
  (`cdfR_gaussianReal_lipschitz`); `∫ |ξ|³ e^{−bξ²} dξ = 1/b²` by the fundamental theorem of
  calculus on `(0, ∞)` and the reflection `∫ f|x| = 2∫_0^∞ f` (`integral_abs_pow_three_mul_exp`).
* `Hadamard.abs_cdfR_coordLaw_sub_gaussian_le`: **the Kolmogorov–Smirnov bound.**  For `H` flat,
  `v ≠ 0` with `|vⱼ| ≤ M`, every `a > 0` and every `x₀`,
  `|F(x₀) − Φ(x₀)| (1 − 8M/(π√d a)) ≤ 16 ∑ⱼ vⱼ⁴/(π‖v‖⁴) + 2a/√(2π‖v‖²/d)`.
  For a spread unit direction (`M = 1/√d`, `∑ vⱼ⁴ = 1/d`) and `a = 16/(πd)` the right side is
  `16/(πd) + 32/(π√(2π) √d)`, so the Kolmogorov–Smirnov distance is at most
  `8.2/√d + 10.2/d`.  The constants are explicit and crude; the rate is the classical one.
-/

open Filter
open scoped Topology FourierTransform
open Orthogonal
namespace Hadamard

/-- `|cos x| ≤ e^{−x²/4}` for `|x| ≤ 1`: the cosine is Gaussian-damped on the unit range. -/
theorem abs_cos_le_exp {x : ℝ} (hx : |x| ≤ 1) : |Real.cos x| ≤ Real.exp (-(x ^ 2) / 4) := by
  have h1 := Real.cos_bound hx
  have hx2 : x ^ 2 ≤ 1 := by
    rw [← sq_abs]; nlinarith [abs_nonneg x]
  have hx4 : |x| ^ 4 = x ^ 4 := by
    rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, pow_mul, sq_abs]
  rw [hx4, abs_le] at h1
  have hexp := Real.add_one_le_exp (-(x ^ 2) / 4)
  have hpos : 0 ≤ Real.cos x := by nlinarith [pow_nonneg (sq_nonneg x) 2]
  rw [abs_of_nonneg hpos]
  nlinarith [pow_nonneg (sq_nonneg x) 2, sq_nonneg x]

/-- A weighted telescoping bound: if `|pᵢ|, |qᵢ| ≤ wᵢ` with `wᵢ > 0`, then
`|∏ pᵢ − ∏ qᵢ| ≤ (∏ wᵢ) ∑ᵢ |pᵢ − qᵢ|/wᵢ`. -/
theorem abs_prod_sub_prod_le_weighted {ι : Type*} (s : Finset ι) (p q w : ι → ℝ)
    (hw : ∀ i ∈ s, 0 < w i) (hp : ∀ i ∈ s, |p i| ≤ w i) (hq : ∀ i ∈ s, |q i| ≤ w i) :
    |∏ i ∈ s, p i - ∏ i ∈ s, q i| ≤ (∏ i ∈ s, w i) * ∑ i ∈ s, |p i - q i| / w i := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
      rw [Finset.prod_insert ha, Finset.prod_insert ha, Finset.prod_insert ha, Finset.sum_insert ha]
      have hw' := fun i hi => hw i (Finset.mem_insert_of_mem hi)
      have hp' := fun i hi => hp i (Finset.mem_insert_of_mem hi)
      have hq' := fun i hi => hq i (Finset.mem_insert_of_mem hi)
      have hwa : 0 < w a := hw a (Finset.mem_insert_self a s)
      have hqa : |q a| ≤ w a := hq a (Finset.mem_insert_self a s)
      have hW : 0 ≤ ∏ i ∈ s, w i := Finset.prod_nonneg fun i hi => (hw' i hi).le
      have hP : |∏ i ∈ s, p i| ≤ ∏ i ∈ s, w i := by
        rw [Finset.abs_prod]
        exact Finset.prod_le_prod (fun i _ => abs_nonneg _) fun i hi => hp' i hi
      have hS : 0 ≤ ∑ i ∈ s, |p i - q i| / w i :=
        Finset.sum_nonneg fun i hi => div_nonneg (abs_nonneg _) (hw' i hi).le
      calc |p a * ∏ i ∈ s, p i - q a * ∏ i ∈ s, q i|
          = |(p a - q a) * ∏ i ∈ s, p i + q a * (∏ i ∈ s, p i - ∏ i ∈ s, q i)| := by
            congr 1; ring
        _ ≤ |(p a - q a) * ∏ i ∈ s, p i| + |q a * (∏ i ∈ s, p i - ∏ i ∈ s, q i)| :=
            abs_add_le _ _
        _ = |p a - q a| * |∏ i ∈ s, p i| + |q a| * |∏ i ∈ s, p i - ∏ i ∈ s, q i| := by
            rw [abs_mul, abs_mul]
        _ ≤ |p a - q a| * ∏ i ∈ s, w i + w a * ((∏ i ∈ s, w i) * ∑ i ∈ s, |p i - q i| / w i) :=
            add_le_add (mul_le_mul_of_nonneg_left hP (abs_nonneg _))
              (mul_le_mul hqa (ih hw' hp' hq') (abs_nonneg _) hwa.le)
        _ = w a * (∏ i ∈ s, w i) * (|p a - q a| / w a + ∑ i ∈ s, |p i - q i| / w i) := by
            have hw0 : w a ≠ 0 := hwa.ne'
            field_simp
            try ring

theorem exp_quarter_le : Real.exp (1 / 4) ≤ 3 / 2 := by
  have h := Real.abs_exp_sub_one_sub_id_le (x := 1 / 4) (by rw [abs_le]; constructor <;> norm_num)
  rw [abs_le] at h
  nlinarith [h.2]

/-- **The Gaussian-damped characteristic-function bound**: for `|t aⱼ| ≤ 1`,
`|∏ cos(t aⱼ) − e^{−t² ∑ aⱼ²/2}| ≤ (t⁴ ∑ aⱼ⁴ / 2) e^{−t² ∑ aⱼ²/4}`. -/
theorem abs_prod_cos_sub_exp_le_damped {n : ℕ} (a : Fin n → ℝ) {t : ℝ}
    (ht : ∀ j, |t * a j| ≤ 1) :
    |∏ j, Real.cos (t * a j) - Real.exp (-(t ^ 2 * ∑ j, a j ^ 2) / 2)|
      ≤ t ^ 4 * (∑ j, a j ^ 4) / 2 * Real.exp (-(t ^ 2 * ∑ j, a j ^ 2) / 4) := by
  have he : ∀ c : ℝ, Real.exp (-(t ^ 2 * ∑ j, a j ^ 2) / c)
      = ∏ j, Real.exp (-((t * a j) ^ 2) / c) := by
    intro c
    rw [← Real.exp_sum]
    congr 1
    rw [Finset.mul_sum, ← Finset.sum_neg_distrib, Finset.sum_div]
    refine Finset.sum_congr rfl fun j _ => ?_; ring
  rw [he 2, he 4]
  have hx2 : ∀ j, (t * a j) ^ 2 ≤ 1 := fun j => by
    rw [← sq_abs]; nlinarith [abs_nonneg (t * a j), ht j]
  refine (abs_prod_sub_prod_le_weighted Finset.univ _ _ (fun j => Real.exp (-((t * a j) ^ 2) / 4))
    (fun j _ => Real.exp_pos _) (fun j _ => abs_cos_le_exp (ht j))
    (fun j _ => by
      rw [abs_of_pos (Real.exp_pos _)]
      exact Real.exp_le_exp.mpr (by nlinarith [sq_nonneg (t * a j)]))).trans ?_
  rw [mul_comm]
  refine mul_le_mul_of_nonneg_right ?_ (Finset.prod_nonneg fun j _ => (Real.exp_pos _).le)
  calc ∑ j, |Real.cos (t * a j) - Real.exp (-((t * a j) ^ 2) / 2)| / Real.exp (-((t * a j) ^ 2) / 4)
      ≤ ∑ j, (t * a j) ^ 4 / 3 * (3 / 2) := by
        refine Finset.sum_le_sum fun j _ => ?_
        have hinv : (Real.exp (-(t * a j) ^ 2 / 4))⁻¹ = Real.exp ((t * a j) ^ 2 / 4) := by
          rw [← Real.exp_neg]; congr 1; ring
        rw [div_eq_mul_inv, hinv]
        refine mul_le_mul (abs_cos_sub_exp_le (ht j)) ?_ (Real.exp_pos _).le
          (by positivity)
        exact (Real.exp_le_exp.mpr (by linarith [hx2 j])).trans exp_quarter_le
    _ = t ^ 4 * (∑ j, a j ^ 4) / 2 := by
        rw [Finset.mul_sum, Finset.sum_div]
        refine Finset.sum_congr rfl fun j _ => ?_; ring

variable {d : ℕ}

/-- **Theorem 1 for the deployed rotation, damped form.**  On the full range `|t vⱼ| ≤ √d`, the
characteristic function of a coordinate of `H D_ε v` is within
`(t⁴ ∑ⱼ vⱼ⁴/(2d²)) e^{−t²‖v‖²/(4d)}` of the Gaussian `e^{−t²‖v‖²/(2d)}`. -/
theorem abs_signAvg_cos_coord_sub_exp_le_damped {H : Od d} (hH : IsFlat H) (v : Ed d)
    (i : Fin d) {t : ℝ} (ht : ∀ j, |t * v j| ≤ Real.sqrt d) :
    |signAvg (fun s => Real.cos (t * (hadRot H s • v) i))
        - Real.exp (-(t ^ 2 * ‖v‖ ^ 2 / d) / 2)|
      ≤ t ^ 4 * (∑ j, v j ^ 4) / (2 * (d : ℝ) ^ 2) * Real.exp (-(t ^ 2 * ‖v‖ ^ 2 / d) / 4) := by
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
    have hs : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd
    have : |t * a j| = |t * v j| / Real.sqrt d := by
      rw [ha]; simp only
      rw [show t * ((H : Matrix (Fin d) (Fin d) ℝ) i j * v j)
        = (H : Matrix (Fin d) (Fin d) ℝ) i j * (t * v j) by ring, abs_mul, abs_entry_of_isFlat hH]
      field_simp
    rw [this, div_le_one hs]
    exact ht j
  have := abs_prod_cos_sub_exp_le_damped a hta
  rw [h2, h4] at this
  have e2 : -(t ^ 2 * (‖v‖ ^ 2 / d)) / 2 = -(t ^ 2 * ‖v‖ ^ 2 / d) / 2 := by ring
  have e4 : -(t ^ 2 * (‖v‖ ^ 2 / d)) / 4 = -(t ^ 2 * ‖v‖ ^ 2 / d) / 4 := by ring
  rw [e2, e4] at this
  refine this.trans (le_of_eq ?_)
  ring

/-- The damped characteristic-function gap between the coordinate law and `N(0, ‖v‖²/d)`. -/
theorem norm_charFun_coordLaw_sub_gaussian_le_damped {H : Od d} (hH : IsFlat H) (v : Ed d)
    (i : Fin d) {t : ℝ} (ht : ∀ j, |t * v j| ≤ Real.sqrt d) :
    ‖charFun (coordLaw H v i) t - charFun (gaussianReal 0 (Real.toNNReal (‖v‖ ^ 2 / d))) t‖
      ≤ t ^ 4 * (∑ j, v j ^ 4) / (2 * (d : ℝ) ^ 2) * Real.exp (-(t ^ 2 * ‖v‖ ^ 2 / d) / 4) := by
  rw [charFun_coordLaw, charFun_gaussianReal]
  have h := abs_signAvg_cos_coord_sub_exp_le_damped hH v i ht
  have hcoord : ∀ s, (hadRot H s • v) i = radSum (fun j => (H : Matrix (Fin d) (Fin d) ℝ) i j * v j) s :=
    fun s => hadRot_apply_eq_radSum H s v i
  simp_rw [hcoord, signAvg_cos_radSum] at h
  have hnn : (0 : ℝ) ≤ ‖v‖ ^ 2 / d := by positivity
  have hg : Complex.exp (↑t * ↑(0 : ℝ) * Complex.I - ↑↑(Real.toNNReal (‖v‖ ^ 2 / d)) * ↑t ^ 2 / 2)
      = ((Real.exp (-(t ^ 2 * ‖v‖ ^ 2 / d) / 2) : ℝ) : ℂ) := by
    rw [Complex.ofReal_exp]
    congr 1
    rw [Real.coe_toNNReal _ hnn]
    push_cast
    ring
  rw [hg, ← Complex.ofReal_sub, Complex.norm_real, Real.norm_eq_abs]
  exact h

end Hadamard

/-- The Gaussian density `N(0, v)` is at most `(√(2πv))⁻¹`. -/
theorem gaussianPDFReal_le (v : NNReal) (x : ℝ) :
    gaussianPDFReal 0 v x ≤ (Real.sqrt (2 * π * v))⁻¹ := by
  rw [gaussianPDFReal_def]
  simp only
  refine mul_le_of_le_one_right (inv_nonneg.mpr (Real.sqrt_nonneg _)) ?_
  rw [Real.exp_le_one_iff]
  have : 0 ≤ (x - 0) ^ 2 / (2 * v) := by positivity
  rw [neg_div]
  linarith

/-- The Gaussian distribution function `N(0, v)`, `v ≠ 0`, is Lipschitz with constant
`(√(2πv))⁻¹`. -/
theorem cdfR_gaussianReal_lipschitz {v : NNReal} (hv : v ≠ 0) {x y : ℝ} (hxy : x ≤ y) :
    cdfR (gaussianReal 0 v) y - cdfR (gaussianReal 0 v) x ≤ (Real.sqrt (2 * π * v))⁻¹ * (y - x) := by
  have h : cdfR (gaussianReal 0 v) y - cdfR (gaussianReal 0 v) x
      = (gaussianReal 0 v).real (Set.Ioc x y) := by
    unfold cdfR
    rw [Ioc_eq_Iic_diff, measureReal_sdiff (Set.Iic_subset_Iic.mpr hxy) measurableSet_Iic]
  rw [h, measureReal_def, gaussianReal_apply_eq_integral _ hv,
    ENNReal.toReal_ofReal (setIntegral_nonneg measurableSet_Ioc fun z _ => gaussianPDFReal_nonneg _ _ _)]
  calc ∫ z in Set.Ioc x y, gaussianPDFReal 0 v z
      ≤ ∫ z in Set.Ioc x y, (Real.sqrt (2 * π * v))⁻¹ :=
        setIntegral_mono_on (integrable_gaussianPDFReal 0 v).integrableOn
          (integrableOn_const (by rw [Real.volume_Ioc]; exact ENNReal.ofReal_ne_top)) measurableSet_Ioc
          fun z _ => gaussianPDFReal_le v z
    _ = (Real.sqrt (2 * π * v))⁻¹ * (y - x) := by
        rw [setIntegral_const, smul_eq_mul, measureReal_def, Real.volume_Ioc,
          ENNReal.toReal_ofReal (by linarith)]
        ring

/-- `∫_0^∞ u³ e^{−bu²} du = 1/(2b²)`. -/
theorem integral_Ioi_pow_three_mul_exp {b : ℝ} (hb : 0 < b) :
    ∫ u in Set.Ioi (0 : ℝ), u ^ 3 * Real.exp (-b * u ^ 2) = 1 / (2 * b ^ 2) := by
  have hderiv : ∀ u ∈ Set.Ioi (0 : ℝ),
      HasDerivAt (fun u : ℝ => -(u ^ 2 / (2 * b) + 1 / (2 * b ^ 2)) * Real.exp (-b * u ^ 2))
        (u ^ 3 * Real.exp (-b * u ^ 2)) u := by
    intro u _
    have h1 : HasDerivAt (fun u : ℝ => -(u ^ 2 / (2 * b) + 1 / (2 * b ^ 2)))
        (-(((2 : ℕ) : ℝ) * u ^ (2 - 1) / (2 * b))) u :=
      (((hasDerivAt_pow 2 u).div_const (2 * b)).add_const (1 / (2 * b ^ 2))).neg
    have h2 : HasDerivAt (fun u : ℝ => Real.exp (-b * u ^ 2))
        (Real.exp (-b * u ^ 2) * (-b * (((2 : ℕ) : ℝ) * u ^ (2 - 1)))) u :=
      ((hasDerivAt_pow 2 u).const_mul (-b)).exp
    refine (h1.mul h2).congr_deriv ?_
    norm_num
    field_simp
    ring
  have hlim : Tendsto (fun u : ℝ => -(u ^ 2 / (2 * b) + 1 / (2 * b ^ 2)) * Real.exp (-b * u ^ 2))
      atTop (𝓝 0) := by
    have hA : Tendsto (fun u : ℝ => u ^ 2 * Real.exp (-b * u ^ 2)) atTop (𝓝 0) := by
      have hexp : Tendsto (fun x : ℝ => Real.exp (-(1 / 2) * x)) atTop (𝓝 0) := by
        have h1 : Tendsto (fun x : ℝ => -(1 / 2) * x) atTop atBot := by
          have := tendsto_neg_atTop_atBot.comp
            (tendsto_id.const_mul_atTop (by norm_num : (0 : ℝ) < 1 / 2))
          exact this.congr fun x => by show -(1 / 2 * id x) = -(1 / 2) * x; simp only [id]; ring
        exact Real.tendsto_exp_atBot.comp h1
      have := (rpow_mul_exp_neg_mul_sq_isLittleO_exp_neg hb 2).tendsto_zero_of_tendsto hexp
      simpa [Real.rpow_two] using this
    have hB : Tendsto (fun u : ℝ => Real.exp (-b * u ^ 2)) atTop (𝓝 0) := by
      have h1 : Tendsto (fun u : ℝ => -b * u ^ 2) atTop atBot := by
        have := tendsto_neg_atTop_atBot.comp ((tendsto_pow_atTop two_ne_zero).const_mul_atTop hb)
        exact this.congr fun u => by show -(b * u ^ 2) = -b * u ^ 2; ring
      exact Real.tendsto_exp_atBot.comp h1
    have h := (hA.const_mul (-(1 / (2 * b)))).add (hB.const_mul (-(1 / (2 * b ^ 2))))
    simp only [mul_zero, add_zero] at h
    refine h.congr fun u => ?_
    ring
  have h := integral_Ioi_of_hasDerivAt_of_nonneg
    (continuousAt_id.pow 2 |>.div_const _ |>.add_const _ |>.neg |>.mul
      (Real.continuous_exp.continuousAt.comp ((continuousAt_id.pow 2).const_mul (-b)))).continuousWithinAt
    hderiv (fun u hu => by have : (0 : ℝ) < u := hu; positivity) hlim
  rw [h]
  simp

/-- `∫ |ξ|³ e^{−bξ²} dξ = 1/b²`. -/
theorem integral_abs_pow_three_mul_exp {b : ℝ} (hb : 0 < b) :
    ∫ ξ : ℝ, |ξ| ^ 3 * Real.exp (-b * ξ ^ 2) = 1 / b ^ 2 := by
  have h := integral_comp_abs (f := fun u : ℝ => u ^ 3 * Real.exp (-b * u ^ 2))
  simp only [sq_abs] at h
  rw [h, integral_Ioi_pow_three_mul_exp hb]
  field_simp

theorem integrable_abs_pow_three_mul_exp {b : ℝ} (hb : 0 < b) :
    Integrable (fun ξ : ℝ => |ξ| ^ 3 * Real.exp (-b * ξ ^ 2)) := by
  have h := (integrable_rpow_mul_exp_neg_mul_sq hb (s := 3) (by norm_num)).norm
  refine h.congr (Filter.Eventually.of_forall fun ξ => ?_)
  simp only
  rw [Real.norm_eq_abs, abs_mul, abs_of_pos (Real.exp_pos _), show (3 : ℝ) = ((3 : ℕ) : ℝ) by norm_num,
    Real.rpow_natCast, abs_pow]

namespace Hadamard
variable {d : ℕ}

/-- **Kolmogorov–Smirnov bound for the coordinate law of `H D_ε v`.**  For `H` flat, `v ≠ 0` with
`|vⱼ| ≤ M`, every `a > 0` and every `x₀`,
`|F(x₀) − Φ(x₀)| (1 − 8M/(π√d a)) ≤ 16 ∑ⱼ vⱼ⁴/(π‖v‖⁴) + 2a/√(2π‖v‖²/d)`,
with `F` the distribution function of the coordinate law and `Φ` that of `N(0, ‖v‖²/d)`.
For a spread unit direction (`M = 1/√d`, `∑ vⱼ⁴ = 1/d`) and `a = 16/(πd)` this is a
Kolmogorov–Smirnov distance of order `1/√d`. -/
theorem abs_cdfR_coordLaw_sub_gaussian_le {H : Od d} (hH : IsFlat H) {v : Ed d} (hv0 : v ≠ 0)
    {M : ℝ} (hM : 0 < M) (hvM : ∀ j, |v j| ≤ M) (i : Fin d) {a : ℝ} (ha : 0 < a) (x₀ : ℝ) :
    |cdfR (coordLaw H v i) x₀ - cdfR (gaussianReal 0 (Real.toNNReal (‖v‖ ^ 2 / d))) x₀|
        * (1 - 8 * M / (π * Real.sqrt d * a))
      ≤ 16 * (∑ j, v j ^ 4) / (π * ‖v‖ ^ 4)
        + 2 * (Real.sqrt (2 * π * (‖v‖ ^ 2 / d)))⁻¹ * a := by
  have hd : (0 : ℝ) < d := by
    rcases Nat.eq_zero_or_pos d with h0 | h0
    · subst h0; exact Fin.elim0 i
    · exact_mod_cast h0
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd
  have hvn : 0 < ‖v‖ := norm_pos_iff.mpr hv0
  set σ2 : ℝ := ‖v‖ ^ 2 / d with hσ2
  have hσ2pos : 0 < σ2 := by positivity
  have hcoe : ((Real.toNNReal σ2 : NNReal) : ℝ) = σ2 := Real.coe_toNNReal _ hσ2pos.le
  have hvne : Real.toNNReal σ2 ≠ 0 := (Real.toNNReal_pos.mpr hσ2pos).ne'
  set T : ℝ := Real.sqrt d / (2 * π * M) with hT_def
  have hT : 0 < T := by positivity
  set m : ℝ := (Real.sqrt (2 * π * σ2))⁻¹ with hm_def
  have hm : 0 ≤ m := by positivity
  have hG : ∀ x y, x ≤ y → cdfR (gaussianReal 0 (Real.toNNReal σ2)) y
      - cdfR (gaussianReal 0 (Real.toNNReal σ2)) x ≤ m * (y - x) := by
    intro x y hxy
    have := cdfR_gaussianReal_lipschitz hvne hxy
    rwa [hcoe] at this
  set S : ℝ := ∑ j, v j ^ 4 with hS
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun j _ => by positivity
  set b : ℝ := π ^ 2 * σ2 with hb_def
  have hb : 0 < b := by positivity
  set C : ℝ := 8 * π ^ 3 * S / (d : ℝ) ^ 2 with hC
  have hC0 : 0 ≤ C := by positivity
  -- the pointwise bound on the integrand
  have hpt : ∀ ξ : ℝ, tri T ξ / (π * |ξ|)
      * ‖charFun (coordLaw H v i) (2 * π * ξ) - charFun (gaussianReal 0 (Real.toNNReal σ2)) (2 * π * ξ)‖
      ≤ C * (|ξ| ^ 3 * Real.exp (-b * ξ ^ 2)) := by
    intro ξ
    have hR : 0 ≤ C * (|ξ| ^ 3 * Real.exp (-b * ξ ^ 2)) := by positivity
    rcases le_or_gt T |ξ| with hξ | hξ
    · rw [tri_eq_zero hT hξ, zero_div, zero_mul]; exact hR
    rcases eq_or_ne ξ 0 with h0 | h0
    · subst h0; simp
    have hξpos : 0 < |ξ| := abs_pos.mpr h0
    have ht : ∀ j, |2 * π * ξ * v j| ≤ Real.sqrt d := by
      intro j
      rw [abs_mul, abs_mul, abs_of_pos (by positivity : (0 : ℝ) < 2 * π)]
      calc 2 * π * |ξ| * |v j| ≤ 2 * π * T * M := by gcongr; exact hvM j
        _ = Real.sqrt d := by rw [hT_def]; field_simp
    have hΔ := norm_charFun_coordLaw_sub_gaussian_le_damped hH v i ht
    have htri : tri T ξ ≤ 1 := tri_le_one hT ξ
    have hw : 0 ≤ 1 / (π * |ξ|) := by positivity
    calc tri T ξ / (π * |ξ|)
          * ‖charFun (coordLaw H v i) (2 * π * ξ) - charFun (gaussianReal 0 (Real.toNNReal σ2)) (2 * π * ξ)‖
        ≤ 1 / (π * |ξ|) * ((2 * π * ξ) ^ 4 * S / (2 * (d : ℝ) ^ 2)
            * Real.exp (-((2 * π * ξ) ^ 2 * ‖v‖ ^ 2 / d) / 4)) := by
          gcongr
      _ = C * (|ξ| ^ 3 * Real.exp (-b * ξ ^ 2)) := by
          have e1 : Real.exp (-((2 * π * ξ) ^ 2 * ‖v‖ ^ 2 / d) / 4) = Real.exp (-b * ξ ^ 2) := by
            congr 1; rw [hb_def, hσ2]; ring
          have e2 : (2 * π * ξ) ^ 4 = 16 * π ^ 4 * (|ξ| ^ 3 * |ξ|) := by
            rw [← pow_succ, ← abs_pow, abs_of_nonneg (by positivity : (0 : ℝ) ≤ ξ ^ 4)]; ring
          rw [e1, e2, hC]
          field_simp
          ring
  have hdom : Integrable (fun ξ : ℝ => C * (|ξ| ^ 3 * Real.exp (-b * ξ ^ 2))) :=
    (integrable_abs_pow_three_mul_exp hb).const_mul C
  have hmeas : AEStronglyMeasurable (fun ξ : ℝ => tri T ξ / (π * |ξ|)
      * ‖charFun (coordLaw H v i) (2 * π * ξ) - charFun (gaussianReal 0 (Real.toNNReal σ2)) (2 * π * ξ)‖)
      volume := by
    have h2 : Continuous fun ξ : ℝ => 2 * π * ξ := by fun_prop
    have hc : Continuous fun ξ : ℝ => ‖charFun (coordLaw H v i) (2 * π * ξ)
        - charFun (gaussianReal 0 (Real.toNNReal σ2)) (2 * π * ξ)‖ :=
      ((continuous_charFun.comp h2).sub (continuous_charFun.comp h2)).norm
    exact (((continuous_tri T).measurable.div (measurable_const.mul continuous_abs.measurable)).mul
      hc.measurable).aestronglyMeasurable
  have hI : Integrable (fun ξ : ℝ => tri T ξ / (π * |ξ|)
      * ‖charFun (coordLaw H v i) (2 * π * ξ) - charFun (gaussianReal 0 (Real.toNNReal σ2)) (2 * π * ξ)‖) :=
    hdom.mono' hmeas (Filter.Eventually.of_forall fun ξ => by
      rw [Real.norm_of_nonneg (mul_nonneg (div_nonneg (tri_nonneg _ _) (by positivity)) (norm_nonneg _))]
      exact hpt ξ)
  have hint : ∫ ξ, tri T ξ / (π * |ξ|)
      * ‖charFun (coordLaw H v i) (2 * π * ξ) - charFun (gaussianReal 0 (Real.toNNReal σ2)) (2 * π * ξ)‖
      ≤ 8 * S / (π * ‖v‖ ^ 4) := by
    refine (integral_mono hI hdom hpt).trans (le_of_eq ?_)
    rw [integral_const_mul, integral_abs_pow_three_mul_exp hb, hC, hb_def, hσ2]
    field_simp
    try ring
  have h := esseen_inequality_fejer (coordLaw H v i) (gaussianReal 0 (Real.toNNReal σ2)) hm hG hT ha hI x₀
  have hε : 2 * (2 / (π ^ 2 * T * a)) = 8 * M / (π * Real.sqrt d * a) := by
    rw [hT_def]; field_simp; ring
  rw [hε] at h
  have hfin : 16 * S / (π * ‖v‖ ^ 4) = 2 * (8 * S / (π * ‖v‖ ^ 4)) := by ring
  rw [hfin]
  linarith

end Hadamard

