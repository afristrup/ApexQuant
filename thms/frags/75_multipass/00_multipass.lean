/-! ## §75  Multi-pass second-moment law

One pass of §73 at the residual `r`, with `λ = ρ²`, `α = passAlpha q`, `c = (λ - α)/(d - 1)` and
`τ = α - c` (so `τ + c·d = λ`), has quadratic form
`E ⟪v, r⁺⟫² = τ ⟪v, r⟫² + c ‖r‖² ‖v‖²` for every `r`, including `r = 0`
(`MultiPass.integral_inner_rplus_sq`), and `E ‖r⁺‖² = λ ‖r‖²` (`MultiPass.integral_norm_rplus_sq`).

`MultiPass.resP q w P g` is the residual after `P` passes with rotations `g : Fin P → O(d)`,
under the product `MultiPass.haarPi d P` of `P` independent Haar measures.  Splitting off the
last rotation (`MultiPass.integral_resP_succ`) and inducting gives

  `E ‖r_P‖² = λ^P ‖w‖²`  (`MultiPass.integral_norm_resP_sq`),
  `E ⟪v, r_P⟫² = τ^P ⟪v, w⟫² + ((λ^P - τ^P)/d) ‖w‖² ‖v‖²`  (`MultiPass.integral_inner_resP_sq`).

Divided by `λ^P ‖w‖²` this is `η^P ⟪v, ŵ⟫² + (1 - η^P) ‖v‖²/d` with `η = τ/λ`
(`MultiPass.integral_inner_resP_sq_normalized`), and `-1/(d-1) ≤ η ≤ 1` when `α ≤ λ`, with
`η < 1` when `α < λ` (`MultiPass.passEta_mem`, `MultiPass.passEta_lt_one`), so `|η| < 1` for
`d ≥ 3` (`MultiPass.abs_passEta_lt_one`).

First moments: `E ⟪v, r_P⟫ = β^P ⟪v, w⟫` (`MultiPass.integral_inner_resP`), its weighted finite
sums (`MultiPass.integral_sum_inner_resP`) and the centred form
`E (⟪v, r_P⟫ - β^P ⟪v, w⟫)² = (τ^P - β^(2P)) ⟪v, w⟫² + ((λ^P - τ^P)/d) ‖w‖² ‖v‖²`
(`MultiPass.integral_inner_resP_sub_sq`).  For a symmetric `H`,
`E ⟪r_P, H r_P⟫ = τ^P ⟪w, H w⟫ + ((λ^P - τ^P)/d) ‖w‖² tr H` (`MultiPass.integral_inner_resP_map`,
normalised in `MultiPass.integral_inner_resP_map_normalized`). -/

namespace MultiPass

open Matrix Orthogonal PassCov
open scoped InnerProductSpace

variable {d : ℕ}

/-- `λ = ρ² = E ‖U - q U‖²` for `U` uniform on the sphere. -/
noncomputable def passLambda (q : Ed d → Ed d) : ℝ :=
  stageDistortion q (sphereUniformAmbient (Ed d))

/-- The transverse eigenvalue `c = (λ - α)/(d - 1)` of one pass. -/
noncomputable def passC (q : Ed d → Ed d) : ℝ :=
  (passLambda q - passAlpha q) / ((d : ℝ) - 1)

/-- `τ = α - c`, the excess of the radial over the transverse eigenvalue. -/
noncomputable def passTau (q : Ed d → Ed d) : ℝ :=
  passAlpha q - passC q

theorem passTau_add (hd : 2 ≤ d) (q : Ed d → Ed d) :
    passTau q + passC q * d = passLambda q := by
  have h : (d : ℝ) - 1 ≠ 0 := by
    have : (2 : ℝ) ≤ d := by exact_mod_cast hd
    linarith
  simp only [passTau, passC]
  field_simp
  ring

theorem rplus_zero (q : Ed d → Ed d) (g : Od d) : rplus q 0 g = 0 := by
  simp [rplus]

theorem ofLp_dot (x y : Ed d) : (WithLp.ofLp x : Fin d → ℝ) ⬝ᵥ WithLp.ofLp y = ⟪x, y⟫_ℝ := by
  simp [dotProduct, sum_mul_eq_inner]

/-- **One pass, quadratic form.**  For every `r`, including `r = 0`, and every `v`,
`E ⟪v, r⁺⟫² = τ ⟪v, r⟫² + c ‖r‖² ‖v‖²`. -/
theorem integral_inner_rplus_sq (hd : 2 ≤ d) {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (r v : Ed d) :
    ∫ g, ⟪v, rplus q r g⟫_ℝ ^ 2 ∂(haarOd d)
      = passTau q * ⟪v, r⟫_ℝ ^ 2 + passC q * ‖r‖ ^ 2 * ‖v‖ ^ 2 := by
  by_cases hr : r = 0
  · subst hr
    simp [rplus_zero]
  have hn : ‖r‖ ≠ 0 := norm_ne_zero_iff.mpr hr
  have h1 : ∀ g : Od d, ⟪v, rplus q r g⟫_ℝ ^ 2
      = (∑ i, (WithLp.ofLp v : Fin d → ℝ) i * rplus q r g i)
        * (∑ j, (WithLp.ofLp v : Fin d → ℝ) j * rplus q r g j) := fun g => by
    rw [sum_mul_eq_inner, sq]
  rw [integral_congr_ae (ae_of_all _ h1), ← dot_secondMoment hr hq hB, secondMoment_eq hd hr hq hB]
  simp only [Matrix.smul_mulVec, Matrix.add_mulVec, Matrix.sub_mulVec, Matrix.one_mulVec,
    vecMulVec_mulVec, op_smul_eq_smul, dotProduct_smul, dotProduct_add, dotProduct_sub, ofLp_dot, smul_eq_mul,
    real_inner_smul_left, real_inner_smul_right, real_inner_self_eq_norm_sq]
  rw [real_inner_comm r v]
  have hd1 : (d : ℝ) - 1 ≠ 0 := by
    have : (2 : ℝ) ≤ d := by exact_mod_cast hd
    linarith
  simp only [passTau, passC, passLambda]
  field_simp
  ring

/-- **One pass, decay.**  For every `r`, `E ‖r⁺‖² = λ ‖r‖²`. -/
theorem integral_norm_rplus_sq {q : Ed d → Ed d} (hq : Measurable q) (r : Ed d) :
    ∫ g, ‖rplus q r g‖ ^ 2 ∂(haarOd d) = passLambda q * ‖r‖ ^ 2 := by
  by_cases hr : r = 0
  · subst hr
    simp [rplus_zero]
  have hF : Measurable fun u : Ed d => ‖u - q u‖ ^ 2 :=
    (measurable_id.sub hq).norm.pow_const 2
  rw [integral_congr_ae (ae_of_all _ fun g => norm_sq_rplus (q := q) hr g), integral_const_mul,
    integral_haar_smul_hat hr hF, passLambda, stageDistortion, mul_comm]

/-- The residual after `P` passes with independent rotations `g 0, …, g (P-1)`. -/
noncomputable def resP (q : Ed d → Ed d) (w : Ed d) : (P : ℕ) → (Fin P → Od d) → Ed d
  | 0, _ => w
  | P + 1, g => rplus q (resP q w P (Fin.init g)) (g (Fin.last P))

/-- The product of `P` independent Haar measures. -/
noncomputable abbrev haarPi (d P : ℕ) : Measure (Fin P → Od d) :=
  Measure.pi fun _ => haarOd d

theorem measurable_rplus_uncurry {q : Ed d → Ed d} (hq : Measurable q) :
    Measurable fun x : Ed d × Od d => rplus q x.1 x.2 := by
  have h1 : Measurable fun x : Ed d × Od d => ‖x.1‖⁻¹ • x.1 :=
    measurable_fst.norm.inv.smul measurable_fst
  have h2 : Measurable fun x : Ed d × Od d => x.2 • (‖x.1‖⁻¹ • x.1) := measurable_snd.smul h1
  have h3 : Measurable fun x : Ed d × Od d => x.2⁻¹ • q (x.2 • (‖x.1‖⁻¹ • x.1)) :=
    measurable_snd.inv.smul (hq.comp h2)
  exact measurable_fst.sub (measurable_fst.norm.smul h3)

theorem measurable_resP {q : Ed d → Ed d} (hq : Measurable q) (w : Ed d) :
    ∀ P, Measurable (resP q w P)
  | 0 => measurable_const
  | P + 1 => by
    have hi : Measurable fun g : Fin (P + 1) → Od d => Fin.init g :=
      measurable_pi_lambda _ fun j => measurable_pi_apply _
    exact (measurable_rplus_uncurry hq).comp
      (((measurable_resP hq w P).comp hi).prodMk (measurable_pi_apply _))

theorem norm_rplus_le_abs {q : Ed d → Ed d} {B : ℝ} (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B)
    (r : Ed d) (g : Od d) : ‖rplus q r g‖ ≤ ‖r‖ * (1 + |B|) := by
  by_cases hr : r = 0
  · simp [hr, rplus_zero]
  exact (norm_rplus_le hr hB g).trans
    (mul_le_mul_of_nonneg_left (by linarith [le_abs_self B]) (norm_nonneg r))

theorem norm_resP_le {q : Ed d → Ed d} {B : ℝ} (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (w : Ed d) :
    ∀ P (g : Fin P → Od d), ‖resP q w P g‖ ≤ ‖w‖ * (1 + |B|) ^ P
  | 0, _ => by simp [resP]
  | P + 1, g => by
    have h := norm_rplus_le_abs hB (resP q w P (Fin.init g)) (g (Fin.last P))
    have ih := norm_resP_le hB w P (Fin.init g)
    have hB0 : 0 ≤ 1 + |B| := by positivity
    calc ‖resP q w (P + 1) g‖ ≤ ‖resP q w P (Fin.init g)‖ * (1 + |B|) := h
      _ ≤ ‖w‖ * (1 + |B|) ^ P * (1 + |B|) := mul_le_mul_of_nonneg_right ih hB0
      _ = ‖w‖ * (1 + |B|) ^ (P + 1) := by ring

/-- Splitting off the last rotation, for `|Φ x| ≤ K ‖x‖ ^ n`. -/
theorem integral_resP_succ_pow {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (w : Ed d) (P : ℕ) {Φ : Ed d → ℝ} (hΦ : Measurable Φ)
    {K : ℝ} {n : ℕ} (hK : ∀ x, |Φ x| ≤ K * ‖x‖ ^ n) :
    ∫ g, Φ (resP q w (P + 1) g) ∂(haarPi d (P + 1))
      = ∫ h, ∫ a, Φ (rplus q (resP q w P h) a) ∂(haarOd d) ∂(haarPi d P) := by
  set H : Od d × (Fin P → Od d) → ℝ := fun x => Φ (rplus q (resP q w P x.2) x.1) with hH
  have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (P + 1) => haarOd d) (Fin.last P)
  have hcomp : ∀ g : Fin (P + 1) → Od d,
      Φ (resP q w (P + 1) g) = H (MeasurableEquiv.piFinSuccAbove (fun _ => Od d) (Fin.last P) g) :=
    fun g => by simp [hH, resP]
  have hmeas : Measurable H :=
    hΦ.comp ((measurable_rplus_uncurry hq).comp
      (((measurable_resP hq w P).comp measurable_snd).prodMk measurable_fst))
  have hint : Integrable H ((haarOd d).prod (haarPi d P)) := by
    refine Integrable.of_bound hmeas.aestronglyMeasurable
      (|K| * (‖w‖ * (1 + |B|) ^ (P + 1)) ^ n) (ae_of_all _ fun x => ?_)
    have h1 := norm_rplus_le_abs hB (resP q w P x.2) x.1
    have h2 := norm_resP_le hB w P x.2
    have hB0 : 0 ≤ 1 + |B| := by positivity
    have h3 : ‖rplus q (resP q w P x.2) x.1‖ ≤ ‖w‖ * (1 + |B|) ^ (P + 1) :=
      h1.trans (by rw [pow_succ, ← mul_assoc]; exact mul_le_mul_of_nonneg_right h2 hB0)
    rw [Real.norm_eq_abs]
    calc |H x| ≤ K * ‖rplus q (resP q w P x.2) x.1‖ ^ n := hK _
      _ ≤ |K| * ‖rplus q (resP q w P x.2) x.1‖ ^ n :=
        mul_le_mul_of_nonneg_right (le_abs_self K) (pow_nonneg (norm_nonneg _) n)
      _ ≤ |K| * (‖w‖ * (1 + |B|) ^ (P + 1)) ^ n :=
        mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (norm_nonneg _) h3 n) (abs_nonneg K)
  rw [integral_congr_ae (ae_of_all _ hcomp), hmp.integral_comp' (g := H),
    integral_prod_symm H hint]

theorem integrable_resP_pow {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (w : Ed d) (P : ℕ) {Φ : Ed d → ℝ} (hΦ : Measurable Φ)
    {K : ℝ} {n : ℕ} (hK : ∀ x, |Φ x| ≤ K * ‖x‖ ^ n) :
    Integrable (fun g => Φ (resP q w P g)) (haarPi d P) := by
  refine Integrable.of_bound (hΦ.comp (measurable_resP hq w P)).aestronglyMeasurable
    (|K| * (‖w‖ * (1 + |B|) ^ P) ^ n) (ae_of_all _ fun g => ?_)
  rw [Real.norm_eq_abs]
  calc |Φ (resP q w P g)| ≤ K * ‖resP q w P g‖ ^ n := hK _
    _ ≤ |K| * ‖resP q w P g‖ ^ n :=
      mul_le_mul_of_nonneg_right (le_abs_self K) (pow_nonneg (norm_nonneg _) n)
    _ ≤ |K| * (‖w‖ * (1 + |B|) ^ P) ^ n :=
      mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (norm_nonneg _) (norm_resP_le hB w P g) n)
        (abs_nonneg K)
/-- Splitting off the last rotation: Fubini over `haar × haarPi d P`. -/
theorem integral_resP_succ {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (w : Ed d) (P : ℕ) {Φ : Ed d → ℝ} (hΦ : Measurable Φ)
    {K : ℝ} (hK : ∀ x, |Φ x| ≤ K * ‖x‖ ^ 2) :
    ∫ g, Φ (resP q w (P + 1) g) ∂(haarPi d (P + 1))
      = ∫ h, ∫ a, Φ (rplus q (resP q w P h) a) ∂(haarOd d) ∂(haarPi d P) :=
  integral_resP_succ_pow hq hB w P hΦ hK

theorem integrable_resP {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (w : Ed d) (P : ℕ) {Φ : Ed d → ℝ} (hΦ : Measurable Φ)
    {K : ℝ} (hK : ∀ x, |Φ x| ≤ K * ‖x‖ ^ 2) :
    Integrable (fun g => Φ (resP q w P g)) (haarPi d P) :=
  integrable_resP_pow hq hB w P hΦ hK


theorem abs_norm_sq_le (x : Ed d) : |‖x‖ ^ 2| ≤ 1 * ‖x‖ ^ 2 := by
  rw [one_mul, abs_of_nonneg (sq_nonneg _)]

theorem abs_inner_sq_le (v x : Ed d) : |⟪v, x⟫_ℝ ^ 2| ≤ ‖v‖ ^ 2 * ‖x‖ ^ 2 := by
  rw [abs_of_nonneg (sq_nonneg _), ← mul_pow]
  exact pow_le_pow_left₀ (abs_nonneg _) (abs_real_inner_le_norm v x) 2 |>.trans_eq' (sq_abs _)

theorem measurable_inner_sq (v : Ed d) : Measurable fun x : Ed d => ⟪v, x⟫_ℝ ^ 2 :=
  (measurable_const.inner measurable_id).pow_const 2

/-- **Decay over `P` independent passes.**  `E ‖r_P‖² = λ^P ‖w‖²` for every `w`. -/
theorem integral_norm_resP_sq {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (w : Ed d) :
    ∀ P, ∫ g, ‖resP q w P g‖ ^ 2 ∂(haarPi d P) = passLambda q ^ P * ‖w‖ ^ 2
  | 0 => by simp [resP]
  | P + 1 => by
    have hΦ : Measurable fun x : Ed d => ‖x‖ ^ 2 := measurable_norm.pow_const 2
    rw [integral_resP_succ hq hB w P hΦ abs_norm_sq_le]
    simp only [integral_norm_rplus_sq hq]
    rw [integral_const_mul, integral_norm_resP_sq hq hB w P]
    ring

/-- **Multi-pass second-moment law.**  With `P` independent Haar rotations, for every `w` and
`v`, `E ⟪v, r_P⟫² = τ^P ⟪v, w⟫² + ((λ^P - τ^P)/d) ‖w‖² ‖v‖²`. -/
theorem integral_inner_resP_sq (hd : 2 ≤ d) {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (w v : Ed d) :
    ∀ P, ∫ g, ⟪v, resP q w P g⟫_ℝ ^ 2 ∂(haarPi d P)
      = passTau q ^ P * ⟪v, w⟫_ℝ ^ 2
        + (passLambda q ^ P - passTau q ^ P) / d * ‖w‖ ^ 2 * ‖v‖ ^ 2
  | 0 => by simp [resP]
  | P + 1 => by
    rw [integral_resP_succ hq hB w P (measurable_inner_sq v) (abs_inner_sq_le v)]
    simp only [integral_inner_rplus_sq hd hq hB]
    have hΦ : Measurable fun x : Ed d => ‖x‖ ^ 2 := measurable_norm.pow_const 2
    rw [integral_add ((integrable_resP hq hB w P (measurable_inner_sq v)
        (abs_inner_sq_le v)).const_mul _)
      (((integrable_resP hq hB w P hΦ abs_norm_sq_le).const_mul _).mul_const _),
      integral_const_mul, integral_mul_const, integral_const_mul,
      integral_inner_resP_sq hd hq hB w v P, integral_norm_resP_sq hq hB w P]
    have hd0 : (d : ℝ) ≠ 0 := by
      have : (2 : ℝ) ≤ d := by exact_mod_cast hd
      linarith
    have hsum := passTau_add hd q
    rw [← hsum]
    field_simp
    ring

/-- `η = τ/λ`, the per-pass contraction of the anisotropic part. -/
noncomputable def passEta (q : Ed d → Ed d) : ℝ :=
  passTau q / passLambda q

/-- Normalisation by the decay `λ^P ‖w‖²`, pure algebra. -/
theorem normalized_form {q : Ed d → Ed d} (hl : 0 < passLambda q) {w : Ed d} (hw : w ≠ 0)
    (v : Ed d) (P : ℕ) :
    (passTau q ^ P * ⟪v, w⟫_ℝ ^ 2
        + (passLambda q ^ P - passTau q ^ P) / d * ‖w‖ ^ 2 * ‖v‖ ^ 2)
      / (passLambda q ^ P * ‖w‖ ^ 2)
      = passEta q ^ P * ⟪v, ‖w‖⁻¹ • w⟫_ℝ ^ 2 + (1 - passEta q ^ P) * ‖v‖ ^ 2 / d := by
  have hn : ‖w‖ ≠ 0 := norm_ne_zero_iff.mpr hw
  have hlP : passLambda q ^ P ≠ 0 := pow_ne_zero _ hl.ne'
  rw [real_inner_smul_right, passEta, div_pow]
  field_simp

/-- **Multi-pass law, normalised.**  For `λ > 0` and `w ≠ 0`,
`E ⟪v, r_P⟫² / (λ^P ‖w‖²) = η^P ⟪v, ŵ⟫² + (1 - η^P) ‖v‖²/d`. -/
theorem integral_inner_resP_sq_normalized (hd : 2 ≤ d) {q : Ed d → Ed d} (hq : Measurable q)
    {B : ℝ} (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (hl : 0 < passLambda q) {w : Ed d} (hw : w ≠ 0)
    (v : Ed d) (P : ℕ) :
    (∫ g, ⟪v, resP q w P g⟫_ℝ ^ 2 ∂(haarPi d P)) / (passLambda q ^ P * ‖w‖ ^ 2)
      = passEta q ^ P * ⟪v, ‖w‖⁻¹ • w⟫_ℝ ^ 2 + (1 - passEta q ^ P) * ‖v‖ ^ 2 / d := by
  rw [integral_inner_resP_sq hd hq hB w v P, normalized_form hl hw v P]

theorem passAlpha_nonneg (q : Ed d → Ed d) : 0 ≤ passAlpha q :=
  integral_nonneg fun _ => sq_nonneg _

/-- **Range of `η`.**  For `d ≥ 2`, `λ > 0` and `α ≤ λ`, `-1/(d-1) ≤ η ≤ 1`. -/
theorem passEta_mem (hd : 2 ≤ d) {q : Ed d → Ed d} (hl : 0 < passLambda q)
    (hal : passAlpha q ≤ passLambda q) :
    -1 / ((d : ℝ) - 1) ≤ passEta q ∧ passEta q ≤ 1 := by
  have hd1 : 0 < (d : ℝ) - 1 := by
    have : (2 : ℝ) ≤ d := by exact_mod_cast hd
    linarith
  have ha := passAlpha_nonneg q
  simp only [passEta, passTau, passC]
  constructor
  · rw [div_le_div_iff₀ hd1 hl]
    have e : (passAlpha q - (passLambda q - passAlpha q) / ((d : ℝ) - 1)) * ((d : ℝ) - 1)
        = passAlpha q * d - passLambda q := by
      field_simp
      ring
    rw [e]
    have : (0 : ℝ) ≤ d := by positivity
    nlinarith
  · rw [div_le_one hl, _root_.sub_le_iff_le_add]
    have : 0 ≤ (passLambda q - passAlpha q) / ((d : ℝ) - 1) := div_nonneg (by linarith) hd1.le
    linarith

/-- **Strict contraction.**  If moreover `α < λ`, then `η < 1`. -/
theorem passEta_lt_one (hd : 2 ≤ d) {q : Ed d → Ed d} (hl : 0 < passLambda q)
    (hal : passAlpha q < passLambda q) : passEta q < 1 := by
  have hd1 : 0 < (d : ℝ) - 1 := by
    have : (2 : ℝ) ≤ d := by exact_mod_cast hd
    linarith
  simp only [passEta, passTau, passC]
  rw [div_lt_one hl, sub_lt_iff_lt_add]
  have : 0 < (passLambda q - passAlpha q) / ((d : ℝ) - 1) := div_pos (by linarith) hd1
  linarith

/-- **One pass, mean.**  For every `r` and `v`, `E ⟪v, r⁺⟫ = β ⟪v, r⟫`. -/
theorem integral_inner_rplus {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (r v : Ed d) :
    ∫ g, ⟪v, rplus q r g⟫_ℝ ∂(haarOd d) = passBeta q * ⟪v, r⟫_ℝ := by
  by_cases hr : r = 0
  · subst hr
    simp [rplus_zero]
  rw [integral_inner (integrable_rplus hr hq hB), ← mean, mean_eq hr hq hB,
    real_inner_smul_right]

theorem abs_inner_le (v x : Ed d) : |⟪v, x⟫_ℝ| ≤ ‖v‖ * ‖x‖ ^ 1 := by
  rw [pow_one]
  exact abs_real_inner_le_norm v x

theorem measurable_inner_const (v : Ed d) : Measurable fun x : Ed d => ⟪v, x⟫_ℝ :=
  measurable_const.inner measurable_id

theorem integrable_inner_resP {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (w v : Ed d) (P : ℕ) :
    Integrable (fun g => ⟪v, resP q w P g⟫_ℝ) (haarPi d P) :=
  integrable_resP_pow hq hB w P (measurable_inner_const v) (abs_inner_le v)

/-- **Multi-pass mean.**  `E ⟪v, r_P⟫ = β^P ⟪v, w⟫` for every `w` and `v`. -/
theorem integral_inner_resP {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (w v : Ed d) :
    ∀ P, ∫ g, ⟪v, resP q w P g⟫_ℝ ∂(haarPi d P) = passBeta q ^ P * ⟪v, w⟫_ℝ
  | 0 => by simp [resP]
  | P + 1 => by
    rw [integral_resP_succ_pow hq hB w P (measurable_inner_const v) (abs_inner_le v)]
    simp only [integral_inner_rplus hq hB]
    rw [integral_const_mul, integral_inner_resP hq hB w v P]
    ring

/-- **Multi-pass centred second moment.**
`E (⟪v, r_P⟫ - β^P ⟪v, w⟫)² = (τ^P - β^(2P)) ⟪v, w⟫² + ((λ^P - τ^P)/d) ‖w‖² ‖v‖²`. -/
theorem integral_inner_resP_sub_sq (hd : 2 ≤ d) {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (w v : Ed d) (P : ℕ) :
    ∫ g, (⟪v, resP q w P g⟫_ℝ - passBeta q ^ P * ⟪v, w⟫_ℝ) ^ 2 ∂(haarPi d P)
      = (passTau q ^ P - passBeta q ^ (2 * P)) * ⟪v, w⟫_ℝ ^ 2
        + (passLambda q ^ P - passTau q ^ P) / d * ‖w‖ ^ 2 * ‖v‖ ^ 2 := by
  set m := passBeta q ^ P * ⟪v, w⟫_ℝ with hm
  have h1 : ∀ g, (⟪v, resP q w P g⟫_ℝ - m) ^ 2
      = (⟪v, resP q w P g⟫_ℝ ^ 2 - (2 * m) * ⟪v, resP q w P g⟫_ℝ) + m ^ 2 := fun g => by ring
  have hi2 := integrable_resP hq hB w P (measurable_inner_sq v) (abs_inner_sq_le v)
  have hi1 := (integrable_inner_resP hq hB w v P).const_mul (2 * m)
  have hi3 : Integrable (fun g => ⟪v, resP q w P g⟫_ℝ ^ 2 - (2 * m) * ⟪v, resP q w P g⟫_ℝ)
      (haarPi d P) := hi2.sub hi1
  rw [integral_congr_ae (ae_of_all _ h1), integral_add hi3 (integrable_const _),
    integral_sub hi2 hi1, integral_const_mul, integral_inner_resP_sq hd hq hB w v P,
    integral_inner_resP hq hB w v P, integral_const]
  simp only [probReal_univ, one_smul, hm]
  ring

/-- `⟪x, H x⟫ = ∑ₖ μₖ ⟪eₖ, x⟫²` in an orthonormal eigenbasis of a symmetric `H`. -/
theorem inner_map_self_eq_sum {H : Ed d →ₗ[ℝ] Ed d} (hH : H.IsSymmetric) (x : Ed d) :
    ⟪x, H x⟫_ℝ = ∑ k, hH.eigenvalues finrank_euclideanSpace_fin k
      * ⟪hH.eigenvectorBasis finrank_euclideanSpace_fin k, x⟫_ℝ ^ 2 := by
  rw [← (hH.eigenvectorBasis finrank_euclideanSpace_fin).sum_inner_mul_inner x (H x)]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [← hH, hH.apply_eigenvectorBasis, real_inner_smul_left, real_inner_comm x]
  simp only [RCLike.ofReal_real_eq_id, id]
  ring

/-- **Curvature form.**  For a symmetric `H`,
`E ⟪r_P, H r_P⟫ = τ^P ⟪w, H w⟫ + ((λ^P - τ^P)/d) ‖w‖² tr H`. -/
theorem integral_inner_resP_map (hd : 2 ≤ d) {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) {H : Ed d →ₗ[ℝ] Ed d} (hH : H.IsSymmetric) (w : Ed d)
    (P : ℕ) :
    ∫ g, ⟪resP q w P g, H (resP q w P g)⟫_ℝ ∂(haarPi d P)
      = passTau q ^ P * ⟪w, H w⟫_ℝ
        + (passLambda q ^ P - passTau q ^ P) / d * ‖w‖ ^ 2 * LinearMap.trace ℝ (Ed d) H := by
  set e := hH.eigenvectorBasis finrank_euclideanSpace_fin
  set μ := hH.eigenvalues finrank_euclideanSpace_fin
  simp only [inner_map_self_eq_sum hH]
  rw [integral_finsetSum _ fun k _ =>
    (integrable_resP hq hB w P (measurable_inner_sq (e k)) (abs_inner_sq_le (e k))).const_mul _]
  simp only [integral_const_mul, integral_inner_resP_sq hd hq hB, e.orthonormal.1, one_pow,
    mul_one, hH.trace_eq_sum_eigenvalues finrank_euclideanSpace_fin, RCLike.ofReal_real_eq_id,
    id, mul_add, Finset.sum_add_distrib, Finset.mul_sum]
  congr 1
  · exact Finset.sum_congr rfl fun k _ => by ring
  · exact Finset.sum_congr rfl fun k _ => by ring

/-- **Curvature form, normalised.**  For `λ > 0` and `w ≠ 0`,
`E ⟪r_P, H r_P⟫ / (λ^P ‖w‖²) = η^P ⟪ŵ, H ŵ⟫ + (1 - η^P) tr H / d`. -/
theorem integral_inner_resP_map_normalized (hd : 2 ≤ d) {q : Ed d → Ed d} (hq : Measurable q)
    {B : ℝ} (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) {H : Ed d →ₗ[ℝ] Ed d} (hH : H.IsSymmetric)
    (hl : 0 < passLambda q) {w : Ed d} (hw : w ≠ 0) (P : ℕ) :
    (∫ g, ⟪resP q w P g, H (resP q w P g)⟫_ℝ ∂(haarPi d P)) / (passLambda q ^ P * ‖w‖ ^ 2)
      = passEta q ^ P * ⟪‖w‖⁻¹ • w, H (‖w‖⁻¹ • w)⟫_ℝ
        + (1 - passEta q ^ P) * LinearMap.trace ℝ (Ed d) H / d := by
  have hn : ‖w‖ ≠ 0 := norm_ne_zero_iff.mpr hw
  have hlP : passLambda q ^ P ≠ 0 := pow_ne_zero _ hl.ne'
  rw [integral_inner_resP_map hd hq hB hH w P, map_smul, real_inner_smul_left,
    real_inner_smul_right, passEta, div_pow]
  field_simp

/-- **Expected reversion, weighted.**  For finitely many fixed directions `a i` with weights
`c i`, `E ∑ᵢ cᵢ ⟪aᵢ, r_P⟫ = β^P ∑ᵢ cᵢ ⟪aᵢ, w⟫`. -/
theorem integral_sum_inner_resP {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) {ι : Type*} (s : Finset ι) (c : ι → ℝ) (a : ι → Ed d)
    (w : Ed d) (P : ℕ) :
    ∫ g, ∑ i ∈ s, c i * ⟪a i, resP q w P g⟫_ℝ ∂(haarPi d P)
      = passBeta q ^ P * ∑ i ∈ s, c i * ⟪a i, w⟫_ℝ := by
  rw [integral_finsetSum _ fun i _ => (integrable_inner_resP hq hB w (a i) P).const_mul _,
    Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [integral_const_mul, integral_inner_resP hq hB w (a i) P]
  ring

/-- **Convergence.**  For `d ≥ 3`, `λ > 0` and `α < λ`, `|η| < 1`.  At `d = 2` the lower end
`η = -1/(d-1) = -1` is reached when `α = 0`. -/
theorem abs_passEta_lt_one (hd : 3 ≤ d) {q : Ed d → Ed d} (hl : 0 < passLambda q)
    (hal : passAlpha q < passLambda q) : |passEta q| < 1 := by
  have h := (passEta_mem (by omega) hl hal.le).1
  have hlt := passEta_lt_one (by omega) hl hal
  have h3 : (3 : ℝ) ≤ d := by exact_mod_cast hd
  have h2 : -1 < -1 / ((d : ℝ) - 1) := by
    rw [lt_div_iff₀ (by linarith)]
    linarith
  exact abs_lt.mpr ⟨h2.trans_le h, hlt⟩

end MultiPass

