import Mathlib

open Finset MeasureTheory

namespace ApexQuant.ScalarRHT

section Codebook

variable {K : ℕ}

/-- Nearest-level squared loss of a scalar codebook `c`. -/
noncomputable def ell (c : Fin (K + 1) → ℝ) (z : ℝ) : ℝ :=
  univ.inf' univ_nonempty fun k => (z - c k) ^ 2

/-- The convex piecewise-affine part of the loss: `g z = max_k (2 c_k z - c_k^2)`. -/
noncomputable def g (c : Fin (K + 1) → ℝ) (z : ℝ) : ℝ :=
  univ.sup' univ_nonempty fun k => 2 * c k * z - c k ^ 2

/-- Least codebook level. -/
noncomputable def cmin (c : Fin (K + 1) → ℝ) : ℝ := univ.inf' univ_nonempty c

/-- Greatest codebook level. -/
noncomputable def cmax (c : Fin (K + 1) → ℝ) : ℝ := univ.sup' univ_nonempty c

/-- The tilted convex part `gt z = g z - (c_min + c_max) z`. -/
noncomputable def gt (c : Fin (K + 1) → ℝ) (z : ℝ) : ℝ :=
  g c z - (cmin c + cmax c) * z

theorem cmin_le (c : Fin (K + 1) → ℝ) (k : Fin (K + 1)) : cmin c ≤ c k :=
  Finset.inf'_le c (mem_univ k)

theorem le_cmax (c : Fin (K + 1) → ℝ) (k : Fin (K + 1)) : c k ≤ cmax c :=
  Finset.le_sup' c (mem_univ k)

theorem cmin_le_cmax (c : Fin (K + 1) → ℝ) : cmin c ≤ cmax c :=
  (cmin_le c 0).trans (le_cmax c 0)

/-- The loss splits into the quadratic `z^2` minus a max of affine functions. -/
theorem loss_decomposition (c : Fin (K + 1) → ℝ) (z : ℝ) :
    ell c z = z ^ 2 - g c z := by
  apply le_antisymm
  · obtain ⟨k, -, hk⟩ := Finset.exists_mem_eq_sup' (s := univ) univ_nonempty
      fun k => 2 * c k * z - c k ^ 2
    have h : ell c z ≤ (z - c k) ^ 2 :=
      Finset.inf'_le (fun k => (z - c k) ^ 2) (mem_univ k)
    have hg : g c z = 2 * c k * z - c k ^ 2 := hk
    rw [hg]
    linarith [h, show (z - c k) ^ 2 = z ^ 2 - (2 * c k * z - c k ^ 2) by ring]
  · obtain ⟨k, -, hk⟩ := Finset.exists_mem_eq_inf' (s := univ) univ_nonempty
      fun k => (z - c k) ^ 2
    have h : 2 * c k * z - c k ^ 2 ≤ g c z :=
      Finset.le_sup' (fun k => 2 * c k * z - c k ^ 2) (mem_univ k)
    have he : ell c z = (z - c k) ^ 2 := hk
    rw [he]
    linarith [h, show (z - c k) ^ 2 = z ^ 2 - (2 * c k * z - c k ^ 2) by ring]

theorem ell_eq_sq_sub_gt (c : Fin (K + 1) → ℝ) (z : ℝ) :
    ell c z = z ^ 2 - gt c z - (cmin c + cmax c) * z := by
  rw [loss_decomposition, gt]
  ring

private theorem gt_sub_le (c : Fin (K + 1) → ℝ) (x y : ℝ) :
    gt c x - gt c y ≤ (cmax c - cmin c) * |x - y| := by
  obtain ⟨k, -, hk⟩ := Finset.exists_mem_eq_sup' (s := univ) univ_nonempty
    fun k => 2 * c k * x - c k ^ 2
  have hx : g c x = 2 * c k * x - c k ^ 2 := hk
  have hy : 2 * c k * y - c k ^ 2 ≤ g c y :=
    Finset.le_sup' (fun k => 2 * c k * y - c k ^ 2) (mem_univ k)
  have hslope : |2 * c k - (cmin c + cmax c)| ≤ cmax c - cmin c := by
    rw [abs_le]
    constructor <;> linarith [cmin_le c k, le_cmax c k]
  calc
    gt c x - gt c y ≤ (2 * c k - (cmin c + cmax c)) * (x - y) := by
      simp only [gt, hx]
      linarith
    _ ≤ |(2 * c k - (cmin c + cmax c)) * (x - y)| := le_abs_self _
    _ = |2 * c k - (cmin c + cmax c)| * |x - y| := abs_mul _ _
    _ ≤ (cmax c - cmin c) * |x - y| :=
      mul_le_mul_of_nonneg_right hslope (abs_nonneg _)

/-- The tilted convex part is Lipschitz with constant `c_max - c_min`. -/
theorem tilted_lipschitz (c : Fin (K + 1) → ℝ) (x y : ℝ) :
    |gt c x - gt c y| ≤ (cmax c - cmin c) * |x - y| := by
  rw [abs_le]
  constructor
  · have h := gt_sub_le c y x
    rw [abs_sub_comm] at h
    linarith
  · exact gt_sub_le c x y

theorem continuous_gt (c : Fin (K + 1) → ℝ) : Continuous (gt c) := by
  have hL : (0 : ℝ) ≤ cmax c - cmin c := sub_nonneg.mpr (cmin_le_cmax c)
  have h : LipschitzWith ⟨cmax c - cmin c, hL⟩ (gt c) :=
    LipschitzWith.of_dist_le_mul fun x y => by
      rw [Real.dist_eq, Real.dist_eq]
      exact tilted_lipschitz c x y
  exact h.continuous

end Codebook

section Distortion

variable {K : ℕ} {Ω Ω' Ω'' : Type*} [MeasurableSpace Ω] [MeasurableSpace Ω']
  [MeasurableSpace Ω'']

theorem integrable_gt_comp (c : Fin (K + 1) → ℝ) {μ : Measure Ω} [IsFiniteMeasure μ]
    {S : Ω → ℝ} (hS : Integrable S μ) : Integrable (fun ω => gt c (S ω)) μ := by
  have hbound : Integrable (fun ω => |gt c 0| + (cmax c - cmin c) * |S ω|) μ :=
    (integrable_const _).add (hS.abs.const_mul _)
  refine hbound.mono' ((continuous_gt c).comp_aestronglyMeasurable hS.aestronglyMeasurable) ?_
  refine Filter.Eventually.of_forall fun ω => ?_
  have h := tilted_lipschitz c (S ω) 0
  rw [sub_zero] at h
  rw [Real.norm_eq_abs]
  calc
    |gt c (S ω)| = |(gt c (S ω) - gt c 0) + gt c 0| := by ring_nf
    _ ≤ |gt c (S ω) - gt c 0| + |gt c 0| := abs_add_le _ _
    _ ≤ |gt c 0| + (cmax c - cmin c) * |S ω| := by linarith

/-- For a centered square-integrable variable the expected loss is the second
moment minus the expectation of the tilted convex part. -/
theorem integral_ell (c : Fin (K + 1) → ℝ) {μ : Measure Ω} [IsFiniteMeasure μ]
    {S : Ω → ℝ} (hS : Integrable S μ) (hS2 : Integrable (fun ω => S ω ^ 2) μ)
    (hmean : ∫ ω, S ω ∂μ = 0) :
    ∫ ω, ell c (S ω) ∂μ = ∫ ω, S ω ^ 2 ∂μ - ∫ ω, gt c (S ω) ∂μ := by
  have hgt := integrable_gt_comp c hS
  have h1 : ∫ ω, (S ω ^ 2 - gt c (S ω)) - (cmin c + cmax c) * S ω ∂μ =
      ∫ ω, S ω ^ 2 - gt c (S ω) ∂μ - ∫ ω, (cmin c + cmax c) * S ω ∂μ :=
    integral_sub (hS2.sub hgt) (hS.const_mul _)
  have h2 : ∫ ω, S ω ^ 2 - gt c (S ω) ∂μ = ∫ ω, S ω ^ 2 ∂μ - ∫ ω, gt c (S ω) ∂μ :=
    integral_sub hS2 hgt
  simp only [ell_eq_sq_sub_gt]
  rw [h1, h2, integral_const_mul, hmean]
  ring

/-- Distortion comparison. If `S` and `T` are centered with equal second moments
and are `w`-close in the Lipschitz-test-function (Wasserstein-1) sense `hW`, then
their expected nearest-level losses differ by at most `(c_max - c_min) * w`.

The normal-approximation inputs are hypotheses, not theorems of this file: the
bound `hW` is where Stein's method (`w ≤ 3 Σ |v_i|^3` for a Rademacher sum against
a Gaussian) or the `sqrt (2 / d)` bound between a scaled sphere coordinate and a
Gaussian would enter. Neither of those Wasserstein bounds is proved here. -/
theorem distortion_gap_le (c : Fin (K + 1) → ℝ)
    {μ : Measure Ω} [IsProbabilityMeasure μ] {μ' : Measure Ω'} [IsProbabilityMeasure μ']
    {S : Ω → ℝ} {T : Ω' → ℝ} {w : ℝ}
    (hS : Integrable S μ) (hS2 : Integrable (fun ω => S ω ^ 2) μ)
    (hT : Integrable T μ') (hT2 : Integrable (fun ω => T ω ^ 2) μ')
    (hSmean : ∫ ω, S ω ∂μ = 0) (hTmean : ∫ ω, T ω ∂μ' = 0)
    (hvar : ∫ ω, S ω ^ 2 ∂μ = ∫ ω, T ω ^ 2 ∂μ')
    (hW : ∀ L : ℝ, 0 ≤ L → ∀ f : ℝ → ℝ, (∀ x y, |f x - f y| ≤ L * |x - y|) →
      |∫ ω, f (S ω) ∂μ - ∫ ω, f (T ω) ∂μ'| ≤ L * w) :
    |∫ ω, ell c (S ω) ∂μ - ∫ ω, ell c (T ω) ∂μ'| ≤ (cmax c - cmin c) * w := by
  have h := hW (cmax c - cmin c) (sub_nonneg.mpr (cmin_le_cmax c)) (gt c)
    (tilted_lipschitz c)
  rw [integral_ell c hS hS2 hSmean, integral_ell c hT hT2 hTmean, hvar]
  rw [abs_sub_comm] at h
  simpa using h

/-- Triangle version of `distortion_gap_le` through an intermediate variable `Z`
(for example a Gaussian). As before, the two normal-approximation inputs `hW₁`
and `hW₂` are hypotheses and are not proved here. -/
theorem distortion_gap_le_triangle (c : Fin (K + 1) → ℝ)
    {μ : Measure Ω} [IsProbabilityMeasure μ] {μ' : Measure Ω'} [IsProbabilityMeasure μ']
    {γ : Measure Ω''}
    {S : Ω → ℝ} {T : Ω' → ℝ} {Z : Ω'' → ℝ} {w₁ w₂ : ℝ}
    (hS : Integrable S μ) (hS2 : Integrable (fun ω => S ω ^ 2) μ)
    (hT : Integrable T μ') (hT2 : Integrable (fun ω => T ω ^ 2) μ')
    (hSmean : ∫ ω, S ω ∂μ = 0) (hTmean : ∫ ω, T ω ∂μ' = 0)
    (hvar : ∫ ω, S ω ^ 2 ∂μ = ∫ ω, T ω ^ 2 ∂μ')
    (hW₁ : ∀ L : ℝ, 0 ≤ L → ∀ f : ℝ → ℝ, (∀ x y, |f x - f y| ≤ L * |x - y|) →
      |∫ ω, f (S ω) ∂μ - ∫ ω, f (Z ω) ∂γ| ≤ L * w₁)
    (hW₂ : ∀ L : ℝ, 0 ≤ L → ∀ f : ℝ → ℝ, (∀ x y, |f x - f y| ≤ L * |x - y|) →
      |∫ ω, f (T ω) ∂μ' - ∫ ω, f (Z ω) ∂γ| ≤ L * w₂) :
    |∫ ω, ell c (S ω) ∂μ - ∫ ω, ell c (T ω) ∂μ'| ≤ (cmax c - cmin c) * (w₁ + w₂) := by
  refine distortion_gap_le c hS hS2 hT hT2 hSmean hTmean hvar fun L hL f hf => ?_
  have h₁ := hW₁ L hL f hf
  have h₂ := hW₂ L hL f hf
  rw [abs_sub_comm] at h₂
  calc
    |∫ ω, f (S ω) ∂μ - ∫ ω, f (T ω) ∂μ'|
        = |(∫ ω, f (S ω) ∂μ - ∫ ω, f (Z ω) ∂γ) +
            (∫ ω, f (Z ω) ∂γ - ∫ ω, f (T ω) ∂μ')| := by ring_nf
    _ ≤ |∫ ω, f (S ω) ∂μ - ∫ ω, f (Z ω) ∂γ| +
          |∫ ω, f (Z ω) ∂γ - ∫ ω, f (T ω) ∂μ'| := abs_add_le _ _
    _ ≤ L * (w₁ + w₂) := by linarith

end Distortion

section CubeSum

/-- On the unit sphere the third absolute moment is at most any sup-norm bound. -/
theorem sum_abs_cube_le_sup {d : ℕ} (v : Fin d → ℝ) (hv : ∑ i, v i ^ 2 = 1)
    {M : ℝ} (hM : ∀ i, |v i| ≤ M) : ∑ i, |v i| ^ 3 ≤ M := by
  calc
    ∑ i, |v i| ^ 3 ≤ ∑ i, M * v i ^ 2 := by
      refine Finset.sum_le_sum fun i _ => ?_
      have h : |v i| ^ 3 = |v i| * v i ^ 2 := by rw [← sq_abs (v i)]; ring
      rw [h]
      exact mul_le_mul_of_nonneg_right (hM i) (sq_nonneg _)
    _ = M := by rw [← Finset.mul_sum, hv, mul_one]

/-- The same bound against the largest absolute coordinate. -/
theorem sum_abs_cube_le_max {d : ℕ} (v : Fin (d + 1) → ℝ) (hv : ∑ i, v i ^ 2 = 1) :
    ∑ i, |v i| ^ 3 ≤ univ.sup' univ_nonempty fun i => |v i| :=
  sum_abs_cube_le_sup v hv fun i => Finset.le_sup' (fun i => |v i|) (mem_univ i)

end CubeSum

section Rademacher

/-- The sign `±1` encoded by a boolean. -/
def sgn (b : Bool) : ℝ := if b then 1 else -1

/-- The Rademacher sum `S ε = Σ_i sgn (ε i) * v i`. -/
def rsum {d : ℕ} (v : Fin d → ℝ) (ε : Fin d → Bool) : ℝ := ∑ i, sgn (ε i) * v i

/-- Uniform average over all sign patterns. -/
noncomputable def avg {d : ℕ} (f : (Fin d → Bool) → ℝ) : ℝ := (∑ ε, f ε) / 2 ^ d

theorem rsum_cons {d : ℕ} (v : Fin (d + 1) → ℝ) (b : Bool) (ε : Fin d → Bool) :
    rsum v (Fin.cons b ε) = sgn b * v 0 + rsum (Fin.tail v) ε := by
  simp [rsum, Fin.sum_univ_succ, Fin.tail]

/-- Conditioning on the first sign. -/
theorem sum_rsum_succ {d : ℕ} (v : Fin (d + 1) → ℝ) (F : ℝ → ℝ) :
    ∑ ε, F (rsum v ε) =
      ∑ ε : Fin d → Bool,
        (F (v 0 + rsum (Fin.tail v) ε) + F (-v 0 + rsum (Fin.tail v) ε)) := by
  rw [← (Fin.consEquiv fun _ : Fin (d + 1) => Bool).sum_comp, Fintype.sum_prod_type,
    Fintype.sum_bool, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun ε _ => ?_
  simp [Fin.consEquiv, rsum_cons, sgn]

theorem card_signs (d : ℕ) : ∑ _ε : Fin d → Bool, (1 : ℝ) = 2 ^ d := by
  simp

theorem sum_rsum {d : ℕ} (v : Fin d → ℝ) : ∑ ε, rsum v ε = 0 := by
  induction d with
  | zero => simp [rsum]
  | succ d ih =>
      rw [sum_rsum_succ v fun x => x]
      have h : ∀ ε : Fin d → Bool,
          (v 0 + rsum (Fin.tail v) ε) + (-v 0 + rsum (Fin.tail v) ε) =
            2 * rsum (Fin.tail v) ε := fun ε => by ring
      simp only [h, ← Finset.mul_sum, ih, mul_zero]

theorem sum_rsum_sq {d : ℕ} (v : Fin d → ℝ) :
    ∑ ε, rsum v ε ^ 2 = 2 ^ d * ∑ i, v i ^ 2 := by
  induction d with
  | zero => simp [rsum]
  | succ d ih =>
      rw [sum_rsum_succ v fun x => x ^ 2]
      have h : ∀ ε : Fin d → Bool,
          (v 0 + rsum (Fin.tail v) ε) ^ 2 + (-v 0 + rsum (Fin.tail v) ε) ^ 2 =
            2 * v 0 ^ 2 * 1 + 2 * rsum (Fin.tail v) ε ^ 2 := fun ε => by ring
      simp only [h, Finset.sum_add_distrib, ← Finset.mul_sum, ih, card_signs,
        Fin.sum_univ_succ (fun i => v i ^ 2)]
      simp only [Fin.tail]
      ring

theorem sum_rsum_pow_four {d : ℕ} (v : Fin d → ℝ) :
    ∑ ε, rsum v ε ^ 4 = 2 ^ d * (3 * (∑ i, v i ^ 2) ^ 2 - 2 * ∑ i, v i ^ 4) := by
  induction d with
  | zero => simp [rsum]
  | succ d ih =>
      rw [sum_rsum_succ v fun x => x ^ 4]
      have h : ∀ ε : Fin d → Bool,
          (v 0 + rsum (Fin.tail v) ε) ^ 4 + (-v 0 + rsum (Fin.tail v) ε) ^ 4 =
            2 * v 0 ^ 4 * 1 + 12 * v 0 ^ 2 * rsum (Fin.tail v) ε ^ 2 +
              2 * rsum (Fin.tail v) ε ^ 4 := fun ε => by ring
      simp only [h, Finset.sum_add_distrib, ← Finset.mul_sum, ih, sum_rsum_sq, card_signs,
        Fin.sum_univ_succ (fun i => v i ^ 2), Fin.sum_univ_succ (fun i => v i ^ 4)]
      simp only [Fin.tail]
      ring

/-- A Rademacher sum is centered. -/
theorem avg_rsum {d : ℕ} (v : Fin d → ℝ) : avg (rsum v) = 0 := by
  simp [avg, sum_rsum]

/-- Second moment of a Rademacher sum. -/
theorem avg_rsum_sq {d : ℕ} (v : Fin d → ℝ) :
    avg (fun ε => rsum v ε ^ 2) = ∑ i, v i ^ 2 := by
  rw [avg, sum_rsum_sq]
  field_simp

/-- Fourth moment of a Rademacher sum. -/
theorem avg_rsum_pow_four {d : ℕ} (v : Fin d → ℝ) :
    avg (fun ε => rsum v ε ^ 4) = 3 * (∑ i, v i ^ 2) ^ 2 - 2 * ∑ i, v i ^ 4 := by
  rw [avg, sum_rsum_pow_four]
  field_simp

end Rademacher

section Axis

variable {d : ℕ}

/-- Coordinatewise scalar stage `q x i = qbar (sqrt d * x i) / sqrt d`. -/
noncomputable def stage (qbar : ℝ → ℝ) (x : EuclideanSpace ℝ (Fin d)) :
    EuclideanSpace ℝ (Fin d) :=
  WithLp.toLp 2 fun i => qbar (Real.sqrt d * x i) / Real.sqrt d

/-- A vector whose coordinates are all `±1 / sqrt d`. -/
def IsFlat (x : EuclideanSpace ℝ (Fin d)) : Prop :=
  ∀ i, x i = 1 / Real.sqrt d ∨ x i = -1 / Real.sqrt d

theorem isFlat_of_signs {x : EuclideanSpace ℝ (Fin d)} {σ : Fin d → ℝ}
    (hσ : ∀ i, σ i = 1 ∨ σ i = -1) (hx : ∀ i, x i = σ i / Real.sqrt d) : IsFlat x := by
  intro i
  rcases hσ i with h | h <;> simp [hx i, h]

theorem IsFlat.neg {x : EuclideanSpace ℝ (Fin d)} (hx : IsFlat x) : IsFlat (-x) := by
  intro i
  rcases hx i with h | h
  · right; simp [h]; ring
  · left; simp [h]; ring

/-- An odd scalar stage acts on a flat sign vector as multiplication by `qbar 1`. -/
theorem stage_flat {qbar : ℝ → ℝ} (hodd : ∀ x, qbar (-x) = -qbar x)
    {x : EuclideanSpace ℝ (Fin d)} (hx : IsFlat x) : stage qbar x = qbar 1 • x := by
  ext i
  have hd : Real.sqrt d ≠ 0 := by
    have : 0 < d := i.pos
    positivity
  change qbar (Real.sqrt d * x i) / Real.sqrt d = qbar 1 * x i
  rcases hx i with h | h
  · have h1 : Real.sqrt d * x i = 1 := by rw [h]; field_simp
    rw [h1, h]
    ring
  · have h1 : Real.sqrt d * x i = -1 := by rw [h]; field_simp
    rw [h1, hodd, h]
    ring

theorem residual_flat {qbar : ℝ → ℝ} (hodd : ∀ x, qbar (-x) = -qbar x)
    {x : EuclideanSpace ℝ (Fin d)} (hx : IsFlat x) :
    x - stage qbar x = (1 - qbar 1) • x := by
  rw [stage_flat hodd hx, sub_smul, one_smul]

theorem norm_sq_residual_flat {qbar : ℝ → ℝ} (hodd : ∀ x, qbar (-x) = -qbar x)
    {x : EuclideanSpace ℝ (Fin d)} (hx : IsFlat x) :
    ‖x - stage qbar x‖ ^ 2 = (1 - qbar 1) ^ 2 * ‖x‖ ^ 2 := by
  rw [residual_flat hodd hx, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]

/-- Axis obstruction for one rotated pass: if the frame `R` sends `v` to a flat
sign vector, the residual is the fixed multiple `(1 - qbar 1) • v`. -/
theorem residual_of_flat_frame {qbar : ℝ → ℝ} (hodd : ∀ x, qbar (-x) = -qbar x)
    (R : EuclideanSpace ℝ (Fin d) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin d))
    {v : EuclideanSpace ℝ (Fin d)} (hv : IsFlat (R v)) :
    v - R.symm (stage qbar (R v)) = (1 - qbar 1) • v := by
  rw [stage_flat hodd hv, map_smul, R.symm_apply_apply, sub_smul, one_smul]

/-- The sign-vector phrasing of `residual_of_flat_frame`. -/
theorem residual_of_sign_frame {qbar : ℝ → ℝ} (hodd : ∀ x, qbar (-x) = -qbar x)
    (R : EuclideanSpace ℝ (Fin d) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin d))
    {v : EuclideanSpace ℝ (Fin d)} {σ : Fin d → ℝ}
    (hσ : ∀ i, σ i = 1 ∨ σ i = -1) (hv : ∀ i, R v i = σ i / Real.sqrt d) :
    v - R.symm (stage qbar (R v)) = (1 - qbar 1) • v :=
  residual_of_flat_frame hodd R (isFlat_of_signs hσ hv)

/-- One gain-shape pass: the stage quantizes the rotated unit direction of `r`
and the result is rescaled by `‖r‖`. -/
noncomputable def pass (qbar : ℝ → ℝ)
    (R : EuclideanSpace ℝ (Fin d) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin d))
    (r : EuclideanSpace ℝ (Fin d)) : EuclideanSpace ℝ (Fin d) :=
  r - ‖r‖ • R.symm (stage qbar (R (‖r‖⁻¹ • r)))

/-- A gain-shape pass keeps a multiple of a flat-frame unit axis on that axis. -/
theorem pass_axis {qbar : ℝ → ℝ} (hodd : ∀ x, qbar (-x) = -qbar x)
    (R : EuclideanSpace ℝ (Fin d) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin d))
    {e : EuclideanSpace ℝ (Fin d)} (he : ‖e‖ = 1) (hflat : IsFlat (R e)) (a : ℝ) :
    pass qbar R (a • e) = (a * (1 - qbar 1)) • e := by
  have hnorm : ‖a • e‖ = |a| := by rw [norm_smul, he, mul_one, Real.norm_eq_abs]
  rw [pass, hnorm]
  rcases lt_trichotomy a 0 with ha | ha | ha
  · have hdir : |a|⁻¹ • a • e = -e := by
      rw [smul_smul, abs_of_neg ha, show (-a)⁻¹ * a = -1 by rw [inv_neg, neg_mul, inv_mul_cancel₀ ha.ne], neg_one_smul]
    have hflat' : IsFlat (R (-e)) := by rw [map_neg]; exact hflat.neg
    rw [hdir, stage_flat hodd hflat', map_smul, R.symm_apply_apply, abs_of_neg ha]
    module
  · subst ha
    simp
  · have hdir : |a|⁻¹ • a • e = e := by
      rw [smul_smul, abs_of_pos ha, inv_mul_cancel₀ ha.ne', one_smul]
    rw [hdir, stage_flat hodd hflat, map_smul, R.symm_apply_apply, abs_of_pos ha]
    module

/-- Iterated axis obstruction: if every pass frame maps the unit axis `e` to a
flat sign vector, the residual after `P` passes is `(1 - qbar 1)^P` times the
initial residual `s • e`. -/
theorem iterate_axis {qbar : ℝ → ℝ} (hodd : ∀ x, qbar (-x) = -qbar x)
    (R : ℕ → EuclideanSpace ℝ (Fin d) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin d))
    {e : EuclideanSpace ℝ (Fin d)} (he : ‖e‖ = 1) (hflat : ∀ p, IsFlat (R p e))
    (s : ℝ) (r : ℕ → EuclideanSpace ℝ (Fin d)) (h0 : r 0 = s • e)
    (hstep : ∀ p, r (p + 1) = pass qbar (R p) (r p)) (P : ℕ) :
    r P = (s * (1 - qbar 1) ^ P) • e := by
  induction P with
  | zero => simp [h0]
  | succ P ih => rw [hstep, ih, pass_axis hodd (R P) he (hflat P), pow_succ, mul_assoc]

/-- After `P` flat-frame passes the squared residual norm is exactly
`(1 - qbar 1)^(2P)` times the initial one: no decay faster than geometric with
this fixed ratio, whatever the frames. -/
theorem norm_sq_iterate_axis {qbar : ℝ → ℝ} (hodd : ∀ x, qbar (-x) = -qbar x)
    (R : ℕ → EuclideanSpace ℝ (Fin d) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin d))
    {e : EuclideanSpace ℝ (Fin d)} (he : ‖e‖ = 1) (hflat : ∀ p, IsFlat (R p e))
    (s : ℝ) (r : ℕ → EuclideanSpace ℝ (Fin d)) (h0 : r 0 = s • e)
    (hstep : ∀ p, r (p + 1) = pass qbar (R p) (r p)) (P : ℕ) :
    ‖r P‖ ^ 2 = (1 - qbar 1) ^ (2 * P) * ‖r 0‖ ^ 2 := by
  rw [iterate_axis hodd R he hflat s r h0 hstep P, h0, norm_smul, norm_smul, he,
    Real.norm_eq_abs, Real.norm_eq_abs, mul_one, mul_one, sq_abs, sq_abs, pow_mul']
  ring

end Axis

section Envelope

/-- Two-sided geometric envelope from per-step ratio bounds. -/
theorem envelope {a : ℕ → ℝ} {Dm Dp : ℝ} (ha : ∀ p, 0 ≤ a p) (hDm : 0 ≤ Dm)
    (hlo : ∀ p, Dm * a p ≤ a (p + 1)) (hhi : ∀ p, a (p + 1) ≤ Dp * a p) (P : ℕ) :
    Dm ^ P * a 0 ≤ a P ∧ a P ≤ Dp ^ P * a 0 := by
  constructor
  · induction P with
    | zero => simp
    | succ P ih =>
        calc
          Dm ^ (P + 1) * a 0 = Dm * (Dm ^ P * a 0) := by ring
          _ ≤ Dm * a P := mul_le_mul_of_nonneg_left ih hDm
          _ ≤ a (P + 1) := hlo P
  · rcases (ha 0).eq_or_lt with h0 | h0
    · have hz : ∀ p, a p = 0 := by
        intro p
        induction p with
        | zero => exact h0.symm
        | succ p ih =>
            have h := hhi p
            rw [ih, mul_zero] at h
            exact le_antisymm h (ha _)
      rw [hz P, hz 0, mul_zero]
    · have hDp : 0 ≤ Dp := by
        have h := (hlo 0).trans (hhi 0)
        exact hDm.trans (le_of_mul_le_mul_right h h0)
      induction P with
      | zero => simp
      | succ P ih =>
          calc
            a (P + 1) ≤ Dp * a P := hhi P
            _ ≤ Dp * (Dp ^ P * a 0) := mul_le_mul_of_nonneg_left ih hDp
            _ = Dp ^ (P + 1) * a 0 := by ring

end Envelope

end ApexQuant.ScalarRHT
