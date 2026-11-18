/-! ## §73  Lemma 2: error covariance of a Haar pass

For `r ≠ 0` in `ℝ^d`, `rhat = r/‖r‖`, `g` Haar on `O(d)` and a measurable stage `q` bounded on
the unit sphere, the next residual is `rplus g = r - ‖r‖ • g⁻¹ • q (g • rhat)`.  With
`α = E (1 - ⟪U, q U⟫)²`, `β = E (1 - ⟪U, q U⟫)` and `ρ² = E ‖U - q U‖²` for `U` uniform on the
sphere (`passAlpha`, `passBeta`, `stageDistortion q (sphereUniformAmbient _)`),

  `E rplus = β • r`  (`PassCov.mean_eq`),
  `E rplus rplusᵀ = ‖r‖² (α • rhat rhatᵀ + ((ρ² - α)/(d-1)) • (1 - rhat rhatᵀ))`
  (`PassCov.secondMoment_eq`, for `d ≥ 2`),

and `β = ρ²` under the centroid condition (`PassCov.passBeta_eq_stageDistortion`).  The proof is
the symmetry argument: right invariance of Haar makes the law of `rplus` invariant under every
rotation fixing `rhat` (`PassCov.map_rplus_smul`), a symmetric matrix with that invariance has two
eigenvalues (`PassCov.invariant_structure`, by Householder reflections), and the two scalars are
read off against the uniform law by `Orthogonal.orbitLaw_haar`.  The Schur and centroid parts are
the body of `rht/PassCovariance.lean`. -/

namespace PassCov

open Matrix Orthogonal
open scoped InnerProductSpace

section Schur

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- `M` is fixed under conjugation by every orthogonal matrix fixing `r`. -/
def Invariant (M : Matrix n n ℝ) (r : n → ℝ) : Prop :=
  ∀ Q : Matrix n n ℝ, Q * Qᵀ = 1 → Q *ᵥ r = r → Q * M * Qᵀ = M

/-- The reflection across the hyperplane orthogonal to `w`. -/
noncomputable def householder (w : n → ℝ) : Matrix n n ℝ :=
  1 - (2 / (w ⬝ᵥ w)) • vecMulVec w w

theorem householder_mulVec (w x : n → ℝ) :
    householder w *ᵥ x = x - (2 / (w ⬝ᵥ w) * (w ⬝ᵥ x)) • w := by
  simp [householder, sub_mulVec, smul_mulVec, vecMulVec_mulVec, smul_smul]

theorem householder_transpose (w : n → ℝ) : (householder w)ᵀ = householder w := by
  simp [householder, transpose_vecMulVec]

theorem householder_mul_transpose {w : n → ℝ} (hw : w ⬝ᵥ w ≠ 0) :
    householder w * (householder w)ᵀ = 1 := by
  rw [householder_transpose]
  apply Matrix.mulVec_injective
  funext x
  rw [← mulVec_mulVec, one_mulVec, householder_mulVec, householder_mulVec]
  have h : 2 / (w ⬝ᵥ w) * (w ⬝ᵥ (x - (2 / (w ⬝ᵥ w) * (w ⬝ᵥ x)) • w)) =
      -(2 / (w ⬝ᵥ w) * (w ⬝ᵥ x)) := by
    rw [dotProduct_sub, dotProduct_smul, smul_eq_mul]
    field_simp
    ring
  rw [h]
  module

omit [DecidableEq n] in
theorem dot_mulVec_symm {M : Matrix n n ℝ} (hM : Mᵀ = M) (v w : n → ℝ) :
    v ⬝ᵥ M *ᵥ w = M *ᵥ v ⬝ᵥ w := by
  rw [dotProduct_mulVec, ← mulVec_transpose, hM]

/-- Reflection step: every nonzero vector orthogonal to `r` is an eigenvector. -/
theorem eigen_of_invariant {M : Matrix n n ℝ} {r : n → ℝ} (hinv : Invariant M r)
    {w : n → ℝ} (hwr : w ⬝ᵥ r = 0) (hw : w ≠ 0) :
    M *ᵥ w = ((w ⬝ᵥ M *ᵥ w) / (w ⬝ᵥ w)) • w := by
  have hww : w ⬝ᵥ w ≠ 0 := by simpa using hw
  have hfix : householder w *ᵥ r = r := by
    rw [householder_mulVec, hwr, mul_zero, zero_smul, sub_zero]
  have h := hinv (householder w) (householder_mul_transpose hww) hfix
  have hHw : householder w *ᵥ w = -w := by
    rw [householder_mulVec, div_mul_cancel₀ _ hww]
    module
  have h2 : householder w *ᵥ (M *ᵥ (householder w *ᵥ w)) = M *ᵥ w := by
    conv_rhs => rw [← h]
    rw [householder_transpose, mulVec_mulVec, mulVec_mulVec]
  rw [hHw, mulVec_neg, mulVec_neg, householder_mulVec] at h2
  funext i
  have h3 := congrFun h2 i
  simp only [Pi.neg_apply, Pi.sub_apply, Pi.smul_apply, smul_eq_mul] at h3 ⊢
  linear_combination (-1 / 2 : ℝ) * h3

omit [DecidableEq n] in
theorem eigenvalue_eq_of_dot_ne {M : Matrix n n ℝ} (hM : Mᵀ = M) {w₁ w₂ : n → ℝ}
    {l₁ l₂ : ℝ} (h₁ : M *ᵥ w₁ = l₁ • w₁) (h₂ : M *ᵥ w₂ = l₂ • w₂)
    (hne : w₁ ⬝ᵥ w₂ ≠ 0) : l₁ = l₂ := by
  have h := dot_mulVec_symm hM w₁ w₂
  rw [h₁, h₂, dotProduct_smul, smul_dotProduct, smul_eq_mul, smul_eq_mul] at h
  exact (mul_right_cancel₀ hne h).symm

/-- All of `r^⊥` is a single eigenspace. -/
theorem exists_common_eigenvalue {M : Matrix n n ℝ} (hM : Mᵀ = M) {r : n → ℝ}
    (hinv : Invariant M r) : ∃ b : ℝ, ∀ w : n → ℝ, w ⬝ᵥ r = 0 → M *ᵥ w = b • w := by
  by_cases hex : ∃ w₀ : n → ℝ, w₀ ⬝ᵥ r = 0 ∧ w₀ ≠ 0
  · obtain ⟨w₀, h₀r, h₀⟩ := hex
    have e₀ := eigen_of_invariant hinv h₀r h₀
    refine ⟨(w₀ ⬝ᵥ M *ᵥ w₀) / (w₀ ⬝ᵥ w₀), fun w hwr => ?_⟩
    by_cases hw : w = 0
    · simp [hw]
    have e := eigen_of_invariant hinv hwr hw
    have hww : w ⬝ᵥ w ≠ 0 := by simpa using hw
    have h₀₀ : w₀ ⬝ᵥ w₀ ≠ 0 := by simpa using h₀
    by_cases hd : w ⬝ᵥ w₀ = 0
    · have h3r : (w + w₀) ⬝ᵥ r = 0 := by rw [add_dotProduct, hwr, h₀r, add_zero]
      have h3w : (w + w₀) ⬝ᵥ w ≠ 0 := by
        rw [add_dotProduct, dotProduct_comm w₀ w, hd, add_zero]
        exact hww
      have h30 : (w + w₀) ⬝ᵥ w₀ ≠ 0 := by
        rw [add_dotProduct, hd, zero_add]
        exact h₀₀
      have h3 : w + w₀ ≠ 0 := by
        intro hz
        rw [hz, zero_dotProduct] at h3w
        exact h3w rfl
      have e₃ := eigen_of_invariant hinv h3r h3
      have ha := eigenvalue_eq_of_dot_ne hM e₃ e h3w
      have hb := eigenvalue_eq_of_dot_ne hM e₃ e₀ h30
      rw [e, ← ha, hb]
    · rw [e, eigenvalue_eq_of_dot_ne hM e e₀ hd]
  · refine ⟨0, fun w hwr => ?_⟩
    have hw : w = 0 := by
      by_contra hw
      exact hex ⟨w, hwr, hw⟩
    simp [hw]

omit [DecidableEq n] in
/-- The fixed direction is itself an eigenvector with eigenvalue `rᵀ M r`. -/
theorem mulVec_fixed {M : Matrix n n ℝ} (hM : Mᵀ = M) {r : n → ℝ} (hr : r ⬝ᵥ r = 1)
    {b : ℝ} (hb : ∀ w : n → ℝ, w ⬝ᵥ r = 0 → M *ᵥ w = b • w) :
    M *ᵥ r = (r ⬝ᵥ M *ᵥ r) • r := by
  have hyr : (M *ᵥ r - (r ⬝ᵥ M *ᵥ r) • r) ⬝ᵥ r = 0 := by
    rw [sub_dotProduct, smul_dotProduct, hr, smul_eq_mul, mul_one, dotProduct_comm]
    exact sub_self _
  have hyy : (M *ᵥ r - (r ⬝ᵥ M *ᵥ r) • r) ⬝ᵥ (M *ᵥ r - (r ⬝ᵥ M *ᵥ r) • r) = 0 := by
    have h1 : (M *ᵥ r - (r ⬝ᵥ M *ᵥ r) • r) ⬝ᵥ M *ᵥ r = 0 := by
      rw [dot_mulVec_symm hM, hb _ hyr, smul_dotProduct, hyr, smul_zero]
    rw [dotProduct_sub, h1, dotProduct_smul, hyr, smul_zero, sub_zero]
  exact sub_eq_zero.mp (dotProduct_self_eq_zero.mp hyy)

/-- Schur-type structure, existence form: an invariant symmetric second moment
is `a` on the fixed direction and a single scalar `b` on its complement. -/
theorem invariant_structure_exists {M : Matrix n n ℝ} (hM : Mᵀ = M) {r : n → ℝ}
    (hr : r ⬝ᵥ r = 1) (hinv : Invariant M r) :
    ∃ b : ℝ, M = (r ⬝ᵥ M *ᵥ r) • vecMulVec r r + b • (1 - vecMulVec r r) := by
  obtain ⟨b, hb⟩ := exists_common_eigenvalue hM hinv
  refine ⟨b, ?_⟩
  have hMr := mulVec_fixed hM hr hb
  apply Matrix.mulVec_injective
  funext x
  have hw : (x - (r ⬝ᵥ x) • r) ⬝ᵥ r = 0 := by
    rw [sub_dotProduct, smul_dotProduct, hr, smul_eq_mul, mul_one, dotProduct_comm]
    exact sub_self _
  have hMx : M *ᵥ x = b • (x - (r ⬝ᵥ x) • r) + (r ⬝ᵥ x) • (r ⬝ᵥ M *ᵥ r) • r := by
    rw [← hb _ hw, ← hMr, ← mulVec_smul, ← mulVec_add]
    congr 1
    module
  rw [hMx]
  simp only [add_mulVec, smul_mulVec, sub_mulVec, one_mulVec, vecMulVec_mulVec,
    op_smul_eq_smul]
  module

/-- Consequence of the two-eigenvalue form: the trace. -/
theorem trace_of_form {M : Matrix n n ℝ} {r : n → ℝ} {a b : ℝ} (hr : r ⬝ᵥ r = 1)
    (hform : M = a • vecMulVec r r + b • (1 - vecMulVec r r)) :
    M.trace = a + (Fintype.card n - 1) * b := by
  rw [hform]
  simp [trace_add, trace_smul, trace_sub, hr]
  ring

/-- Consequence of the two-eigenvalue form: pairing with any matrix `S`. -/
theorem trace_mul_of_form {M : Matrix n n ℝ} {r : n → ℝ} {a b : ℝ}
    (hform : M = a • vecMulVec r r + b • (1 - vecMulVec r r)) (S : Matrix n n ℝ) :
    (M * S).trace = a * (r ⬝ᵥ S *ᵥ r) + b * (S.trace - r ⬝ᵥ S *ᵥ r) := by
  have h : (vecMulVec r r * S).trace = r ⬝ᵥ S *ᵥ r := by
    rw [vecMulVec_mul, trace_vecMulVec, dotProduct_mulVec, dotProduct_comm]
  rw [hform, Matrix.add_mul, Matrix.smul_mul, Matrix.smul_mul, Matrix.sub_mul,
    Matrix.one_mul, trace_add, trace_smul, trace_smul, trace_sub, h]
  simp

/-- Schur-type structure of an invariant second moment. If the symmetric matrix
`M` satisfies `Q M Qᵀ = M` for every orthogonal `Q` fixing the unit vector `r`,
and the dimension is at least two, then
`M = a • r rᵀ + b • (1 - r rᵀ)` with `a = rᵀ M r` and `b = (trace M - a) / (d - 1)`. -/
theorem invariant_structure {M : Matrix n n ℝ} (hM : Mᵀ = M) {r : n → ℝ}
    (hr : r ⬝ᵥ r = 1) (hinv : Invariant M r) (hd : 2 ≤ Fintype.card n) :
    M = (r ⬝ᵥ M *ᵥ r) • vecMulVec r r +
      ((M.trace - r ⬝ᵥ M *ᵥ r) / (Fintype.card n - 1)) • (1 - vecMulVec r r) := by
  obtain ⟨b, hb⟩ := invariant_structure_exists hM hr hinv
  have htr := trace_of_form hr hb
  have hd' : (Fintype.card n : ℝ) - 1 ≠ 0 := by
    have : (2 : ℝ) ≤ Fintype.card n := by exact_mod_cast hd
    linarith
  have hbval : (M.trace - r ⬝ᵥ M *ᵥ r) / (Fintype.card n - 1) = b := by
    rw [htr]
    field_simp
    ring
  rw [hbval]
  exact hb

/-- Trace of an invariant second moment against any matrix `S`. -/
theorem invariant_trace_mul {M : Matrix n n ℝ} (hM : Mᵀ = M) {r : n → ℝ}
    (hr : r ⬝ᵥ r = 1) (hinv : Invariant M r) (hd : 2 ≤ Fintype.card n)
    (S : Matrix n n ℝ) :
    (M * S).trace = (r ⬝ᵥ M *ᵥ r) * (r ⬝ᵥ S *ᵥ r) +
      (M.trace - r ⬝ᵥ M *ᵥ r) / (Fintype.card n - 1) * (S.trace - r ⬝ᵥ S *ᵥ r) :=
  trace_mul_of_form (invariant_structure hM hr hinv hd) S

end Schur

section Centroid

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  {Ω : Type*} [MeasurableSpace Ω]

/-- Centroid identity: when the quantization error is uncorrelated with the
reconstruction, the radial shrink `E (1 - ⟪u, q u⟫)` equals the distortion
`E ‖u - q u‖^2` for a random unit vector `u`. -/
theorem centroid_identity {μ : Measure Ω} (u : Ω → E) (q : E → E)
    (hu : ∀ᵐ ω ∂μ, ‖u ω‖ = 1)
    (hdist : Integrable (fun ω => ‖u ω - q (u ω)‖ ^ 2) μ)
    (hcross : Integrable (fun ω => inner ℝ (u ω - q (u ω)) (q (u ω))) μ)
    (hcentroid : ∫ ω, inner ℝ (u ω - q (u ω)) (q (u ω)) ∂μ = 0) :
    ∫ ω, (1 - inner ℝ (u ω) (q (u ω))) ∂μ = ∫ ω, ‖u ω - q (u ω)‖ ^ 2 ∂μ := by
  have hae : (fun ω => 1 - inner ℝ (u ω) (q (u ω))) =ᵐ[μ]
      fun ω => ‖u ω - q (u ω)‖ ^ 2 + inner ℝ (u ω - q (u ω)) (q (u ω)) := by
    filter_upwards [hu] with ω hω
    rw [norm_sub_sq_real, inner_sub_left, real_inner_self_eq_norm_sq, hω]
    ring
  rw [integral_congr_ae hae, integral_add hdist hcross, hcentroid, add_zero]

end Centroid

section Haar

variable {d : ℕ}

/-- The residual after one rotated gain-shape pass on `r`, as a function of the rotation `g`:
`rplus = r - ‖r‖ • g⁻¹ • q (g • rhat)` with `rhat = ‖r‖⁻¹ • r`. -/
noncomputable def rplus (q : Ed d → Ed d) (r : Ed d) (g : Od d) : Ed d :=
  r - ‖r‖ • (g⁻¹ • q (g • (‖r‖⁻¹ • r)))

/-- `α = E (1 - ⟪U, q U⟫)²` for `U` uniform on the sphere. -/
noncomputable def passAlpha (q : Ed d → Ed d) : ℝ :=
  ∫ u, (1 - ⟪u, q u⟫_ℝ) ^ 2 ∂(sphereUniformAmbient (Ed d))

/-- `β = E (1 - ⟪U, q U⟫)` for `U` uniform on the sphere. -/
noncomputable def passBeta (q : Ed d → Ed d) : ℝ :=
  ∫ u, (1 - ⟪u, q u⟫_ℝ) ∂(sphereUniformAmbient (Ed d))

/-- The mean of the next residual under Haar. -/
noncomputable def mean (q : Ed d → Ed d) (r : Ed d) : Ed d :=
  ∫ g, rplus q r g ∂(haarOd d)

/-- The second-moment matrix of the next residual under Haar, entrywise. -/
noncomputable def secondMoment (q : Ed d → Ed d) (r : Ed d) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => ∫ g, rplus q r g i * rplus q r g j ∂(haarOd d)

theorem od_smul_sub (g : Od d) (x y : Ed d) : g • (x - y) = g • x - g • y := by
  rw [← iso_apply, ← iso_apply, ← iso_apply, map_sub]

theorem od_smul_smul (g : Od d) (c : ℝ) (x : Ed d) : g • (c • x) = c • (g • x) := by
  rw [← iso_apply, ← iso_apply, map_smul]

theorem norm_hat {r : Ed d} (hr : r ≠ 0) : ‖‖r‖⁻¹ • r‖ = 1 := by
  rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr hr)]

theorem smul_hat {r : Ed d} (hr : r ≠ 0) : ‖r‖ • (‖r‖⁻¹ • r) = r := by
  rw [smul_smul, mul_inv_cancel₀ (norm_ne_zero_iff.mpr hr), one_smul]

/-- A rotation fixing `rhat` fixes `r`. -/
theorem smul_eq_of_smul_hat {r : Ed d} (hr : r ≠ 0) {Q : Od d}
    (hQ : Q • (‖r‖⁻¹ • r) = ‖r‖⁻¹ • r) : Q • r = r :=
  calc Q • r = Q • (‖r‖ • (‖r‖⁻¹ • r)) := by rw [smul_hat hr]
    _ = ‖r‖ • (‖r‖⁻¹ • r) := by rw [od_smul_smul, hQ]
    _ = r := smul_hat hr

/-- Equivariance: right translation of the frame by `Q⁻¹`, `Q` fixing `r`, rotates the residual
by `Q`. -/
theorem rplus_mul_inv (q : Ed d → Ed d) (r : Ed d) (g Q : Od d) (hQ : Q • r = r) :
    rplus q r (g * Q⁻¹) = Q • rplus q r g := by
  have h1 : Q⁻¹ • (‖r‖⁻¹ • r) = ‖r‖⁻¹ • r := by rw [inv_smul_eq_iff, od_smul_smul, hQ]
  simp only [rplus]
  rw [_root_.mul_inv_rev, inv_inv, mul_smul, mul_smul, h1, od_smul_sub, hQ, od_smul_smul Q ‖r‖]

theorem measurable_rplus {q : Ed d → Ed d} (hq : Measurable q) (r : Ed d) :
    Measurable (rplus q r) := by
  have h1 : Measurable fun g : Od d => g • (‖r‖⁻¹ • r) :=
    MeasurableSMul.measurable_smul_const _
  have h2 : Measurable fun g : Od d => g⁻¹ • q (g • (‖r‖⁻¹ • r)) :=
    measurable_inv.smul (hq.comp h1)
  have h3 : Measurable fun g : Od d => ‖r‖ • (g⁻¹ • q (g • (‖r‖⁻¹ • r))) :=
    h2.const_smul ‖r‖
  exact measurable_const.sub h3

/-- The residual is bounded: `‖rplus‖ ≤ ‖r‖ (1 + B)`. -/
theorem norm_rplus_le {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (g : Od d) : ‖rplus q r g‖ ≤ ‖r‖ * (1 + B) := by
  have h1 : ‖q (g • (‖r‖⁻¹ • r))‖ ≤ B := hB _ (by rw [Orthogonal.norm_smul' d, norm_hat hr])
  calc ‖rplus q r g‖ ≤ ‖r‖ + ‖‖r‖ • (g⁻¹ • q (g • (‖r‖⁻¹ • r)))‖ := norm_sub_le _ _
    _ = ‖r‖ + ‖r‖ * ‖q (g • (‖r‖⁻¹ • r))‖ := by
      rw [norm_smul, norm_norm, Orthogonal.norm_smul' d]
    _ ≤ ‖r‖ * (1 + B) := by nlinarith [norm_nonneg r]

/-- Radial component of the next residual. -/
theorem inner_rplus {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0) (g : Od d) :
    ⟪‖r‖⁻¹ • r, rplus q r g⟫_ℝ
      = ‖r‖ * (1 - ⟪g • (‖r‖⁻¹ • r), q (g • (‖r‖⁻¹ • r))⟫_ℝ) := by
  have hn : ‖r‖ ≠ 0 := norm_ne_zero_iff.mpr hr
  have h1 : ⟪‖r‖⁻¹ • r, r⟫_ℝ = ‖r‖ := by
    rw [real_inner_smul_left, real_inner_self_eq_norm_sq]
    field_simp
  have h2 : ∀ y : Ed d, ⟪‖r‖⁻¹ • r, g⁻¹ • y⟫_ℝ = ⟪g • (‖r‖⁻¹ • r), y⟫_ℝ := by
    intro y
    rw [← Orthogonal.inner_smul d g, smul_inv_smul]
  rw [rplus, inner_sub_right, real_inner_smul_right, h1, h2]
  ring

/-- Squared norm of the next residual. -/
theorem norm_sq_rplus {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0) (g : Od d) :
    ‖rplus q r g‖ ^ 2
      = ‖r‖ ^ 2 * ‖g • (‖r‖⁻¹ • r) - q (g • (‖r‖⁻¹ • r))‖ ^ 2 := by
  have h : rplus q r g = ‖r‖ • (g⁻¹ • (g • (‖r‖⁻¹ • r) - q (g • (‖r‖⁻¹ • r)))) := by
    rw [od_smul_sub, inv_smul_smul, smul_sub, smul_hat hr, rplus]
  rw [h, norm_smul, norm_norm, Orthogonal.norm_smul' d, mul_pow]

/-- Integrals of functions of `g • rhat` against Haar are integrals against the uniform law. -/
theorem integral_haar_smul_hat {r : Ed d} (hr : r ≠ 0) {F : Ed d → ℝ} (hF : Measurable F) :
    ∫ g, F (g • (‖r‖⁻¹ • r)) ∂(haarOd d) = ∫ u, F u ∂(sphereUniformAmbient (Ed d)) := by
  rw [← orbitLaw_haar d _ (norm_hat hr), orbitLaw,
    integral_map (MeasurableSMul.measurable_smul_const _).aemeasurable hF.aestronglyMeasurable]

/-- **Invariance in law.**  The law of the next residual under Haar is invariant under every
rotation `Q` fixing `r`. -/
theorem map_rplus_smul' {q : Ed d → Ed d} (hq : Measurable q) (r : Ed d) (Q : Od d)
    (hQ : Q • r = r) :
    ((haarOd d).map (rplus q r)).map (fun x => Q • x) = (haarOd d).map (rplus q r) := by
  have hcomp : (fun x : Ed d => Q • x) ∘ rplus q r = rplus q r ∘ fun g => g * Q⁻¹ := by
    funext g
    exact (rplus_mul_inv q r g Q hQ).symm
  rw [Measure.map_map (measurable_const_smul Q) (measurable_rplus hq r), hcomp,
    ← Measure.map_map (measurable_rplus hq r) (measurable_mul_const _), map_mul_right_eq_self]

/-- **Invariance in law**, with the fixed direction `rhat = r/‖r‖`. -/
theorem map_rplus_smul {q : Ed d → Ed d} (hq : Measurable q) {r : Ed d} (hr : r ≠ 0) (Q : Od d)
    (hQ : Q • (‖r‖⁻¹ • r) = ‖r‖⁻¹ • r) :
    ((haarOd d).map (rplus q r)).map (fun x => Q • x) = (haarOd d).map (rplus q r) :=
  map_rplus_smul' hq r Q (smul_eq_of_smul_hat hr hQ)

theorem integrable_rplus {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0) (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) : Integrable (rplus q r) (haarOd d) :=
  Integrable.of_bound (measurable_rplus hq r).aestronglyMeasurable _
    (ae_of_all _ (norm_rplus_le hr hB))

theorem integrable_rplus_mul {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0) (hq : Measurable q)
    {B : ℝ} (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (i j : Fin d) :
    Integrable (fun g => rplus q r g i * rplus q r g j) (haarOd d) := by
  have hm : ∀ k : Fin d, Measurable fun g => rplus q r g k := fun k =>
    (EuclideanSpace.proj k).continuous.measurable.comp (measurable_rplus hq r)
  have hb : ∀ (k : Fin d) (g : Od d), ‖rplus q r g k‖ ≤ ‖r‖ * (1 + B) := fun k g =>
    (PiLp.norm_apply_le _ k).trans (norm_rplus_le hr hB g)
  refine Integrable.of_bound ((hm i).mul (hm j)).aestronglyMeasurable
    ((‖r‖ * (1 + B)) * (‖r‖ * (1 + B))) (ae_of_all _ fun g => ?_)
  rw [norm_mul]
  exact mul_le_mul (hb i g) (hb j g) (norm_nonneg _) ((norm_nonneg _).trans (hb i g))

theorem sum_mul_eq_inner (x y : Ed d) : ∑ i, x i * y i = ⟪x, y⟫_ℝ := by
  simp [PiLp.inner_apply, mul_comm]

/-- The bilinear form of the second moment is the mixed moment of the two projections. -/
theorem dot_secondMoment {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0) (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (v w : Fin d → ℝ) :
    v ⬝ᵥ secondMoment q r *ᵥ w
      = ∫ g, (∑ i, v i * rplus q r g i) * (∑ j, w j * rplus q r g j) ∂(haarOd d) := by
  have h : ∀ g : Od d, (∑ i, v i * rplus q r g i) * (∑ j, w j * rplus q r g j)
      = ∑ i, ∑ j, v i * (rplus q r g i * rplus q r g j * w j) := fun g => by
    rw [Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
  have hint : ∀ i j : Fin d,
      Integrable (fun g => v i * (rplus q r g i * rplus q r g j * w j)) (haarOd d) := fun i j =>
    ((integrable_rplus_mul hr hq hB i j).mul_const _).const_mul _
  rw [integral_congr_ae (ae_of_all _ h),
    integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hint i j]
  simp only [dotProduct, mulVec, secondMoment, Matrix.of_apply]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [integral_finsetSum _ fun j _ => hint i j, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [integral_const_mul, integral_mul_const]

/-- The second moment is symmetric. -/
theorem secondMoment_transpose (q : Ed d → Ed d) (r : Ed d) :
    (secondMoment q r)ᵀ = secondMoment q r := by
  ext i j
  simp only [transpose_apply, secondMoment, Matrix.of_apply]
  exact integral_congr_ae (ae_of_all _ fun g => mul_comm _ _)

/-- The second moment is fixed by conjugation with every rotation fixing `r`. -/
theorem secondMoment_conj {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0) (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) (Q : Od d) (hQ : Q • r = r) :
    Q.1 * secondMoment q r * Q.1ᵀ = secondMoment q r := by
  ext i j
  have h1 : (Q.1 * secondMoment q r * Q.1ᵀ) i j
      = (fun k => Q.1 i k) ⬝ᵥ secondMoment q r *ᵥ (fun l => Q.1 j l) := by
    simp only [Matrix.mul_apply, transpose_apply, dotProduct, mulVec, Finset.sum_mul,
      Finset.mul_sum]
    rw [Finset.sum_comm]
    simp only [mul_assoc]
  have h2 : ∀ (g : Od d) (i : Fin d), ∑ k, Q.1 i k * rplus q r g k = rplus q r (g * Q⁻¹) i := by
    intro g i
    rw [rplus_mul_inv q r g Q hQ, Orthogonal.smul_def]
    rfl
  rw [h1, dot_secondMoment hr hq hB]
  simp only [h2, secondMoment, Matrix.of_apply]
  exact integral_mul_right_eq_self (μ := haarOd d) (fun g => rplus q r g i * rplus q r g j) Q⁻¹

/-- The second moment is invariant in the sense of the Schur lemma, with fixed vector `rhat`. -/
theorem invariant_secondMoment {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0) (hq : Measurable q)
    {B : ℝ} (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) :
    Invariant (secondMoment q r) (WithLp.ofLp (‖r‖⁻¹ • r)) := by
  intro Q hQQ hQr
  have hmem : Q ∈ Matrix.orthogonalGroup (Fin d) ℝ := by
    rw [Matrix.mem_orthogonalGroup_iff]
    simpa [Matrix.star_eq_conjTranspose] using hQQ
  have h0 : (⟨Q, hmem⟩ : Od d) • (‖r‖⁻¹ • r) = ‖r‖⁻¹ • r := by
    rw [Orthogonal.smul_def]
    show WithLp.toLp 2 (Q *ᵥ WithLp.ofLp (‖r‖⁻¹ • r)) = ‖r‖⁻¹ • r
    rw [hQr, WithLp.toLp_ofLp]
  exact secondMoment_conj hr hq hB ⟨Q, hmem⟩ (smul_eq_of_smul_hat hr h0)

theorem dot_hat {r : Ed d} (hr : r ≠ 0) :
    (WithLp.ofLp (‖r‖⁻¹ • r) : Fin d → ℝ) ⬝ᵥ WithLp.ofLp (‖r‖⁻¹ • r) = 1 := by
  show ∑ i, (‖r‖⁻¹ • r) i * (‖r‖⁻¹ • r) i = 1
  rw [sum_mul_eq_inner, real_inner_self_eq_norm_sq, norm_hat hr]
  norm_num

/-- The radial eigenvalue: `rhatᵀ M rhat = ‖r‖² α`. -/
theorem hat_dot_secondMoment {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0) (hq : Measurable q)
    {B : ℝ} (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) :
    (WithLp.ofLp (‖r‖⁻¹ • r) : Fin d → ℝ) ⬝ᵥ secondMoment q r *ᵥ WithLp.ofLp (‖r‖⁻¹ • r)
      = ‖r‖ ^ 2 * passAlpha q := by
  have hF : Measurable fun u : Ed d => (1 - ⟪u, q u⟫_ℝ) ^ 2 :=
    (measurable_const.sub (measurable_id.inner hq)).pow_const 2
  have key : ∀ g : Od d,
      (∑ i, (WithLp.ofLp (‖r‖⁻¹ • r) : Fin d → ℝ) i * rplus q r g i)
        * (∑ j, (WithLp.ofLp (‖r‖⁻¹ • r) : Fin d → ℝ) j * rplus q r g j)
      = ‖r‖ ^ 2 * (1 - ⟪g • (‖r‖⁻¹ • r), q (g • (‖r‖⁻¹ • r))⟫_ℝ) ^ 2 := fun g => by
    rw [sum_mul_eq_inner, inner_rplus hr]
    ring
  rw [dot_secondMoment hr hq hB, integral_congr_ae (ae_of_all _ key), integral_const_mul,
    integral_haar_smul_hat hr hF, passAlpha]

/-- The trace: `tr M = ‖r‖² ρ²`. -/
theorem trace_secondMoment {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0) (hq : Measurable q)
    {B : ℝ} (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) :
    (secondMoment q r).trace = ‖r‖ ^ 2 * stageDistortion q (sphereUniformAmbient (Ed d)) := by
  have hF : Measurable fun u : Ed d => ‖u - q u‖ ^ 2 :=
    (measurable_id.sub hq).norm.pow_const 2
  have key : ∀ g : Od d, ∑ i, rplus q r g i * rplus q r g i
      = ‖r‖ ^ 2 * ‖g • (‖r‖⁻¹ • r) - q (g • (‖r‖⁻¹ • r))‖ ^ 2 := fun g => by
    rw [sum_mul_eq_inner, real_inner_self_eq_norm_sq, norm_sq_rplus hr]
  simp only [Matrix.trace, Matrix.diag, secondMoment, Matrix.of_apply]
  rw [← integral_finsetSum _ fun i _ => integrable_rplus_mul hr hq hB i i,
    integral_congr_ae (ae_of_all _ key), integral_const_mul, integral_haar_smul_hat hr hF,
    stageDistortion]

/-- **Lemma 2, second moment.**  For `d ≥ 2`, `r ≠ 0`, and a measurable stage bounded on the
unit sphere, `E rplus rplusᵀ = ‖r‖² (α • rhat rhatᵀ + ((ρ² - α)/(d-1)) • (1 - rhat rhatᵀ))`. -/
theorem secondMoment_eq (hd : 2 ≤ d) {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0)
    (hq : Measurable q) {B : ℝ} (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) :
    secondMoment q r = ‖r‖ ^ 2 •
      (passAlpha q • vecMulVec (WithLp.ofLp (‖r‖⁻¹ • r)) (WithLp.ofLp (‖r‖⁻¹ • r))
        + ((stageDistortion q (sphereUniformAmbient (Ed d)) - passAlpha q) / ((d : ℝ) - 1)) •
          (1 - vecMulVec (WithLp.ofLp (‖r‖⁻¹ • r)) (WithLp.ofLp (‖r‖⁻¹ • r)))) := by
  have h := invariant_structure (secondMoment_transpose q r) (dot_hat hr)
    (invariant_secondMoment hr hq hB) (by simpa using hd)
  rw [hat_dot_secondMoment hr hq hB, trace_secondMoment hr hq hB, Fintype.card_fin] at h
  have e : (‖r‖ ^ 2 * stageDistortion q (sphereUniformAmbient (Ed d)) - ‖r‖ ^ 2 * passAlpha q)
      / ((d : ℝ) - 1)
      = ‖r‖ ^ 2 * ((stageDistortion q (sphereUniformAmbient (Ed d)) - passAlpha q)
        / ((d : ℝ) - 1)) := by ring
  rw [e] at h
  rw [smul_add, smul_smul, smul_smul]
  exact h

/-- Every linear isometry of `ℝ^d` is the action of an element of `O(d)`. -/
theorem exists_od_eq_iso (L : Ed d ≃ₗᵢ[ℝ] Ed d) : ∃ g : Od d, ∀ x : Ed d, g • x = L x := by
  let b := EuclideanSpace.basisFun (Fin d) ℝ
  refine ⟨⟨LinearMap.toMatrix b.toBasis b.toBasis L.toLinearEquiv,
    L.toMatrix_mem_unitaryGroup b b⟩, fun x => ?_⟩
  show Matrix.toLpLin 2 2 (LinearMap.toMatrix b.toBasis b.toBasis L.toLinearEquiv) x = L x
  rw [Matrix.toLpLin_eq_toLin, ← EuclideanSpace.basisFun_toBasis, Matrix.toLin_toMatrix]
  rfl

/-- The mean is fixed by every rotation fixing `r`. -/
theorem smul_mean (q : Ed d → Ed d) (r : Ed d) (Q : Od d) (hQ : Q • r = r) :
    Q • mean q r = mean q r :=
  calc Q • mean q r = iso d Q (∫ g, rplus q r g ∂(haarOd d)) := (iso_apply d Q _).symm
    _ = ∫ g, iso d Q (rplus q r g) ∂(haarOd d) :=
      ((iso d Q).toContinuousLinearEquiv.integral_comp_comm (rplus q r)).symm
    _ = ∫ g, rplus q r (g * Q⁻¹) ∂(haarOd d) := by
      simp only [iso_apply, rplus_mul_inv q r _ Q hQ]
    _ = mean q r := integral_mul_right_eq_self (μ := haarOd d) (rplus q r) Q⁻¹

/-- The radial component of the mean. -/
theorem inner_mean {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0) (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) : ⟪‖r‖⁻¹ • r, mean q r⟫_ℝ = ‖r‖ * passBeta q := by
  have hF : Measurable fun u : Ed d => 1 - ⟪u, q u⟫_ℝ :=
    measurable_const.sub (measurable_id.inner hq)
  rw [mean, ← integral_inner (integrable_rplus hr hq hB),
    integral_congr_ae (ae_of_all _ fun g => inner_rplus hr g), integral_const_mul,
    integral_haar_smul_hat hr hF, passBeta]

/-- **Lemma 2, mean.**  For `r ≠ 0` and a measurable stage bounded on the unit sphere,
`E rplus = β • r`.  The component of the mean orthogonal to `r` is reversed by the reflection
across its own orthogonal hyperplane, which fixes `r`, hence it vanishes. -/
theorem mean_eq {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0) (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) : mean q r = passBeta q • r := by
  have hn : ‖r‖ ≠ 0 := norm_ne_zero_iff.mpr hr
  have hc := inner_mean hr hq hB
  set m := mean q r with hm
  set w : Ed d := m - (‖r‖ * passBeta q) • (‖r‖⁻¹ • r) with hw
  have hwr : ⟪w, r⟫_ℝ = 0 := by
    have h1 : ⟪m, r⟫_ℝ = ‖r‖ * (‖r‖ * passBeta q) := by
      rw [← hc, real_inner_smul_left, real_inner_comm]
      field_simp
    rw [hw, inner_sub_left, real_inner_smul_left, real_inner_smul_left,
      real_inner_self_eq_norm_sq, h1]
    field_simp
    ring
  obtain ⟨g, hg⟩ := exists_od_eq_iso ((ℝ ∙ w)ᗮ.reflection)
  have hLr : (ℝ ∙ w)ᗮ.reflection r = r :=
    Submodule.reflection_mem_subspace_eq_self
      (Submodule.mem_orthogonal_singleton_iff_inner_right.mpr hwr)
  have hLw : (ℝ ∙ w)ᗮ.reflection w = -w :=
    Submodule.reflection_orthogonalComplement_singleton_eq_neg w
  have hLm : (ℝ ∙ w)ᗮ.reflection m = m := by
    rw [← hg]
    exact smul_mean q r g (by rw [hg]; exact hLr)
  have hLw' : (ℝ ∙ w)ᗮ.reflection w = w := by
    conv_lhs => rw [hw]
    rw [map_sub, map_smul, map_smul, hLm, hLr]
  have h2 : (2 : ℝ) • w = 0 := by
    rw [two_smul]
    nth_rewrite 1 [hLw'.symm.trans hLw]
    exact neg_add_cancel w
  have hw0 : w = 0 := (smul_eq_zero.mp h2).resolve_left two_ne_zero
  rw [sub_eq_zero.mp hw0, smul_smul]
  congr 1
  field_simp

/-- **Centroid identity.**  If the quantisation error is uncorrelated with the reconstruction
under the uniform law, the radial shrink `β` equals the distortion `ρ²`. -/
theorem passBeta_eq_stageDistortion (hd : 0 < d) {q : Ed d → Ed d} (hq : Measurable q) {B : ℝ}
    (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B)
    (hcent : ∫ u, ⟪u - q u, q u⟫_ℝ ∂(sphereUniformAmbient (Ed d)) = 0) :
    passBeta q = stageDistortion q (sphereUniformAmbient (Ed d)) := by
  haveI : Nontrivial (Ed d) :=
    ⟨⟨EuclideanSpace.single ⟨0, hd⟩ (1 : ℝ), 0, fun h => by
      have := congrArg (fun v : Ed d => ‖v‖) h
      simp at this⟩⟩
  have hae := sphereUniformAmbient_ae_norm (E := Ed d)
  have hdist : Integrable (fun u : Ed d => ‖u - q u‖ ^ 2) (sphereUniformAmbient (Ed d)) := by
    refine Integrable.of_bound ((measurable_id.sub hq).norm.pow_const 2).aestronglyMeasurable
      ((1 + B) ^ 2) ?_
    filter_upwards [hae] with u hu
    have h1 : ‖u - q u‖ ≤ 1 + B := (norm_sub_le _ _).trans (by rw [hu]; linarith [hB u hu])
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    exact pow_le_pow_left₀ (norm_nonneg _) h1 2
  have hcross : Integrable (fun u : Ed d => ⟪u - q u, q u⟫_ℝ)
      (sphereUniformAmbient (Ed d)) := by
    refine Integrable.of_bound ((measurable_id.sub hq).inner hq).aestronglyMeasurable
      ((1 + B) * B) ?_
    filter_upwards [hae] with u hu
    have h1 : ‖u - q u‖ ≤ 1 + B := (norm_sub_le _ _).trans (by rw [hu]; linarith [hB u hu])
    exact (norm_inner_le_norm _ _).trans
      (mul_le_mul h1 (hB u hu) (norm_nonneg _) ((norm_nonneg _).trans h1))
  exact centroid_identity (μ := sphereUniformAmbient (Ed d)) (fun u => u) q hae hdist hcross hcent

/-- **Lemma 2 (error covariance of a Haar pass).**  For `d ≥ 2`, `r ≠ 0`, `g` Haar on `O(d)` and
a measurable stage `q` bounded on the unit sphere, the next residual
`rplus = r - ‖r‖ • g⁻¹ • q (g • rhat)` has mean `β • r` and second moment
`‖r‖² (α • rhat rhatᵀ + ((ρ² - α)/(d-1)) • (1 - rhat rhatᵀ))`. -/
theorem pass_covariance (hd : 2 ≤ d) {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0)
    (hq : Measurable q) {B : ℝ} (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B) :
    mean q r = passBeta q • r ∧
    secondMoment q r = ‖r‖ ^ 2 •
      (passAlpha q • vecMulVec (WithLp.ofLp (‖r‖⁻¹ • r)) (WithLp.ofLp (‖r‖⁻¹ • r))
        + ((stageDistortion q (sphereUniformAmbient (Ed d)) - passAlpha q) / ((d : ℝ) - 1)) •
          (1 - vecMulVec (WithLp.ofLp (‖r‖⁻¹ • r)) (WithLp.ofLp (‖r‖⁻¹ • r)))) :=
  ⟨mean_eq hr hq hB, secondMoment_eq hd hr hq hB⟩

/-- **Lemma 2 under the centroid condition.**  If moreover `E ⟪U - q U, q U⟫ = 0` for `U` uniform
on the sphere, the mean is `ρ² • r`. -/
theorem pass_covariance_centroid (hd : 2 ≤ d) {q : Ed d → Ed d} {r : Ed d} (hr : r ≠ 0)
    (hq : Measurable q) {B : ℝ} (hB : ∀ u, ‖u‖ = 1 → ‖q u‖ ≤ B)
    (hcent : ∫ u, ⟪u - q u, q u⟫_ℝ ∂(sphereUniformAmbient (Ed d)) = 0) :
    mean q r = stageDistortion q (sphereUniformAmbient (Ed d)) • r ∧
    secondMoment q r = ‖r‖ ^ 2 •
      (passAlpha q • vecMulVec (WithLp.ofLp (‖r‖⁻¹ • r)) (WithLp.ofLp (‖r‖⁻¹ • r))
        + ((stageDistortion q (sphereUniformAmbient (Ed d)) - passAlpha q) / ((d : ℝ) - 1)) •
          (1 - vecMulVec (WithLp.ofLp (‖r‖⁻¹ • r)) (WithLp.ofLp (‖r‖⁻¹ • r)))) := by
  obtain ⟨h1, h2⟩ := pass_covariance hd hr hq hB
  exact ⟨by rw [h1, passBeta_eq_stageDistortion (by omega) hq hB hcent], h2⟩

end Haar

end PassCov
