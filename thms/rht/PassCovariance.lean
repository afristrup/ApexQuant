import Mathlib

open MeasureTheory Matrix

namespace ApexQuant.PassCovariance

section OnePass

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Residual after one rotated gain-shape pass on `s • rhat`: with `u = R rhat`,
`rplus = s • rhat - s • R⁻¹ (q u)`. -/
def residual (R : E ≃ₗᵢ[ℝ] E) (q : E → E) (s : ℝ) (rhat : E) : E :=
  s • rhat - s • R.symm (q (R rhat))

/-- Radial component of the next residual. -/
theorem inner_residual (R : E ≃ₗᵢ[ℝ] E) (q : E → E) (s : ℝ) {rhat : E}
    (hr : ‖rhat‖ = 1) :
    inner ℝ rhat (residual R q s rhat) = s * (1 - inner ℝ (R rhat) (q (R rhat))) := by
  rw [residual, inner_sub_right, inner_smul_right, inner_smul_right,
    real_inner_self_eq_norm_sq, hr, ← R.inner_map_map rhat (R.symm _),
    R.apply_symm_apply]
  ring

/-- Squared norm of the next residual. -/
theorem norm_sq_residual (R : E ≃ₗᵢ[ℝ] E) (q : E → E) (s : ℝ) (rhat : E) :
    ‖residual R q s rhat‖ ^ 2 = s ^ 2 * ‖R rhat - q (R rhat)‖ ^ 2 := by
  have h : residual R q s rhat = s • R.symm (R rhat - q (R rhat)) := by
    rw [residual, map_sub, R.symm_apply_apply, smul_sub]
  rw [h, norm_smul, R.symm.norm_map, mul_pow, Real.norm_eq_abs, sq_abs]

/-- Equivariance: precomposing the frame with an isometry `Q` that fixes `rhat`
leaves `u = R rhat` unchanged and rotates the residual by `Q`. -/
theorem frame_apply_of_fix (R Q : E ≃ₗᵢ[ℝ] E) {rhat : E} (hQ : Q rhat = rhat) :
    (Q.symm.trans R) rhat = R rhat := by
  have h : Q.symm rhat = rhat := by rw [← hQ, Q.symm_apply_apply, hQ]
  simp [h]

theorem residual_equivariant (R Q : E ≃ₗᵢ[ℝ] E) (q : E → E) (s : ℝ) {rhat : E}
    (hQ : Q rhat = rhat) :
    residual (Q.symm.trans R) q s rhat = Q (residual R q s rhat) := by
  have h : Q.symm rhat = rhat := by rw [← hQ, Q.symm_apply_apply, hQ]
  simp [residual, h, hQ]

end OnePass

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

end ApexQuant.PassCovariance
