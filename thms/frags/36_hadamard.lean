/-! ## §36  The deployed rotation: what Proposition 2 becomes under `H D_ε`

The paper deploys `R = H D_ε`, a fixed flat orthogonal matrix (every entry `±1/√d`, the
Walsh–Hadamard matrix in practice, constructed in §38) times a uniformly random diagonal sign matrix, with fresh
signs at every pass.  This is a finite family inside `O(d)`, Theorem 1 is false for it, and
nothing here certifies the paper's empirical defence.  This section proves what is true.

Rademacher moments are finite sums: `E(∑ aⱼεⱼ)² = ∑ aⱼ²` and
`E(∑ aⱼεⱼ)⁴ = 3(∑ aⱼ²)² − 2∑ aⱼ⁴` (`sum_radSum_sq`, `sum_radSum_pow_four`), so every
coordinate of `H D_ε x` has second moment `‖x‖²/d`, the value of `f_d`, and fourth moment
`(3‖x‖⁴ − 2∑ xⱼ⁴)/d²` (`signAvg_coord_sq`, `signAvg_coord_pow_four`); against the sphere's
`3/(d(d+2))`, proved in §37, the gap is governed by the spikiness `∑ xⱼ⁴` of the input
direction.  In the stream with fresh uniform sign patterns, freezing the fresh pattern
gives the exact per-pass identity `E‖r_{p+1}‖² = E[‖r_p‖² · ρ_H²(r̂_p)]`
(`integral_norm_resid_sq_succ_had`) with `ρ_H²(v)` the sign-averaged pass distortion of the
direction `v` (`hadRate`), hence geometric decay with a rate sandwiched between the extreme
values of `ρ_H²` on the sphere (`integral_norm_resid_sq_le_had`, `le_integral_norm_resid_sq_had`).
For a coordinatewise codebook `ρ_H²` is the quadrature of the scalar distortion against the
Hadamard coordinate laws (`hadRate_blockStage`).  All of it lives in the same scheme model as
§1 and §12; only the law of the rotation differs. -/

namespace Hadamard

open Orthogonal


/-! ### Sign vectors and Rademacher moments -/

/-- A sign pattern: one bit per coordinate. -/
abbrev Signs (d : ℕ) := Fin d → Bool

/-- `±1` from a bit. -/
def sgn (b : Bool) : ℝ := if b then 1 else -1

theorem sgn_true : sgn true = 1 := rfl

theorem sgn_false : sgn false = -1 := rfl

theorem sgn_mul_self (b : Bool) : sgn b * sgn b = 1 := by cases b <;> simp [sgn]

theorem sgn_sq (b : Bool) : sgn b ^ 2 = 1 := by rw [sq, sgn_mul_self]

theorem card_signs (d : ℕ) : Fintype.card (Signs d) = 2 ^ d := by
  simp [Fintype.card_fun]

/-- The average over the `2^d` sign patterns. -/
noncomputable def signAvg {d : ℕ} (f : Signs d → ℝ) : ℝ := (∑ s, f s) / 2 ^ d

theorem signAvg_add {d : ℕ} (f g : Signs d → ℝ) :
    signAvg (fun s => f s + g s) = signAvg f + signAvg g := by
  unfold signAvg; rw [Finset.sum_add_distrib, add_div]

theorem signAvg_const_mul {d : ℕ} (c : ℝ) (f : Signs d → ℝ) :
    signAvg (fun s => c * f s) = c * signAvg f := by
  unfold signAvg; rw [← Finset.mul_sum, mul_div_assoc]

theorem signAvg_sum {d : ℕ} {ι : Type*} (t : Finset ι) (f : ι → Signs d → ℝ) :
    signAvg (fun s => ∑ i ∈ t, f i s) = ∑ i ∈ t, signAvg (f i) := by
  unfold signAvg; rw [Finset.sum_comm, Finset.sum_div]

theorem signAvg_mono {d : ℕ} {f g : Signs d → ℝ} (h : ∀ s, f s ≤ g s) : signAvg f ≤ signAvg g :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun s _ => h s) (by positivity)

theorem signAvg_const {d : ℕ} (c : ℝ) : signAvg (fun _ : Signs d => c) = c := by
  unfold signAvg
  rw [Finset.sum_const, Finset.card_univ, card_signs, nsmul_eq_mul]
  field_simp
  try (push_cast; ring)

/-- The Rademacher sum `∑ⱼ aⱼ εⱼ`. -/
def radSum {n : ℕ} (a : Fin n → ℝ) (s : Signs n) : ℝ := ∑ j, a j * sgn (s j)

theorem radSum_cons {n : ℕ} (a : Fin (n + 1) → ℝ) (b : Bool) (s : Signs n) :
    radSum a (Fin.cons b s) = a 0 * sgn b + radSum (Fin.tail a) s := by
  unfold radSum
  rw [Fin.sum_univ_succ]
  simp [Fin.tail]

/-- Splitting a sum over `Signs (n+1)` along the first bit. -/
theorem sum_signs_succ {n : ℕ} (F : Signs (n + 1) → ℝ) :
    ∑ s, F s = ∑ b : Bool, ∑ s : Signs n, F (Fin.cons b s) := by
  rw [← Fintype.sum_prod_type']
  exact (Fintype.sum_equiv (Fin.consEquiv fun _ => Bool) (fun p => F (Fin.cons p.1 p.2)) F
    fun p => rfl).symm

/-- `E (∑ aⱼ εⱼ)² = ∑ aⱼ²`, as a finite sum. -/
theorem sum_radSum_sq {n : ℕ} (a : Fin n → ℝ) :
    ∑ s : Signs n, radSum a s ^ 2 = 2 ^ n * ∑ j, a j ^ 2 := by
  induction n with
  | zero => simp [radSum]
  | succ n ih =>
      rw [sum_signs_succ, Fin.sum_univ_succ]
      simp only [radSum_cons]
      rw [Fintype.sum_bool]
      simp only [sgn_true, sgn_false]
      have : ∑ s : Signs n, ((a 0 * 1 + radSum (Fin.tail a) s) ^ 2
          + (a 0 * -1 + radSum (Fin.tail a) s) ^ 2)
          = ∑ s : Signs n, (2 * radSum (Fin.tail a) s ^ 2 + 2 * a 0 ^ 2) := by
        refine Finset.sum_congr rfl fun s _ => ?_; ring
      rw [← Finset.sum_add_distrib, this, Finset.sum_add_distrib, ← Finset.mul_sum, ih,
        Finset.sum_const, Finset.card_univ, card_signs, nsmul_eq_mul]
      simp only [Fin.tail]
      push_cast
      ring

/-- `E (∑ aⱼ εⱼ)⁴ = 3 (∑ aⱼ²)² − 2 ∑ aⱼ⁴`, as a finite sum. -/
theorem sum_radSum_pow_four {n : ℕ} (a : Fin n → ℝ) :
    ∑ s : Signs n, radSum a s ^ 4
      = 2 ^ n * (3 * (∑ j, a j ^ 2) ^ 2 - 2 * ∑ j, a j ^ 4) := by
  induction n with
  | zero => simp [radSum]
  | succ n ih =>
      rw [sum_signs_succ, Fin.sum_univ_succ, Fin.sum_univ_succ]
      simp only [radSum_cons]
      rw [Fintype.sum_bool]
      simp only [sgn_true, sgn_false]
      have : ∑ s : Signs n, ((a 0 * 1 + radSum (Fin.tail a) s) ^ 4
          + (a 0 * -1 + radSum (Fin.tail a) s) ^ 4)
          = ∑ s : Signs n, (2 * radSum (Fin.tail a) s ^ 4
              + 12 * a 0 ^ 2 * radSum (Fin.tail a) s ^ 2 + 2 * a 0 ^ 4) := by
        refine Finset.sum_congr rfl fun s _ => ?_; ring
      rw [← Finset.sum_add_distrib, this, Finset.sum_add_distrib, Finset.sum_add_distrib,
        ← Finset.mul_sum, ← Finset.mul_sum, ih, sum_radSum_sq, Finset.sum_const, Finset.card_univ,
        card_signs, nsmul_eq_mul]
      simp only [Fin.tail]
      push_cast
      ring

/-! ### The randomised flat rotation `H D_ε` -/

variable {d : ℕ}

/-- `H` is flat: every entry has magnitude `1/√d`.  The Walsh–Hadamard matrix is the example. -/
def IsFlat (H : Od d) : Prop := ∀ i j, ((H : Matrix (Fin d) (Fin d) ℝ) i j) ^ 2 = 1 / d

/-- The diagonal sign matrix `D_ε`. -/
noncomputable def signMat (s : Signs d) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.diagonal fun j => sgn (s j)

theorem signMat_mem (s : Signs d) : signMat s ∈ Matrix.orthogonalGroup (Fin d) ℝ := by
  rw [Matrix.mem_orthogonalGroup_iff, signMat, Matrix.diagonal_transpose,
    Matrix.diagonal_mul_diagonal]
  rw [show (fun i => sgn (s i) * sgn (s i)) = fun _ => (1 : ℝ) from funext fun i => sgn_mul_self _,
    Matrix.diagonal_one]

/-- `D_ε` as an element of `O(d)`. -/
noncomputable def signRot (s : Signs d) : Od d := ⟨signMat s, signMat_mem s⟩

/-- The deployed rotation at sign pattern `ε`: `H D_ε`. -/
noncomputable def hadRot (H : Od d) (s : Signs d) : Od d := H * signRot s

theorem hadRot_apply (H : Od d) (s : Signs d) (x : Ed d) (i : Fin d) :
    (hadRot H s • x) i = ∑ j, (H : Matrix (Fin d) (Fin d) ℝ) i j * (sgn (s j) * x j) := by
  rw [smul_def, Matrix.toEuclideanLin_apply]
  change (Matrix.mulVec ((H : Matrix _ _ ℝ) * signMat s) (WithLp.ofLp x)) i = _
  rw [← Matrix.mulVec_mulVec]
  show ∑ j, (H : Matrix (Fin d) (Fin d) ℝ) i j * (Matrix.mulVec (signMat s) (WithLp.ofLp x)) j = _
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [signMat, Matrix.mulVec_diagonal]
  try rfl

theorem hadRot_apply_eq_radSum (H : Od d) (s : Signs d) (x : Ed d) (i : Fin d) :
    (hadRot H s • x) i = radSum (fun j => (H : Matrix (Fin d) (Fin d) ℝ) i j * x j) s := by
  rw [hadRot_apply, radSum]
  refine Finset.sum_congr rfl fun j _ => ?_; ring

theorem norm_sq_eq_sum_sq (x : Ed d) : ‖x‖ ^ 2 = ∑ j, x j ^ 2 := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (Finset.sum_nonneg fun j _ => sq_nonneg _)]
  simp [Real.norm_eq_abs, sq_abs]

/-- **Second moment of a coordinate**: `E (H D_ε x)ᵢ² = ‖x‖²/d`, the value of `f_d`. -/
theorem signAvg_coord_sq {H : Od d} (hH : IsFlat H) (x : Ed d) (i : Fin d) :
    signAvg (fun s => (hadRot H s • x) i ^ 2) = ‖x‖ ^ 2 / d := by
  simp_rw [hadRot_apply_eq_radSum]
  unfold signAvg
  rw [sum_radSum_sq, norm_sq_eq_sum_sq]
  simp_rw [mul_pow, hH i]
  rw [← Finset.mul_sum]
  try field_simp

/-- **Fourth moment of a coordinate**: `E (H D_ε x)ᵢ⁴ = (3 ‖x‖⁴ − 2 ∑ⱼ xⱼ⁴)/d²`.  On the sphere
the fourth moment of `f_d` is `3/(d(d+2))` (§37), so the gap is governed by `∑ⱼ xⱼ⁴`, the
spikiness of the input direction: it vanishes at order `1/d³` for spread inputs and is of
order `1/d²` for a coordinate vector. -/
theorem signAvg_coord_pow_four {H : Od d} (hH : IsFlat H) (x : Ed d) (i : Fin d) :
    signAvg (fun s => (hadRot H s • x) i ^ 4)
      = (3 * ‖x‖ ^ 4 - 2 * ∑ j, x j ^ 4) / (d : ℝ) ^ 2 := by
  simp_rw [hadRot_apply_eq_radSum]
  unfold signAvg
  rw [sum_radSum_pow_four]
  have h2 : ∑ j, ((H : Matrix (Fin d) (Fin d) ℝ) i j * x j) ^ 2 = ‖x‖ ^ 2 / d := by
    rw [norm_sq_eq_sum_sq]; simp_rw [mul_pow, hH i]; rw [← Finset.mul_sum]; try ring
  have h4 : ∑ j, ((H : Matrix (Fin d) (Fin d) ℝ) i j * x j) ^ 4 = (∑ j, x j ^ 4) / (d : ℝ) ^ 2 := by
    have : ∀ j, ((H : Matrix (Fin d) (Fin d) ℝ) i j * x j) ^ 4
        = (1 / (d : ℝ)) ^ 2 * x j ^ 4 := fun j => by
      rw [mul_pow, show (4 : ℕ) = 2 * 2 from rfl, pow_mul, hH i]
      try ring
    simp_rw [this]; rw [← Finset.mul_sum]; try ring
  rw [h2, h4]
  rcases Nat.eq_zero_or_pos d with hd | hd
  · subst hd; simp
  · have : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
    field_simp
    try ring

/-! ### The Hadamard stream: what Proposition 2 becomes -/

/-- One pass with a fixed isometry `L`: `r ↦ r − ‖r‖ L⁻¹ q(L r/‖r‖)`. -/
noncomputable def hadStep (L : Ed d ≃ₗᵢ[ℝ] Ed d) (q : Ed d → Ed d) (r : Ed d) : Ed d :=
  r - ‖r‖ • L.symm (q (‖r‖⁻¹ • L r))

theorem measurable_hadStep (L : Ed d ≃ₗᵢ[ℝ] Ed d) {q : Ed d → Ed d} (hq : Measurable q) :
    Measurable (hadStep L q) := by
  unfold hadStep
  exact measurable_id.sub (measurable_norm.smul (L.symm.continuous.measurable.comp
    (hq.comp (measurable_norm.inv.smul (L.continuous.measurable.comp measurable_id)))))

/-- The Hadamard scheme: pass `p` rotates by `H D_{s p}`. -/
noncomputable def hadScheme (H : Od d) (q : Ed d → Ed d) (s : ℕ → Signs d) : Scheme (Ed d) where
  rot p := iso d (hadRot H (s p))
  q := q

/-- **The Hadamard pass distortion** of a direction `v`: the squared quantisation error of
`H D_ε v`, averaged over the sign pattern.  Under Haar this would be the constant `ρ²`; under
Hadamard it depends on `v`. -/
noncomputable def hadRate (H : Od d) (q : Ed d → Ed d) (v : Ed d) : ℝ :=
  signAvg fun s => ‖hadRot H s • v - q (hadRot H s • v)‖ ^ 2

/-- `r_p` as a function of the past `(w, s_0, …, s_{p−1})`. -/
noncomputable def hadPast (H : Od d) (q : Ed d → Ed d) : ∀ p : ℕ, Ed d × (Fin p → Signs d) → Ed d
  | 0, x => x.1
  | p + 1, x => hadStep (iso d (hadRot H (x.2 (Fin.last p)))) q
      (hadPast H q p (x.1, fun i => x.2 i.castSucc))

theorem resid_hadScheme_eq_hadPast (H : Od d) (q : Ed d → Ed d) (w : Ed d) (s : ℕ → Signs d)
    (p : ℕ) : resid (hadScheme H q s) w p = hadPast H q p (w, fun i : Fin p => s i) := by
  induction p with
  | zero => rfl
  | succ p ih =>
      have hcast : (fun i : Fin p => (fun j : Fin (p + 1) => s j) i.castSucc)
          = fun i : Fin p => s i := by funext i; simp
      rw [resid_succ, unitVec, ih, hadPast]
      simp only [hcast, Fin.val_last]
      rfl

theorem measurable_hadPast_left (H : Od d) {q : Ed d → Ed d} (hq : Measurable q) :
    ∀ (p : ℕ) (t : Fin p → Signs d), Measurable fun w => hadPast H q p (w, t)
  | 0, _ => measurable_id
  | p + 1, t => by
      show Measurable fun w => hadStep (iso d (hadRot H (t (Fin.last p)))) q
        (hadPast H q p (w, fun i => t i.castSucc))
      exact (measurable_hadStep _ hq).comp (measurable_hadPast_left H hq p _)

theorem measurable_hadPast (H : Od d) {q : Ed d → Ed d} (hq : Measurable q) (p : ℕ) :
    Measurable (hadPast H q p) :=
  measurable_from_prod_countable_left (measurable_hadPast_left H hq p)

/-- The unit direction at pass `p` is the fresh rotation applied to `dir r_p`. -/
theorem unitVec_hadScheme (H : Od d) (q : Ed d → Ed d) (w : Ed d) (s : ℕ → Signs d) (p : ℕ) :
    unitVec (hadScheme H q s) w p = hadRot H (s p) • dir (resid (hadScheme H q s) w p) := by
  rw [unitVec, dir, show (hadScheme H q s).rot p = iso d (hadRot H (s p)) from rfl, iso_apply,
    ← smul_comm_real (G := Od d) (ι := iso d) (iso_apply d)]

theorem norm_resid_succ_sq_had (H : Od d) (q : Ed d → Ed d) (w : Ed d) (s : ℕ → Signs d)
    (p : ℕ) :
    ‖resid (hadScheme H q s) w (p + 1)‖ ^ 2
      = ‖resid (hadScheme H q s) w p‖ ^ 2 *
          ‖hadRot H (s p) • dir (resid (hadScheme H q s) w p)
            - q (hadRot H (s p) • dir (resid (hadScheme H q s) w p))‖ ^ 2 := by
  rw [norm_resid_succ_sq, unitVec_hadScheme]; rfl

/-- The quantisation error of a direction is bounded by `1 + R` for a codebook in `B(0, R)`. -/
theorem norm_sub_q_le {q : Ed d → Ed d} {C : Set (Ed d)} (hq : IsNearest C q) {R : ℝ}
    (hR : ∀ c ∈ C, ‖c‖ ≤ R) {v : Ed d} (hv : ‖v‖ ≤ 1) : ‖v - q v‖ ≤ 1 + R := by
  obtain ⟨hmem, -⟩ := hq v
  calc ‖v - q v‖ ≤ ‖v‖ + ‖q v‖ := norm_sub_le _ _
    _ ≤ 1 + R := add_le_add hv (hR _ hmem)

theorem hadRate_le {H : Od d} {q : Ed d → Ed d} {C : Set (Ed d)} (hq : IsNearest C q) {R : ℝ}
    (hR : ∀ c ∈ C, ‖c‖ ≤ R) (hR0 : 0 ≤ R) {v : Ed d} (hv : ‖v‖ ≤ 1) :
    hadRate H q v ≤ (1 + R) ^ 2 := by
  unfold hadRate
  calc signAvg (fun s => ‖hadRot H s • v - q (hadRot H s • v)‖ ^ 2)
      ≤ signAvg (fun _ : Signs d => (1 + R) ^ 2) := signAvg_mono fun s => by
        refine pow_le_pow_left₀ (norm_nonneg _) (norm_sub_q_le hq hR ?_) 2
        rw [norm_smul']; exact hv
    _ = (1 + R) ^ 2 := signAvg_const _

theorem hadRate_nonneg (H : Od d) (q : Ed d → Ed d) (v : Ed d) : 0 ≤ hadRate H q v :=
  div_nonneg (Finset.sum_nonneg fun s _ => sq_nonneg _) (by positivity)

section Stream

variable [NeZero d]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {H : Od d} {q : Ed d → Ed d} {W : Ω → Ed d} {s : Ω → ℕ → Signs d}

instance : Nontrivial (Ed d) :=
  ⟨⟨EuclideanSpace.single (0 : Fin d) (1 : ℝ), 0, by
    intro h
    have := congrArg (fun v : Ed d => v 0) h
    simp at this⟩⟩

theorem measurable_hadRot_smul (H : Od d) (t : Signs d) :
    Measurable fun v : Ed d => hadRot H t • v := by
  rw [show (fun v : Ed d => hadRot H t • v) = ⇑(iso d (hadRot H t)) from
    funext fun v => (iso_apply _ _ _).symm]
  exact (iso d _).continuous.measurable

/-- The past at pass `p`, as a random variable. -/
def hadY (W : Ω → Ed d) (s : Ω → ℕ → Signs d) (p : ℕ) (ω : Ω) : Ed d × (Fin p → Signs d) :=
  (W ω, fun i : Fin p => s ω i)

theorem measurable_hadY (hW : Measurable W) (hs : ∀ p, Measurable fun ω => s ω p) (p : ℕ) :
    Measurable (hadY W s p) :=
  hW.prodMk (measurable_pi_lambda _ fun i => hs i)

theorem resid_eq_hadPast_hadY (H : Od d) (q : Ed d → Ed d) (W : Ω → Ed d) (s : Ω → ℕ → Signs d)
    (p : ℕ) (ω : Ω) :
    resid (hadScheme H q (s ω)) (W ω) p = hadPast H q p (hadY W s p ω) :=
  resid_hadScheme_eq_hadPast H q (W ω) (s ω) p

theorem measurable_resid_sq_had (hq : Measurable q) (hW : Measurable W)
    (hs : ∀ p, Measurable fun ω => s ω p) (p : ℕ) :
    Measurable fun ω => ‖resid (hadScheme H q (s ω)) (W ω) p‖ ^ 2 := by
  simp_rw [resid_eq_hadPast_hadY]
  exact (((measurable_hadPast H hq p).comp (measurable_hadY hW hs p)).norm).pow_const 2

/-- Integrability of `‖r_p‖²` for a bounded codebook, from that of `‖w‖²`. -/
theorem integrable_resid_sq_had (hq : Measurable q) {C : Set (Ed d)} (hqC : IsNearest C q)
    {R : ℝ} (hR : ∀ c ∈ C, ‖c‖ ≤ R) (hW : Measurable W) (hs : ∀ p, Measurable fun ω => s ω p)
    (hW2 : Integrable (fun ω => ‖W ω‖ ^ 2) μ) :
    ∀ p, Integrable (fun ω => ‖resid (hadScheme H q (s ω)) (W ω) p‖ ^ 2) μ
  | 0 => hW2
  | p + 1 => by
      have ih := integrable_resid_sq_had hq hqC hR hW hs hW2 p
      have hm := measurable_resid_sq_had (H := H) hq hW hs (p + 1)
      refine (ih.const_mul ((1 + R) ^ 2)).mono' hm.aestronglyMeasurable
        (Filter.Eventually.of_forall fun ω => ?_)
      rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _), norm_resid_succ_sq_had]
      have hv : ‖hadRot H (s ω p) • dir (resid (hadScheme H q (s ω)) (W ω) p)‖ ≤ 1 := by
        rw [norm_smul']; exact norm_dir_le_one _
      calc _ ≤ ‖resid (hadScheme H q (s ω)) (W ω) p‖ ^ 2 * (1 + R) ^ 2 :=
            mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (norm_nonneg _)
              (norm_sub_q_le hqC hR hv) 2) (sq_nonneg _)
        _ = _ := mul_comm _ _

/-- **Proposition 2 for the Hadamard stream: the conditional form.**  With fresh, uniformly
random sign patterns, `E‖r_{p+1}‖² = E[‖r_p‖² · hadRate(r̂_p)]`.  This is the exact identity
that replaces `E‖r_{p+1}‖² = ρ² E‖r_p‖²`: the rate is the Hadamard pass distortion of the
current residual direction, not a constant. -/
theorem integral_norm_resid_sq_succ_had (hq : Measurable q) {C : Set (Ed d)}
    (hqC : IsNearest C q) {R : ℝ} (hR : ∀ c ∈ C, ‖c‖ ≤ R) (hW : Measurable W)
    (hs : ∀ p, Measurable fun ω => s ω p)
    (hlaw : ∀ p (t : Signs d), μ.real {ω | s ω p = t} = 1 / 2 ^ d)
    (hfresh : ∀ p, IndepFun (fun ω => s ω p) (hadY W s p) μ)
    (hW2 : Integrable (fun ω => ‖W ω‖ ^ 2) μ) (p : ℕ) :
    ∫ ω, ‖resid (hadScheme H q (s ω)) (W ω) (p + 1)‖ ^ 2 ∂μ
      = ∫ ω, ‖resid (hadScheme H q (s ω)) (W ω) p‖ ^ 2 *
          hadRate H q (dir (resid (hadScheme H q (s ω)) (W ω) p)) ∂μ := by
  classical
  have hR0 : 0 ≤ R := by
    obtain ⟨hmem, -⟩ := hqC 0
    exact (norm_nonneg _).trans (hR _ hmem)
  -- the integrand as `G (s_p) (Y)`
  set G : Signs d → Ed d × (Fin p → Signs d) → ℝ := fun t y =>
    ‖hadPast H q p y‖ ^ 2 *
      ‖hadRot H t • dir (hadPast H q p y) - q (hadRot H t • dir (hadPast H q p y))‖ ^ 2 with hG
  have hGmeas : ∀ t, Measurable (G t) := fun t => by
    have h1 := measurable_hadPast H hq p
    have hd : Measurable (dir : Ed d → Ed d) := measurable_dir
    have hrot := measurable_hadRot_smul H t
    exact (h1.norm.pow_const 2).mul
      ((((hrot.comp (hd.comp h1)).sub (hq.comp (hrot.comp (hd.comp h1)))).norm).pow_const 2)
  have hGbound : ∀ t y, |G t y| ≤ (1 + R) ^ 2 * ‖hadPast H q p y‖ ^ 2 := fun t y => by
    rw [hG]; simp only
    rw [abs_of_nonneg (by positivity), mul_comm]
    refine mul_le_mul_of_nonneg_right (pow_le_pow_left₀ (norm_nonneg _)
      (norm_sub_q_le hqC hR ?_) 2) (sq_nonneg _)
    rw [norm_smul']; exact norm_dir_le_one _
  have hrp : Integrable (fun ω => ‖hadPast H q p (hadY W s p ω)‖ ^ 2) μ := by
    have := integrable_resid_sq_had (H := H) hq hqC hR hW hs hW2 p
    simpa only [resid_eq_hadPast_hadY] using this
  have hGint : ∀ t, Integrable (fun ω => G t (hadY W s p ω)) μ := fun t =>
    (hrp.const_mul ((1 + R) ^ 2)).mono'
      ((hGmeas t).comp (measurable_hadY hW hs p)).aestronglyMeasurable
      (Filter.Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs]; exact hGbound t _)
  set φ : Signs d → Signs d → ℝ := fun t u => if u = t then 1 else 0 with hφ
  have hpt : ∀ ω, ‖resid (hadScheme H q (s ω)) (W ω) (p + 1)‖ ^ 2
      = ∑ t : Signs d, φ t (s ω p) * G t (hadY W s p ω) := by
    intro ω
    rw [norm_resid_succ_sq_had, resid_eq_hadPast_hadY]
    have : ∀ t : Signs d, φ t (s ω p) * G t (hadY W s p ω)
        = if s ω p = t then G t (hadY W s p ω) else 0 := fun t => by
      rw [hφ]; simp only
      by_cases h : s ω p = t
      · rw [if_pos h, one_mul, if_pos h]
      · rw [if_neg h, zero_mul, if_neg h]
    simp_rw [this]
    rw [Finset.sum_ite_eq]
    simp [hG]
  have hφmeas : ∀ t, Measurable fun ω => φ t (s ω p) := fun t =>
    (measurable_of_countable (φ t)).comp (hs p)
  have hφint : ∀ t, ∫ ω, φ t (s ω p) ∂μ = 1 / 2 ^ d := by
    intro t
    have hmeasset : MeasurableSet {ω | s ω p = t} := (hs p) (measurableSet_singleton t)
    have : (fun ω => φ t (s ω p)) = {ω | s ω p = t}.indicator (1 : Ω → ℝ) := by
      funext ω; rw [hφ]; simp only
      by_cases h : s ω p = t
      · rw [if_pos h, Set.indicator_of_mem (by exact h)]; rfl
      · rw [if_neg h, Set.indicator_of_notMem (by exact h)]
    rw [this, integral_indicator_one hmeasset, hlaw]
  have hφG : ∀ t, Integrable (fun ω => φ t (s ω p) * G t (hadY W s p ω)) μ := fun t => by
    have hm : Measurable fun ω => φ t (s ω p) * G t (hadY W s p ω) :=
      (hφmeas t).mul ((hGmeas t).comp (measurable_hadY hW hs p))
    refine (hGint t).mono' hm.aestronglyMeasurable (Filter.Eventually.of_forall fun ω => ?_)
    have hG0 : 0 ≤ G t (hadY W s p ω) := by rw [hG]; simp only; positivity
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg hG0]
    refine mul_le_of_le_one_left hG0 ?_
    rw [hφ]; simp only
    split_ifs <;> simp
  simp_rw [hpt]
  rw [integral_finsetSum _ fun t _ => hφG t]
  have hterm : ∀ t : Signs d, ∫ ω, φ t (s ω p) * G t (hadY W s p ω) ∂μ
      = (1 / 2 ^ d) * ∫ ω, G t (hadY W s p ω) ∂μ := by
    intro t
    have hind : IndepFun (fun ω => φ t (s ω p)) (fun ω => G t (hadY W s p ω)) μ :=
      (hfresh p).comp (measurable_of_countable (φ t)) (hGmeas t)
    have hmul : ∫ ω, (fun ω => φ t (s ω p)) ω * (fun ω => G t (hadY W s p ω)) ω ∂μ
        = (∫ ω, φ t (s ω p) ∂μ) * ∫ ω, G t (hadY W s p ω) ∂μ :=
      hind.integral_mul_eq_mul_integral (hφmeas t).aestronglyMeasurable
        (hGint t).aestronglyMeasurable
    simp only at hmul
    rw [hmul, hφint]
  simp_rw [hterm]
  rw [← Finset.mul_sum, ← integral_finsetSum _ fun t _ => hGint t, ← integral_const_mul]
  refine integral_congr_ae (Filter.Eventually.of_forall fun ω => ?_)
  simp only [hG, hadRate, signAvg, resid_eq_hadPast_hadY]
  rw [Finset.sum_div, Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun t _ => ?_
  ring


theorem measurable_hadRate (H : Od d) {q : Ed d → Ed d} (hq : Measurable q) :
    Measurable (hadRate H q) := by
  unfold hadRate signAvg
  refine Measurable.div_const (Finset.measurable_sum _ fun t _ => ?_) _
  have hrot := measurable_hadRot_smul H t
  exact ((hrot.sub (hq.comp hrot)).norm).pow_const 2

theorem integrable_resid_sq_mul_hadRate (hq : Measurable q) {C : Set (Ed d)}
    (hqC : IsNearest C q) {R : ℝ} (hR : ∀ c ∈ C, ‖c‖ ≤ R) (hW : Measurable W)
    (hs : ∀ p, Measurable fun ω => s ω p) (hW2 : Integrable (fun ω => ‖W ω‖ ^ 2) μ) (p : ℕ) :
    Integrable (fun ω => ‖resid (hadScheme H q (s ω)) (W ω) p‖ ^ 2 *
      hadRate H q (dir (resid (hadScheme H q (s ω)) (W ω) p))) μ := by
  have hR0 : 0 ≤ R := by
    obtain ⟨hmem, -⟩ := hqC 0
    exact (norm_nonneg _).trans (hR _ hmem)
  have hm1 := measurable_resid_sq_had (H := H) hq hW hs p
  have hm2 : Measurable fun ω => hadRate H q (dir (resid (hadScheme H q (s ω)) (W ω) p)) := by
    simp_rw [resid_eq_hadPast_hadY]
    exact (measurable_hadRate H hq).comp (measurable_dir.comp
      ((measurable_hadPast H hq p).comp (measurable_hadY hW hs p)))
  refine ((integrable_resid_sq_had (H := H) hq hqC hR hW hs hW2 p).const_mul ((1 + R) ^ 2)).mono'
    (hm1.mul hm2).aestronglyMeasurable (Filter.Eventually.of_forall fun ω => ?_)
  rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (sq_nonneg _) (hadRate_nonneg _ _ _)), mul_comm]
  exact mul_le_mul_of_nonneg_right (hadRate_le hqC hR hR0 (norm_dir_le_one _)) (sq_nonneg _)

/-- **The sandwich.**  If the Hadamard pass distortion of every unit direction lies in
`[ρmin, ρmax]`, then `ρmin^P E‖w‖² ≤ E‖r_P‖² ≤ ρmax^P E‖w‖²`: geometric decay with an input-dependent
rate, which is what survives of Proposition 2 for the deployed rotation. -/
theorem integral_norm_resid_sq_le_had (hq : Measurable q) {C : Set (Ed d)}
    (hqC : IsNearest C q) {R : ℝ} (hR : ∀ c ∈ C, ‖c‖ ≤ R) (hW : Measurable W)
    (hs : ∀ p, Measurable fun ω => s ω p)
    (hlaw : ∀ p (t : Signs d), μ.real {ω | s ω p = t} = 1 / 2 ^ d)
    (hfresh : ∀ p, IndepFun (fun ω => s ω p) (hadY W s p) μ)
    (hW2 : Integrable (fun ω => ‖W ω‖ ^ 2) μ) {ρmax : ℝ} (hρ0 : 0 ≤ ρmax)
    (hρ : ∀ v : Ed d, ‖v‖ = 1 → hadRate H q v ≤ ρmax) :
    ∀ P, ∫ ω, ‖resid (hadScheme H q (s ω)) (W ω) P‖ ^ 2 ∂μ ≤ ρmax ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ
  | 0 => by simp [resid]
  | P + 1 => by
      have ih := integral_norm_resid_sq_le_had hq hqC hR hW hs hlaw hfresh hW2 hρ0 hρ P
      rw [integral_norm_resid_sq_succ_had hq hqC hR hW hs hlaw hfresh hW2 P]
      have hpt : ∀ ω, ‖resid (hadScheme H q (s ω)) (W ω) P‖ ^ 2 *
          hadRate H q (dir (resid (hadScheme H q (s ω)) (W ω) P))
          ≤ ρmax * ‖resid (hadScheme H q (s ω)) (W ω) P‖ ^ 2 := fun ω => by
        rcases eq_or_ne (resid (hadScheme H q (s ω)) (W ω) P) 0 with h | h
        · rw [h]; simp
        · rw [mul_comm]
          exact mul_le_mul_of_nonneg_right (hρ _ (norm_dir_of_ne_zero h)) (sq_nonneg _)
      calc ∫ ω, ‖resid (hadScheme H q (s ω)) (W ω) P‖ ^ 2 *
            hadRate H q (dir (resid (hadScheme H q (s ω)) (W ω) P)) ∂μ
          ≤ ∫ ω, ρmax * ‖resid (hadScheme H q (s ω)) (W ω) P‖ ^ 2 ∂μ :=
            integral_mono (integrable_resid_sq_mul_hadRate hq hqC hR hW hs hW2 P)
              ((integrable_resid_sq_had hq hqC hR hW hs hW2 P).const_mul _) hpt
        _ = ρmax * ∫ ω, ‖resid (hadScheme H q (s ω)) (W ω) P‖ ^ 2 ∂μ := integral_const_mul _ _
        _ ≤ ρmax * (ρmax ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ) := mul_le_mul_of_nonneg_left ih hρ0
        _ = ρmax ^ (P + 1) * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by ring

theorem le_integral_norm_resid_sq_had (hq : Measurable q) {C : Set (Ed d)}
    (hqC : IsNearest C q) {R : ℝ} (hR : ∀ c ∈ C, ‖c‖ ≤ R) (hW : Measurable W)
    (hs : ∀ p, Measurable fun ω => s ω p)
    (hlaw : ∀ p (t : Signs d), μ.real {ω | s ω p = t} = 1 / 2 ^ d)
    (hfresh : ∀ p, IndepFun (fun ω => s ω p) (hadY W s p) μ)
    (hW2 : Integrable (fun ω => ‖W ω‖ ^ 2) μ) {ρmin : ℝ} (hρ0 : 0 ≤ ρmin)
    (hρ : ∀ v : Ed d, ‖v‖ = 1 → ρmin ≤ hadRate H q v) :
    ∀ P, ρmin ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ ≤ ∫ ω, ‖resid (hadScheme H q (s ω)) (W ω) P‖ ^ 2 ∂μ
  | 0 => by simp [resid]
  | P + 1 => by
      have ih := le_integral_norm_resid_sq_had hq hqC hR hW hs hlaw hfresh hW2 hρ0 hρ P
      rw [integral_norm_resid_sq_succ_had hq hqC hR hW hs hlaw hfresh hW2 P]
      have hpt : ∀ ω, ρmin * ‖resid (hadScheme H q (s ω)) (W ω) P‖ ^ 2
          ≤ ‖resid (hadScheme H q (s ω)) (W ω) P‖ ^ 2 *
            hadRate H q (dir (resid (hadScheme H q (s ω)) (W ω) P)) := fun ω => by
        rcases eq_or_ne (resid (hadScheme H q (s ω)) (W ω) P) 0 with h | h
        · rw [h]; simp
        · rw [mul_comm]
          exact mul_le_mul_of_nonneg_left (hρ _ (norm_dir_of_ne_zero h)) (sq_nonneg _)
      calc ρmin ^ (P + 1) * ∫ ω, ‖W ω‖ ^ 2 ∂μ
          = ρmin * (ρmin ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ) := by ring
        _ ≤ ρmin * ∫ ω, ‖resid (hadScheme H q (s ω)) (W ω) P‖ ^ 2 ∂μ :=
            mul_le_mul_of_nonneg_left ih hρ0
        _ = ∫ ω, ρmin * ‖resid (hadScheme H q (s ω)) (W ω) P‖ ^ 2 ∂μ := (integral_const_mul _ _).symm
        _ ≤ _ := integral_mono ((integrable_resid_sq_had hq hqC hR hW hs hW2 P).const_mul _)
            (integrable_resid_sq_mul_hadRate hq hqC hR hW hs hW2 P) hpt

end Stream

/-- **The coordinatewise codebook**: the Hadamard pass distortion is the quadrature of the
scalar distortion `(t − cb t)²` against the Hadamard coordinate laws, one per coordinate. -/
theorem hadRate_blockStage (H : Od d) (cb : ℝ → ℝ) (v : Ed d) :
    hadRate H (blockStage (m := d) (F := ℝ) cb) v
      = ∑ i, signAvg fun s => ((hadRot H s • v) i - cb ((hadRot H s • v) i)) ^ 2 := by
  unfold hadRate
  rw [← signAvg_sum]
  congr 1; funext s
  rw [norm_sq_eq_sum_sq]
  rfl

end Hadamard
