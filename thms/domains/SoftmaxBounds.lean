import Mathlib

open Finset

namespace ApexQuant.SoftmaxBounds

/-! # Concrete softmax perturbation bounds

Everything here is for the real softmax on a finite vocabulary, with no
Lipschitz hypothesis: the Lipschitz facts are proved.
-/

section Definitions

variable {V : Type*} [Fintype V]

/-- The real softmax on a finite vocabulary. -/
noncomputable def softmax (z : V → ℝ) : V → ℝ :=
  fun v => Real.exp (z v) / ∑ u, Real.exp (z u)

/-- Total variation distance. -/
noncomputable def TV (p q : V → ℝ) : ℝ :=
  (1 / 2 : ℝ) * ∑ v, |p v - q v|

/-- Finite categorical KL divergence. -/
noncomputable def KL (p q : V → ℝ) : ℝ :=
  ∑ v, p v * Real.log (p v / q v)

end Definitions

section Softmax

variable {V : Type*} [Fintype V]

theorem partition_pos [Nonempty V] (z : V → ℝ) :
    0 < ∑ u, Real.exp (z u) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

theorem softmax_pos [Nonempty V] (z : V → ℝ) (v : V) : 0 < softmax z v :=
  div_pos (Real.exp_pos _) (partition_pos z)

theorem softmax_sum_one [Nonempty V] (z : V → ℝ) : ∑ v, softmax z v = 1 := by
  unfold softmax
  rw [← Finset.sum_div]
  exact div_self (partition_pos z).ne'

omit [Fintype V] in
/-- A sup-norm bound on a nonempty index set forces a nonnegative radius. -/
theorem delta_nonneg [Nonempty V] {h : V → ℝ} {δ : ℝ}
    (hh : ∀ v, |h v| ≤ δ) : 0 ≤ δ :=
  (abs_nonneg _).trans (hh (Classical.arbitrary V))

omit [Fintype V] in
/-- Partial partition sums grow by at most `e^δ` under a `δ` logit shift. -/
theorem sum_exp_shift_le {z h : V → ℝ} {δ : ℝ} (hh : ∀ v, |h v| ≤ δ)
    (S : Finset V) :
    ∑ u ∈ S, Real.exp ((z + h) u) ≤ Real.exp δ * ∑ u ∈ S, Real.exp (z u) := by
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro u _
  rw [Pi.add_apply, Real.exp_add, mul_comm]
  exact mul_le_mul_of_nonneg_right
    (Real.exp_le_exp.mpr (le_of_abs_le (hh u))) (Real.exp_pos _).le

omit [Fintype V] in
/-- Partial partition sums shrink by at most `e^{-δ}` under a `δ` logit shift. -/
theorem le_sum_exp_shift {z h : V → ℝ} {δ : ℝ} (hh : ∀ v, |h v| ≤ δ)
    (S : Finset V) :
    ∑ u ∈ S, Real.exp (z u) ≤ Real.exp δ * ∑ u ∈ S, Real.exp ((z + h) u) := by
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro u _
  have he : Real.exp (z u) = Real.exp (-h u) * Real.exp ((z + h) u) := by
    rw [← Real.exp_add, Pi.add_apply]
    ring_nf
  rw [he]
  exact mul_le_mul_of_nonneg_right
    (Real.exp_le_exp.mpr (by linarith [(abs_le.mp (hh u)).1])) (Real.exp_pos _).le

/-- Item 2: the log-probability ratio moves by at most `2 δ`. -/
theorem log_ratio_le [Nonempty V] {z h : V → ℝ} {δ : ℝ}
    (hh : ∀ v, |h v| ≤ δ) (v : V) :
    |Real.log (softmax z v / softmax (z + h) v)| ≤ 2 * δ := by
  have hZ := partition_pos z
  have hZ' := partition_pos (z + h)
  have e : softmax z v / softmax (z + h) v =
      Real.exp (-h v) *
        ((∑ u, Real.exp ((z + h) u)) / ∑ u, Real.exp (z u)) := by
    unfold softmax
    rw [Pi.add_apply, Real.exp_add, Real.exp_neg]
    have := (Real.exp_pos (z v)).ne'
    have := (Real.exp_pos (h v)).ne'
    field_simp
  rw [e, Real.log_mul (Real.exp_pos _).ne' (div_pos hZ' hZ).ne', Real.log_exp,
    Real.log_div hZ'.ne' hZ.ne']
  have h1 : Real.log (∑ u, Real.exp ((z + h) u)) ≤
      δ + Real.log (∑ u, Real.exp (z u)) := by
    have := Real.log_le_log hZ' (sum_exp_shift_le (z := z) hh Finset.univ)
    rwa [Real.log_mul (Real.exp_pos _).ne' hZ.ne', Real.log_exp] at this
  have h2 : Real.log (∑ u, Real.exp (z u)) ≤
      δ + Real.log (∑ u, Real.exp ((z + h) u)) := by
    have := Real.log_le_log hZ (le_sum_exp_shift (z := z) hh Finset.univ)
    rwa [Real.log_mul (Real.exp_pos _).ne' hZ'.ne', Real.log_exp] at this
  have h3 := abs_le.mp (hh v)
  rw [abs_le]
  constructor <;> linarith [h3.1, h3.2]

end Softmax

section Divergences

variable {V : Type*} [Fintype V]

/-- Gibbs inequality for finite vectors of equal mass. -/
theorem kl_nonneg {p q : V → ℝ} (hp : ∀ v, 0 ≤ p v) (hq : ∀ v, 0 < q v)
    (hsum : ∑ v, p v = ∑ v, q v) : 0 ≤ KL p q := by
  have hterm : ∀ v, p v - q v ≤ p v * Real.log (p v / q v) := by
    intro v
    rcases (hp v).eq_or_lt with h0 | hpos
    · rw [← h0]
      simpa using (hq v).le
    · have hlog := Real.log_le_sub_one_of_pos (div_pos (hq v) hpos)
      rw [Real.log_div (hq v).ne' hpos.ne'] at hlog
      rw [Real.log_div hpos.ne' (hq v).ne']
      have h2 := mul_le_mul_of_nonneg_left hlog hpos.le
      have h3 : p v * (q v / p v - 1) = q v - p v := by
        field_simp
      have h4 : p v * (Real.log (p v) - Real.log (q v)) =
          -(p v * (Real.log (q v) - Real.log (p v))) := by ring
      linarith
  have h0 : ∑ v, (p v - q v) = 0 := by
    rw [Finset.sum_sub_distrib, hsum, sub_self]
  unfold KL
  rw [← h0]
  exact Finset.sum_le_sum (fun v _ => hterm v)

/-- Total variation between probability vectors is at most one. -/
theorem tv_le_one {p q : V → ℝ} (hp : ∀ v, 0 ≤ p v) (hq : ∀ v, 0 ≤ q v)
    (hps : ∑ v, p v = 1) (hqs : ∑ v, q v = 1) : TV p q ≤ 1 := by
  unfold TV
  have hle : ∑ v, |p v - q v| ≤ ∑ v, (p v + q v) := by
    apply Finset.sum_le_sum
    intro v _
    rw [abs_le]
    constructor <;> linarith [hp v, hq v]
  rw [Finset.sum_add_distrib, hps, hqs] at hle
  linarith

theorem tv_nonneg (p q : V → ℝ) : 0 ≤ TV p q := by
  unfold TV
  have : 0 ≤ ∑ v, |p v - q v| := Finset.sum_nonneg (fun v _ => abs_nonneg _)
  linarith

/-- For equal-mass vectors, TV is the mass excess on the set `{q ≤ p}`. -/
theorem tv_eq_sum_filter {p q : V → ℝ} (hsum : ∑ v, p v = ∑ v, q v) :
    TV p q = ∑ v ∈ Finset.univ.filter (fun v => q v ≤ p v), (p v - q v) := by
  have h1 := Finset.sum_filter_add_sum_filter_not Finset.univ
    (fun v => q v ≤ p v) (fun v => |p v - q v|)
  have h2 := Finset.sum_filter_add_sum_filter_not Finset.univ
    (fun v => q v ≤ p v) (fun v => p v - q v)
  have h3 : ∑ v, (p v - q v) = 0 := by
    rw [Finset.sum_sub_distrib, hsum, sub_self]
  have h4 : ∑ v ∈ Finset.univ.filter (fun v => q v ≤ p v), |p v - q v| =
      ∑ v ∈ Finset.univ.filter (fun v => q v ≤ p v), (p v - q v) :=
    Finset.sum_congr rfl (fun v hv =>
      abs_of_nonneg (sub_nonneg.mpr (Finset.mem_filter.mp hv).2))
  have h5 : ∑ v ∈ Finset.univ.filter (fun v => ¬ q v ≤ p v), |p v - q v| =
      -∑ v ∈ Finset.univ.filter (fun v => ¬ q v ≤ p v), (p v - q v) := by
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl (fun v hv =>
      abs_of_neg (sub_neg.mpr (not_le.mp (Finset.mem_filter.mp hv).2)))
  unfold TV
  linarith

end Divergences

section Scalar

/-- Two-cell mass gap: if cell masses move by factors in `[1/s, s]`, the
normalized mass of one cell moves by at most `(s - 1) / (s + 1)`. -/
theorem frac_gap {x y x' y' s : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hxy : 0 < x + y) (hx'0 : 0 ≤ x') (hy'0 : 0 ≤ y') (hxy' : 0 < x' + y')
    (hs : 1 ≤ s) (hx' : x ≤ s * x') (hy' : y' ≤ s * y) :
    x / (x + y) - x' / (x' + y') ≤ (s - 1) / (s + 1) := by
  have hs1 : 0 < s + 1 := by linarith
  have hc : 0 ≤ 2 * s * y + (s - 1) * x :=
    add_nonneg (mul_nonneg (by linarith) hy) (mul_nonneg (by linarith) hx)
  have hR := mul_le_mul_of_nonneg_right hx' hc
  have key : y' * (2 * x - (s - 1) * y) ≤
      x' * (2 * s * y + (s - 1) * x) := by
    rcases le_or_gt (2 * x - (s - 1) * y) 0 with hneg | hneg
    · have h1 := mul_nonpos_of_nonneg_of_nonpos hy'0 hneg
      have h2 := mul_nonneg hx'0 hc
      linarith
    · have hL := mul_le_mul_of_nonneg_right hy' hneg.le
      have hsq : 0 ≤ (s - 1) * (x - s * y) ^ 2 :=
        mul_nonneg (by linarith) (sq_nonneg _)
      have h1 := mul_le_mul_of_nonneg_left hL (by linarith : (0 : ℝ) ≤ s)
      have h3 : s * (y' * (2 * x - (s - 1) * y)) ≤
          s * (x' * (2 * s * y + (s - 1) * x)) := by
        nlinarith [h1, hR, hsq]
      exact le_of_mul_le_mul_left h3 (by linarith)
  rw [div_sub_div _ _ hxy.ne' hxy'.ne',
    div_le_div_iff₀ (mul_pos hxy hxy') hs1]
  nlinarith [key]

/-- `tanh (δ / 2) ≤ δ / 2`, in exponential form. -/
theorem exp_ratio_le_half {δ : ℝ} (hδ : 0 ≤ δ) :
    (Real.exp δ - 1) / (Real.exp δ + 1) ≤ δ / 2 := by
  have hderiv : ∀ x : ℝ, HasDerivAt
      (fun x : ℝ => x * (Real.exp x + 1) - 2 * (Real.exp x - 1))
      (1 + (x - 1) * Real.exp x) x := by
    intro x
    have h := ((hasDerivAt_id x).mul ((Real.hasDerivAt_exp x).add_const 1)).sub
      (((Real.hasDerivAt_exp x).sub_const 1).const_mul 2)
    convert h using 1 <;> first | rfl | (simp only [id]; ring)
  have hmono : Monotone
      (fun x : ℝ => x * (Real.exp x + 1) - 2 * (Real.exp x - 1)) := by
    apply monotone_of_deriv_nonneg (fun x => (hderiv x).differentiableAt)
    intro x
    rw [(hderiv x).deriv]
    have h1 := Real.add_one_le_exp (-x)
    have h2 := mul_le_mul_of_nonneg_right h1 (Real.exp_pos x).le
    rw [← Real.exp_add, neg_add_cancel, Real.exp_zero] at h2
    linarith
  have h0 := hmono hδ
  simp only [Real.exp_zero] at h0
  rw [div_le_div_iff₀ (by linarith [Real.exp_pos δ]) (by norm_num)]
  linarith

/-- The exponential form above is `tanh (δ / 2)`. -/
theorem exp_ratio_eq_tanh (δ : ℝ) :
    (Real.exp δ - 1) / (Real.exp δ + 1) = Real.tanh (δ / 2) := by
  rw [Real.tanh_eq, Real.exp_neg]
  have hpos := Real.exp_pos (δ / 2)
  have hsq : Real.exp δ = Real.exp (δ / 2) * Real.exp (δ / 2) := by
    rw [← Real.exp_add]
    ring_nf
  rw [hsq]
  field_simp

end Scalar

section SoftmaxDivergence

variable {V : Type*} [Fintype V] [Nonempty V]

/-- Item 3, lower half: KL between softmaxes is nonnegative. -/
theorem kl_softmax_nonneg (z h : V → ℝ) :
    0 ≤ KL (softmax z) (softmax (z + h)) :=
  kl_nonneg (fun v => (softmax_pos z v).le) (softmax_pos (z + h))
    (by rw [softmax_sum_one, softmax_sum_one])

/-- Item 3: a sup-norm logit shift of `δ` costs at most `2 δ` nats of KL. -/
theorem kl_le_two_delta {z h : V → ℝ} {δ : ℝ} (hh : ∀ v, |h v| ≤ δ) :
    KL (softmax z) (softmax (z + h)) ≤ 2 * δ := by
  unfold KL
  calc
    ∑ v, softmax z v * Real.log (softmax z v / softmax (z + h) v) ≤
        ∑ v, softmax z v * (2 * δ) := by
      apply Finset.sum_le_sum
      intro v _
      exact mul_le_mul_of_nonneg_left
        ((le_abs_self _).trans (log_ratio_le hh v)) (softmax_pos z v).le
    _ = 2 * δ := by
      rw [← Finset.sum_mul, softmax_sum_one, one_mul]

/-- Item 4, sharp form: TV between softmaxes at sup-norm logit distance `δ`
is at most `tanh (δ / 2) = (e^δ - 1) / (e^δ + 1)`. -/
theorem tv_le_exp_ratio {z h : V → ℝ} {δ : ℝ} (hh : ∀ v, |h v| ≤ δ) :
    TV (softmax z) (softmax (z + h)) ≤
      (Real.exp δ - 1) / (Real.exp δ + 1) := by
  classical
  have hδ : 0 ≤ δ := delta_nonneg hh
  rw [tv_eq_sum_filter (by rw [softmax_sum_one, softmax_sum_one])]
  set A := Finset.univ.filter
    (fun v => softmax (z + h) v ≤ softmax z v) with hA
  have hsplit : ∀ w : V → ℝ,
      ∑ u, Real.exp (w u) =
        ∑ u ∈ A, Real.exp (w u) + ∑ u ∈ Aᶜ, Real.exp (w u) :=
    fun w => (Finset.sum_add_sum_compl A _).symm
  have hnn : ∀ (w : V → ℝ) (S : Finset V), 0 ≤ ∑ u ∈ S, Real.exp (w u) :=
    fun w S => Finset.sum_nonneg (fun u _ => (Real.exp_pos _).le)
  have hZ := partition_pos z
  have hZ' := partition_pos (z + h)
  rw [hsplit z] at hZ
  rw [hsplit (z + h)] at hZ'
  have hrew : ∑ v ∈ A, (softmax z v - softmax (z + h) v) =
      (∑ u ∈ A, Real.exp (z u)) /
          (∑ u ∈ A, Real.exp (z u) + ∑ u ∈ Aᶜ, Real.exp (z u)) -
        (∑ u ∈ A, Real.exp ((z + h) u)) /
          (∑ u ∈ A, Real.exp ((z + h) u) +
            ∑ u ∈ Aᶜ, Real.exp ((z + h) u)) := by
    rw [Finset.sum_sub_distrib]
    unfold softmax
    rw [← Finset.sum_div, ← Finset.sum_div, hsplit z, hsplit (z + h)]
  rw [hrew]
  exact frac_gap (hnn z A) (hnn z Aᶜ) hZ (hnn (z + h) A) (hnn (z + h) Aᶜ) hZ'
    (Real.one_le_exp hδ) (le_sum_exp_shift hh A) (sum_exp_shift_le hh Aᶜ)

/-- Item 4, sharp form stated with `tanh`. -/
theorem tv_le_tanh {z h : V → ℝ} {δ : ℝ} (hh : ∀ v, |h v| ≤ δ) :
    TV (softmax z) (softmax (z + h)) ≤ Real.tanh (δ / 2) := by
  rw [← exp_ratio_eq_tanh]
  exact tv_le_exp_ratio hh

/-- Item 4, linear form with the sharp constant `1 / 2`. -/
theorem tv_le_half_delta {z h : V → ℝ} {δ : ℝ} (hh : ∀ v, |h v| ≤ δ) :
    TV (softmax z) (softmax (z + h)) ≤ δ / 2 :=
  (tv_le_exp_ratio hh).trans (exp_ratio_le_half (delta_nonneg hh))

/-- Item 4 as requested: `TV ≤ δ`. -/
theorem tv_le_delta {z h : V → ℝ} {δ : ℝ} (hh : ∀ v, |h v| ≤ δ) :
    TV (softmax z) (softmax (z + h)) ≤ δ := by
  have := tv_le_half_delta (z := z) hh
  linarith [delta_nonneg hh]

/-- TV between any two softmaxes is at most one. -/
theorem tv_softmax_le_one (z w : V → ℝ) : TV (softmax z) (softmax w) ≤ 1 :=
  tv_le_one (fun v => (softmax_pos z v).le) (fun v => (softmax_pos w v).le)
    (softmax_sum_one z) (softmax_sum_one w)

/-- The paper's claim: `TV ≤ min 1 δ`. -/
theorem tv_le_min_one_delta {z h : V → ℝ} {δ : ℝ} (hh : ∀ v, |h v| ≤ δ) :
    TV (softmax z) (softmax (z + h)) ≤ min 1 δ :=
  le_min (tv_softmax_le_one z (z + h)) (tv_le_delta hh)

/-- Symmetric phrasing in terms of two logit vectors. -/
theorem tv_le_of_logit_dist {z z' : V → ℝ} {δ : ℝ}
    (hzz : ∀ v, |z' v - z v| ≤ δ) :
    TV (softmax z) (softmax z') ≤ min 1 (δ / 2) := by
  have h := tv_le_half_delta (z := z) (h := z' - z) (δ := δ)
    (fun v => by simpa using hzz v)
  rw [add_sub_cancel] at h
  exact le_min (tv_softmax_le_one z z') h

end SoftmaxDivergence

section TwoStep

variable {A B : Type*} [Fintype A] [Fintype B]

/-- Joint law of a first-step law and a conditional kernel. -/
def joint (P1 : A → ℝ) (P2 : A → B → ℝ) : A × B → ℝ :=
  fun ab => P1 ab.1 * P2 ab.1 ab.2

/-- Item 6, two-step KL chain rule. -/
theorem kl_chain_two {P1 Q1 : A → ℝ} {P2 Q2 : A → B → ℝ}
    (hP2 : ∀ a, ∑ b, P2 a b = 1)
    (hQ1 : ∀ a, 0 < Q1 a) (hQ2 : ∀ a b, 0 < Q2 a b) :
    KL (joint P1 P2) (joint Q1 Q2) =
      KL P1 Q1 + ∑ a, P1 a * KL (P2 a) (Q2 a) := by
  unfold KL joint
  rw [Fintype.sum_prod_type, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro a _
  have hterm : ∀ b,
      P1 a * P2 a b * Real.log (P1 a * P2 a b / (Q1 a * Q2 a b)) =
        P2 a b * (P1 a * Real.log (P1 a / Q1 a)) +
          P1 a * (P2 a b * Real.log (P2 a b / Q2 a b)) := by
    intro b
    by_cases h1 : P1 a = 0
    · simp [h1]
    by_cases h2 : P2 a b = 0
    · simp [h2]
    rw [mul_div_mul_comm,
      Real.log_mul (div_ne_zero h1 (hQ1 a).ne') (div_ne_zero h2 (hQ2 a b).ne')]
    ring
  dsimp only
  rw [Finset.sum_congr rfl (fun b _ => hterm b), Finset.sum_add_distrib,
    ← Finset.sum_mul, hP2 a, one_mul, ← Finset.mul_sum]

/-- Item 7, two-step TV bound (no positivity, no coupling). -/
theorem tv_chain_two {P1 Q1 : A → ℝ} {P2 Q2 : A → B → ℝ}
    (hP1 : ∀ a, 0 ≤ P1 a) (hQ2 : ∀ a b, 0 ≤ Q2 a b)
    (hQ2s : ∀ a, ∑ b, Q2 a b = 1) :
    TV (joint P1 P2) (joint Q1 Q2) ≤
      TV P1 Q1 + ∑ a, P1 a * TV (P2 a) (Q2 a) := by
  have hrow : ∀ a, ∑ b, |P1 a * P2 a b - Q1 a * Q2 a b| ≤
      |P1 a - Q1 a| + P1 a * ∑ b, |P2 a b - Q2 a b| := by
    intro a
    calc
      ∑ b, |P1 a * P2 a b - Q1 a * Q2 a b| ≤
          ∑ b, (|P1 a - Q1 a| * Q2 a b + P1 a * |P2 a b - Q2 a b|) := by
        apply Finset.sum_le_sum
        intro b _
        have e : P1 a * P2 a b - Q1 a * Q2 a b =
            (P1 a - Q1 a) * Q2 a b + P1 a * (P2 a b - Q2 a b) := by ring
        rw [e]
        refine (abs_add_le _ _).trans ?_
        rw [abs_mul, abs_mul, abs_of_nonneg (hQ2 a b), abs_of_nonneg (hP1 a)]
      _ = |P1 a - Q1 a| + P1 a * ∑ b, |P2 a b - Q2 a b| := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
          hQ2s a, mul_one]
  have hall := Finset.sum_le_sum (fun a (_ : a ∈ Finset.univ) => hrow a)
  rw [Finset.sum_add_distrib] at hall
  have e : ∑ a, P1 a * ((1 / 2 : ℝ) * ∑ b, |P2 a b - Q2 a b|) =
      (1 / 2 : ℝ) * ∑ a, P1 a * ∑ b, |P2 a b - Q2 a b| := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro a _
    ring
  unfold TV joint
  rw [Fintype.sum_prod_type, e]
  dsimp only
  linarith

/-- Item 7, two-step uniform form. -/
theorem tv_chain_two_uniform {P1 Q1 : A → ℝ} {P2 Q2 : A → B → ℝ} {ε : ℝ}
    (hP1 : ∀ a, 0 ≤ P1 a) (hP1s : ∑ a, P1 a = 1)
    (hQ2 : ∀ a b, 0 ≤ Q2 a b) (hQ2s : ∀ a, ∑ b, Q2 a b = 1)
    (hε : ∀ a, TV (P2 a) (Q2 a) ≤ ε) :
    TV (joint P1 P2) (joint Q1 Q2) ≤ TV P1 Q1 + ε := by
  refine (tv_chain_two hP1 hQ2 hQ2s).trans ?_
  have : ∑ a, P1 a * TV (P2 a) (Q2 a) ≤ ∑ a, P1 a * ε :=
    Finset.sum_le_sum (fun a _ => mul_le_mul_of_nonneg_left (hε a) (hP1 a))
  rw [← Finset.sum_mul, hP1s, one_mul] at this
  linarith

end TwoStep

section Sequences

variable {A : Type*}

/-- Drop the last token of a history. -/
def initSeq {n : ℕ} (x : Fin (n + 1) → A) : Fin n → A :=
  fun i => x i.castSucc

/-- Append a token to a history. -/
def snocSeq {n : ℕ} (y : Fin n → A) (a : A) : Fin (n + 1) → A :=
  Fin.snoc (α := fun _ => A) y a

theorem initSeq_snocSeq {n : ℕ} (y : Fin n → A) (a : A) :
    initSeq (snocSeq y a) = y := by
  funext i
  simp [initSeq, snocSeq]

theorem snocSeq_last {n : ℕ} (y : Fin n → A) (a : A) :
    snocSeq y a (Fin.last n) = a := by
  simp [snocSeq]

/-- Histories of length `n + 1` are a history of length `n` plus a token. -/
def snocSeqEquiv (n : ℕ) : (Fin n → A) × A ≃ (Fin (n + 1) → A) where
  toFun p := snocSeq p.1 p.2
  invFun x := (initSeq x, x (Fin.last n))
  left_inv := by
    rintro ⟨y, a⟩
    simp [initSeq_snocSeq, snocSeq_last]
  right_inv := by
    intro x
    funext i
    refine Fin.lastCases ?_ (fun j => ?_) i <;> simp [snocSeq, initSeq]

/-- Autoregressive law of a length-`n` history under next-token kernels
`K t : history of length t → next-token law`. -/
def seqLaw (K : ∀ n, (Fin n → A) → A → ℝ) : ∀ n, (Fin n → A) → ℝ
  | 0, _ => 1
  | n + 1, x => seqLaw K n (initSeq x) * K n (initSeq x) (x (Fin.last n))

theorem seqLaw_snocSeq (K : ∀ n, (Fin n → A) → A → ℝ) {n : ℕ}
    (y : Fin n → A) (a : A) :
    seqLaw K (n + 1) (snocSeq y a) = seqLaw K n y * K n y a := by
  simp [seqLaw, initSeq_snocSeq, snocSeq_last]

theorem seqLaw_nonneg {K : ∀ n, (Fin n → A) → A → ℝ}
    (hK : ∀ n x a, 0 ≤ K n x a) : ∀ n x, 0 ≤ seqLaw K n x
  | 0, _ => zero_le_one
  | n + 1, _ => mul_nonneg (seqLaw_nonneg hK n _) (hK n _ _)

theorem seqLaw_pos {K : ∀ n, (Fin n → A) → A → ℝ}
    (hK : ∀ n x a, 0 < K n x a) : ∀ n x, 0 < seqLaw K n x
  | 0, _ => zero_lt_one
  | n + 1, _ => mul_pos (seqLaw_pos hK n _) (hK n _ _)

variable [Fintype A]

theorem sum_snocSeq {n : ℕ} (F : (Fin (n + 1) → A) → ℝ) :
    ∑ x, F x = ∑ y : Fin n → A, ∑ a : A, F (snocSeq y a) := by
  rw [← (snocSeqEquiv n).sum_comp, Fintype.sum_prod_type]
  rfl

theorem seqLaw_sum_one {K : ∀ n, (Fin n → A) → A → ℝ}
    (hK : ∀ n x, ∑ a, K n x a = 1) : ∀ n, ∑ x, seqLaw K n x = 1 := by
  intro n
  induction n with
  | zero => simp [seqLaw]
  | succ n ih =>
    rw [sum_snocSeq]
    simp only [seqLaw_snocSeq]
    simp_rw [← Finset.mul_sum, hK, mul_one]
    exact ih

theorem kl_seq_succ (K L : ∀ n, (Fin n → A) → A → ℝ) (n : ℕ) :
    KL (seqLaw K (n + 1)) (seqLaw L (n + 1)) =
      KL (joint (seqLaw K n) (K n)) (joint (seqLaw L n) (L n)) := by
  unfold KL
  rw [sum_snocSeq, Fintype.sum_prod_type]
  simp only [seqLaw_snocSeq, joint]

theorem tv_seq_succ (K L : ∀ n, (Fin n → A) → A → ℝ) (n : ℕ) :
    TV (seqLaw K (n + 1)) (seqLaw L (n + 1)) =
      TV (joint (seqLaw K n) (K n)) (joint (seqLaw L n) (L n)) := by
  unfold TV
  rw [sum_snocSeq, Fintype.sum_prod_type]
  simp only [seqLaw_snocSeq, joint]

/-- Item 6, `T`-step KL chain rule for autoregressive laws on `Fin T → A`. -/
theorem kl_chain_seq {K L : ∀ n, (Fin n → A) → A → ℝ}
    (hK : ∀ n x, ∑ a, K n x a = 1) (hL : ∀ n x a, 0 < L n x a) (T : ℕ) :
    KL (seqLaw K T) (seqLaw L T) =
      ∑ t ∈ Finset.range T, ∑ x : Fin t → A,
        seqLaw K t x * KL (K t x) (L t x) := by
  induction T with
  | zero => simp [KL, seqLaw]
  | succ n ih =>
    rw [Finset.sum_range_succ, kl_seq_succ,
      kl_chain_two (hK n) (seqLaw_pos hL n) (hL n), ih]

/-- Item 6, uniform corollary: per-step conditional KL at most `c` gives
sequence KL at most `T * c`. -/
theorem kl_seq_le_of_uniform {K L : ∀ n, (Fin n → A) → A → ℝ} {c : ℝ}
    (hK0 : ∀ n x a, 0 ≤ K n x a) (hK : ∀ n x, ∑ a, K n x a = 1)
    (hL : ∀ n x a, 0 < L n x a)
    (hc : ∀ n x, KL (K n x) (L n x) ≤ c) (T : ℕ) :
    KL (seqLaw K T) (seqLaw L T) ≤ T * c := by
  rw [kl_chain_seq hK hL]
  calc
    ∑ t ∈ Finset.range T, ∑ x : Fin t → A,
        seqLaw K t x * KL (K t x) (L t x) ≤
        ∑ _t ∈ Finset.range T, c := by
      apply Finset.sum_le_sum
      intro t _
      calc
        ∑ x : Fin t → A, seqLaw K t x * KL (K t x) (L t x) ≤
            ∑ x : Fin t → A, seqLaw K t x * c :=
          Finset.sum_le_sum (fun x _ =>
            mul_le_mul_of_nonneg_left (hc t x) (seqLaw_nonneg hK0 t x))
        _ = c := by rw [← Finset.sum_mul, seqLaw_sum_one hK, one_mul]
    _ = T * c := by simp

/-- Item 7, `T`-step TV bound for autoregressive laws on `Fin T → A`. -/
theorem tv_chain_seq {K L : ∀ n, (Fin n → A) → A → ℝ}
    (hK0 : ∀ n x a, 0 ≤ K n x a)
    (hL0 : ∀ n x a, 0 ≤ L n x a) (hL : ∀ n x, ∑ a, L n x a = 1) (T : ℕ) :
    TV (seqLaw K T) (seqLaw L T) ≤
      ∑ t ∈ Finset.range T, ∑ x : Fin t → A,
        seqLaw K t x * TV (K t x) (L t x) := by
  induction T with
  | zero => simp [TV, seqLaw]
  | succ n ih =>
    rw [Finset.sum_range_succ, tv_seq_succ]
    have := tv_chain_two (P1 := seqLaw K n) (Q1 := seqLaw L n)
      (P2 := K n) (Q2 := L n) (seqLaw_nonneg hK0 n) (hL0 n) (hL n)
    linarith

/-- Item 7, uniform corollary: per-step conditional TV at most `ε t` gives
sequence TV at most `∑ t < T, ε t`. -/
theorem tv_seq_le_of_step_bounds {K L : ∀ n, (Fin n → A) → A → ℝ} {ε : ℕ → ℝ}
    (hK0 : ∀ n x a, 0 ≤ K n x a) (hK : ∀ n x, ∑ a, K n x a = 1)
    (hL0 : ∀ n x a, 0 ≤ L n x a) (hL : ∀ n x, ∑ a, L n x a = 1)
    (hε : ∀ n x, TV (K n x) (L n x) ≤ ε n) (T : ℕ) :
    TV (seqLaw K T) (seqLaw L T) ≤ ∑ t ∈ Finset.range T, ε t := by
  refine (tv_chain_seq hK0 hL0 hL T).trans ?_
  apply Finset.sum_le_sum
  intro t _
  calc
    ∑ x : Fin t → A, seqLaw K t x * TV (K t x) (L t x) ≤
        ∑ x : Fin t → A, seqLaw K t x * ε t :=
      Finset.sum_le_sum (fun x _ =>
        mul_le_mul_of_nonneg_left (hε t x) (seqLaw_nonneg hK0 t x))
    _ = ε t := by rw [← Finset.sum_mul, seqLaw_sum_one hK, one_mul]

/-- Item 7 with a constant per-step bound: sequence TV at most `T * ε`. -/
theorem tv_seq_le_of_uniform {K L : ∀ n, (Fin n → A) → A → ℝ} {ε : ℝ}
    (hK0 : ∀ n x a, 0 ≤ K n x a) (hK : ∀ n x, ∑ a, K n x a = 1)
    (hL0 : ∀ n x a, 0 ≤ L n x a) (hL : ∀ n x, ∑ a, L n x a = 1)
    (hε : ∀ n x, TV (K n x) (L n x) ≤ ε) (T : ℕ) :
    TV (seqLaw K T) (seqLaw L T) ≤ T * ε := by
  have := tv_seq_le_of_step_bounds (ε := fun _ => ε) hK0 hK hL0 hL hε T
  simpa using this

variable [Nonempty A]

/-- Autoregressive softmax decoders whose logits differ by at most `δ` in
sup norm at every history: sequence KL is at most `T * (2 δ)`. -/
theorem kl_softmax_seq_le {z h : ∀ n, (Fin n → A) → A → ℝ} {δ : ℝ}
    (hh : ∀ n x a, |h n x a| ≤ δ) (T : ℕ) :
    KL (seqLaw (fun n x => softmax (z n x)) T)
        (seqLaw (fun n x => softmax (z n x + h n x)) T) ≤ T * (2 * δ) :=
  kl_seq_le_of_uniform (fun n x a => (softmax_pos (z n x) a).le)
    (fun n x => softmax_sum_one (z n x))
    (fun n x a => softmax_pos (z n x + h n x) a)
    (fun n x => kl_le_two_delta (hh n x)) T

/-- Same setting: sequence TV is at most `T * (δ / 2)`. -/
theorem tv_softmax_seq_le {z h : ∀ n, (Fin n → A) → A → ℝ} {δ : ℝ}
    (hh : ∀ n x a, |h n x a| ≤ δ) (T : ℕ) :
    TV (seqLaw (fun n x => softmax (z n x)) T)
        (seqLaw (fun n x => softmax (z n x + h n x)) T) ≤ T * (δ / 2) :=
  tv_seq_le_of_uniform (fun n x a => (softmax_pos (z n x) a).le)
    (fun n x => softmax_sum_one (z n x))
    (fun n x a => (softmax_pos (z n x + h n x) a).le)
    (fun n x => softmax_sum_one (z n x + h n x))
    (fun n x => tv_le_half_delta (hh n x)) T

end Sequences

end ApexQuant.SoftmaxBounds
