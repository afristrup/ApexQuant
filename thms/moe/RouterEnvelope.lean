import Mathlib

open Finset

namespace Progressive.RouterEnvelope

variable {E : Type*} [Fintype E] {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]

/-! ### 1. Routed error is bounded by the gate-weighted expert errors -/

/-- The gate-weighted sum of expert output errors is bounded in norm by the
gate-weighted sum of the per-expert error bounds. -/
theorem routed_norm_le {g : E → ℝ} {f fh : E → F} {Err : E → ℝ}
    (hg : ∀ e, 0 ≤ g e) (herr : ∀ e, ‖f e - fh e‖ ≤ Err e) :
    ‖∑ e, g e • (f e - fh e)‖ ≤ ∑ e, g e * Err e :=
  calc ‖∑ e, g e • (f e - fh e)‖ ≤ ∑ e, ‖g e • (f e - fh e)‖ := norm_sum_le _ _
    _ ≤ ∑ e, g e * Err e := Finset.sum_le_sum fun e _ => by
        rw [norm_smul, Real.norm_of_nonneg (hg e)]
        exact mul_le_mul_of_nonneg_left (herr e) (hg e)

/-- The same bound for the difference of the exact and the approximate routed
outputs. -/
theorem routed_sub_norm_le {g : E → ℝ} {f fh : E → F} {Err : E → ℝ}
    (hg : ∀ e, 0 ≤ g e) (herr : ∀ e, ‖f e - fh e‖ ≤ Err e) :
    ‖(∑ e, g e • f e) - (∑ e, g e • fh e)‖ ≤ ∑ e, g e * Err e := by
  have h : (∑ e, g e • f e) - (∑ e, g e • fh e) = ∑ e, g e • (f e - fh e) := by
    simp only [smul_sub, Finset.sum_sub_distrib]
  rw [h]
  exact routed_norm_le hg herr

/-! ### 2. Linear experts -/

/-- For linear experts with operator-norm weight errors `‖W e - Wh e‖ ≤ U e`,
the routed output error is at most `(∑ e, g e * U e) * ‖x‖`. -/
theorem linear_expert_envelope {X : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X]
    {g : E → ℝ} {W Wh : E → X →L[ℝ] F} {U : E → ℝ}
    (hg : ∀ e, 0 ≤ g e) (hU : ∀ e, ‖W e - Wh e‖ ≤ U e) (x : X) :
    ‖(∑ e, g e • W e x) - (∑ e, g e • Wh e x)‖ ≤ (∑ e, g e * U e) * ‖x‖ := by
  have herr : ∀ e, ‖W e x - Wh e x‖ ≤ U e * ‖x‖ := fun e =>
    calc ‖W e x - Wh e x‖ = ‖(W e - Wh e) x‖ := rfl
      _ ≤ ‖W e - Wh e‖ * ‖x‖ := (W e - Wh e).le_opNorm x
      _ ≤ U e * ‖x‖ := mul_le_mul_of_nonneg_right (hU e) (norm_nonneg x)
  calc ‖(∑ e, g e • W e x) - (∑ e, g e • Wh e x)‖
      ≤ ∑ e, g e * (U e * ‖x‖) := routed_sub_norm_le hg herr
    _ = (∑ e, g e * U e) * ‖x‖ := by
        rw [Finset.sum_mul]
        exact Finset.sum_congr rfl fun e _ => (mul_assoc _ _ _).symm

/-! ### 3. Geometric depth rule -/

/-- The depth rule for one expert: with a geometric error law `Err ≤ A * γ ^ p`
and depth `p ≥ log (τ / (g * A)) / log γ`, the gated error is at most `τ`. -/
theorem gate_mul_err_le_of_depth {g A τ γ Err : ℝ} {p : ℕ}
    (hγ0 : 0 < γ) (hγ1 : γ < 1) (hg : 0 < g) (hA : 0 < A) (hτ : 0 < τ)
    (hErr : Err ≤ A * γ ^ p)
    (hp : Real.log (τ / (g * A)) / Real.log γ ≤ p) :
    g * Err ≤ τ := by
  have hgA : 0 < g * A := mul_pos hg hA
  have hlog : Real.log γ < 0 := Real.log_neg hγ0 hγ1
  have h1 : (p : ℝ) * Real.log γ ≤ Real.log (τ / (g * A)) := (div_le_iff_of_neg hlog).mp hp
  rw [← Real.log_pow] at h1
  have h2 : γ ^ p ≤ τ / (g * A) :=
    (Real.log_le_log_iff (pow_pos hγ0 p) (div_pos hτ hgA)).mp h1
  have h3 : γ ^ p * (g * A) ≤ τ := (le_div_iff₀ hgA).mp h2
  calc g * Err ≤ g * (A * γ ^ p) := mul_le_mul_of_nonneg_left hErr hg.le
    _ = γ ^ p * (g * A) := by ring
    _ ≤ τ := h3

omit [Fintype E] in
/-- Geometric depth rule, per expert: every gated expert error meets its
tolerance. -/
theorem depth_rule_pointwise {g A τ Err : E → ℝ} {γ : ℝ} {p : E → ℕ}
    (hγ0 : 0 < γ) (hγ1 : γ < 1) (hg : ∀ e, 0 < g e) (hA : ∀ e, 0 < A e)
    (hτ : ∀ e, 0 < τ e) (hErr : ∀ e, Err e ≤ A e * γ ^ p e)
    (hp : ∀ e, Real.log (τ e / (g e * A e)) / Real.log γ ≤ p e) (e : E) :
    g e * Err e ≤ τ e :=
  gate_mul_err_le_of_depth hγ0 hγ1 (hg e) (hA e) (hτ e) (hErr e) (hp e)

/-- Geometric depth rule: under the depth condition for every expert the routed
error is at most the total tolerance `∑ e, τ e`. -/
theorem depth_rule {g A τ Err : E → ℝ} {γ : ℝ} {p : E → ℕ} {f fh : E → F}
    (hγ0 : 0 < γ) (hγ1 : γ < 1) (hg : ∀ e, 0 < g e) (hA : ∀ e, 0 < A e)
    (hτ : ∀ e, 0 < τ e) (herr : ∀ e, ‖f e - fh e‖ ≤ Err e)
    (hErr : ∀ e, Err e ≤ A e * γ ^ p e)
    (hp : ∀ e, Real.log (τ e / (g e * A e)) / Real.log γ ≤ p e) :
    ‖(∑ e, g e • f e) - (∑ e, g e • fh e)‖ ≤ ∑ e, τ e :=
  (routed_sub_norm_le (fun e => (hg e).le) herr).trans
    (Finset.sum_le_sum fun e _ => depth_rule_pointwise hγ0 hγ1 hg hA hτ hErr hp e)

omit [Fintype E] in
/-- Per-expert depth rule with inactive experts: an expert with zero gate needs
no depth at all. -/
theorem depth_rule_pointwise_of_nonneg {g A τ Err : E → ℝ} {γ : ℝ} {p : E → ℕ}
    (hγ0 : 0 < γ) (hγ1 : γ < 1) (hg : ∀ e, 0 ≤ g e) (hA : ∀ e, 0 < A e)
    (hτ : ∀ e, 0 < τ e) (hErr : ∀ e, Err e ≤ A e * γ ^ p e)
    (hp : ∀ e, 0 < g e → Real.log (τ e / (g e * A e)) / Real.log γ ≤ p e) (e : E) :
    g e * Err e ≤ τ e := by
  rcases (hg e).eq_or_lt with h0 | hpos
  · rw [← h0, zero_mul]
    exact (hτ e).le
  · exact gate_mul_err_le_of_depth hγ0 hγ1 hpos (hA e) (hτ e) (hErr e) (hp e hpos)

/-- Geometric depth rule with inactive experts: the depth condition is only
required of experts with a positive gate. -/
theorem depth_rule_of_nonneg {g A τ Err : E → ℝ} {γ : ℝ} {p : E → ℕ} {f fh : E → F}
    (hγ0 : 0 < γ) (hγ1 : γ < 1) (hg : ∀ e, 0 ≤ g e) (hA : ∀ e, 0 < A e)
    (hτ : ∀ e, 0 < τ e) (herr : ∀ e, ‖f e - fh e‖ ≤ Err e)
    (hErr : ∀ e, Err e ≤ A e * γ ^ p e)
    (hp : ∀ e, 0 < g e → Real.log (τ e / (g e * A e)) / Real.log γ ≤ p e) :
    ‖(∑ e, g e • f e) - (∑ e, g e • fh e)‖ ≤ ∑ e, τ e :=
  (routed_sub_norm_le hg herr).trans
    (Finset.sum_le_sum fun e _ =>
      depth_rule_pointwise_of_nonneg hγ0 hγ1 hg hA hτ hErr hp e)

/-! ### 4. Simplex gates -/

/-- Gates on the simplex and a uniform expert error bound `ε` give routed error
at most `ε`. -/
theorem simplex_envelope {g : E → ℝ} {f fh : E → F} {ε : ℝ}
    (hg : ∀ e, 0 ≤ g e) (hsum : ∑ e, g e = 1) (herr : ∀ e, ‖f e - fh e‖ ≤ ε) :
    ‖(∑ e, g e • f e) - (∑ e, g e • fh e)‖ ≤ ε :=
  calc ‖(∑ e, g e • f e) - (∑ e, g e • fh e)‖ ≤ ∑ e, g e * ε := routed_sub_norm_le hg herr
    _ = ε := by rw [← Finset.sum_mul, hsum, one_mul]

end Progressive.RouterEnvelope
