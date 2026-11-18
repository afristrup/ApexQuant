import Mathlib

open Finset

namespace Progressive.MoE

/-- The router-weighted residual contribution of an expert mixture. -/
noncomputable def routerError {E : Type*} [Fintype E]
    (gate : E → ℝ) (residual : E → ℕ → ℝ) (passes : E → ℕ) : ℝ :=
  ∑ e, gate e * residual e (passes e)

/-- The corresponding geometric envelope after nested decoding. -/
noncomputable def routerEnvelope {E : Type*} [Fintype E]
    (ρ : ℝ) (gate initial : E → ℝ) (passes : E → ℕ) : ℝ :=
  ∑ e, gate e * (initial e * ρ ^ passes e)

/-- Per-expert residuals contract by `ρ` at every additional nested pass. -/
structure GeometricResidual {E : Type*} [Fintype E]
    (ρ : ℝ) (residual : E → ℕ → ℝ) : Prop where
  initial_nonneg : ∀ e, 0 ≤ residual e 0
  step : ∀ e p, residual e (p + 1) ≤ ρ * residual e p

/-- The contraction law gives the expected geometric envelope for every expert. -/
theorem residual_le_geometric {E : Type*} [Fintype E]
    {ρ : ℝ} {residual : E → ℕ → ℝ}
    (hρ : 0 ≤ ρ) (h : GeometricResidual ρ residual) (e : E) (p : ℕ) :
    residual e p ≤ residual e 0 * ρ ^ p := by
  induction p with
  | zero => simp
  | succ p ih =>
      calc
        residual e (p + 1) ≤ ρ * residual e p := h.step e p
        _ ≤ ρ * (residual e 0 * ρ ^ p) :=
          mul_le_mul_of_nonneg_left ih hρ
        _ = residual e 0 * ρ ^ (p + 1) := by
          rw [pow_succ]
          ring

/-- Router weights turn the per-expert geometric law into a mixture guarantee. -/
theorem router_error_le_envelope {E : Type*} [Fintype E]
    {ρ : ℝ} {gate : E → ℝ} {residual : E → ℕ → ℝ} {passes : E → ℕ}
    (hρ : 0 ≤ ρ) (hdecay : GeometricResidual ρ residual)
    (hgate : ∀ e, 0 ≤ gate e) :
    routerError gate residual passes ≤
      routerEnvelope ρ gate (fun e => residual e 0) passes := by
  apply Finset.sum_le_sum
  intro e _
  exact mul_le_mul_of_nonneg_left
    (residual_le_geometric hρ hdecay e (passes e)) (hgate e)

/-- Giving an expert a longer prefix cannot increase the router envelope. -/
theorem router_envelope_mono {E : Type*} [Fintype E]
    {ρ : ℝ} {gate initial : E → ℝ} {short long : E → ℕ}
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (hgate : ∀ e, 0 ≤ gate e)
    (hinitial : ∀ e, 0 ≤ initial e)
    (hpasses : ∀ e, short e ≤ long e) :
    routerEnvelope ρ gate initial long ≤
      routerEnvelope ρ gate initial short := by
  apply Finset.sum_le_sum
  intro e _
  have hpow : ρ ^ long e ≤ ρ ^ short e :=
    pow_le_pow_of_le_one hρ0 hρ1 (hpasses e)
  have hterm : initial e * ρ ^ long e ≤ initial e * ρ ^ short e :=
    mul_le_mul_of_nonneg_left hpow (hinitial e)
  exact mul_le_mul_of_nonneg_left hterm (hgate e)

/-- Any pass assignment is bounded by the zero-pass initial residual. -/
theorem router_envelope_le_initial {E : Type*} [Fintype E]
    {ρ : ℝ} {gate initial : E → ℝ} {passes : E → ℕ}
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (hgate : ∀ e, 0 ≤ gate e)
    (hinitial : ∀ e, 0 ≤ initial e) :
    routerEnvelope ρ gate initial passes ≤
      ∑ e, gate e * initial e := by
  calc
    routerEnvelope ρ gate initial passes ≤
        routerEnvelope ρ gate initial (fun _ => 0) :=
          router_envelope_mono hρ0 hρ1 hgate hinitial (fun _ => Nat.zero_le _)
    _ = ∑ e, gate e * initial e := by simp [routerEnvelope]

/-- If every expert runs at least one pass, the envelope contracts by `ρ`. -/
theorem router_envelope_le_first_pass {E : Type*} [Fintype E]
    {ρ : ℝ} {gate initial : E → ℝ} {passes : E → ℕ}
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (hgate : ∀ e, 0 ≤ gate e)
    (hinitial : ∀ e, 0 ≤ initial e) (hone : ∀ e, 1 ≤ passes e) :
    routerEnvelope ρ gate initial passes ≤
      ρ * ∑ e, gate e * initial e := by
  calc
    routerEnvelope ρ gate initial passes ≤
        routerEnvelope ρ gate initial (fun _ => 1) :=
          router_envelope_mono hρ0 hρ1 hgate hinitial hone
    _ = ρ * ∑ e, gate e * initial e := by
      simp [routerEnvelope, mul_sum]
      apply Finset.sum_congr rfl
      intro e _
      ring

/-- Per-expert error budgets imply a total router-error budget. -/
theorem router_error_le_of_budget {E : Type*} [Fintype E]
    {ρ : ℝ} {gate : E → ℝ} {residual : E → ℕ → ℝ} {passes : E → ℕ}
    {τ : E → ℝ} (hρ : 0 ≤ ρ)
    (hdecay : GeometricResidual ρ residual)
    (hgate : ∀ e, 0 ≤ gate e)
    (hbudget : ∀ e, gate e * (residual e 0 * ρ ^ passes e) ≤ τ e) :
    routerError gate residual passes ≤ ∑ e, τ e := by
  calc
    routerError gate residual passes ≤
        routerEnvelope ρ gate (fun e => residual e 0) passes :=
      router_error_le_envelope hρ hdecay hgate
    _ ≤ ∑ e, τ e := Finset.sum_le_sum (fun e _ => hbudget e)

/-- Equalized per-expert budgets imply a total error tolerance. -/
theorem router_error_le_eps_of_equalized {E : Type*} [Fintype E] [Nonempty E]
    {ρ ε : ℝ} {gate : E → ℝ} {residual : E → ℕ → ℝ} {passes : E → ℕ}
    (hρ : 0 ≤ ρ) (hdecay : GeometricResidual ρ residual)
    (hgate : ∀ e, 0 ≤ gate e)
    (hequal : ∀ e, gate e * (residual e 0 * ρ ^ passes e) ≤
      ε / (Fintype.card E : ℝ)) :
    routerError gate residual passes ≤ ε := by
  calc
    routerError gate residual passes ≤
        ∑ _e : E, (ε / (Fintype.card E : ℝ)) :=
      router_error_le_of_budget hρ hdecay hgate hequal
    _ = (Fintype.card E : ℝ) * (ε / (Fintype.card E : ℝ)) := by
      simp [sum_const, nsmul_eq_mul]
    _ = ε := mul_div_cancel₀ ε (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)

/-- The least natural pass count whose geometric envelope is at most `ε`. -/
noncomputable def thresholdPass (ρ ε : ℝ) : ℕ :=
  ⌈Real.log ε / Real.log ρ⌉₊

/-- The logarithmic threshold rule reaches its requested geometric tolerance. -/
theorem threshold_pass_spec {ρ ε : ℝ}
    (hρ0 : 0 < ρ) (hρ1 : ρ < 1) (hε0 : 0 < ε) :
    ρ ^ thresholdPass ρ ε ≤ ε := by
  unfold thresholdPass
  have hL : Real.log ρ < 0 := Real.log_neg hρ0 hρ1
  have hstep : (⌈Real.log ε / Real.log ρ⌉₊ : ℝ) * Real.log ρ ≤ Real.log ε := by
    have hceil := mul_le_mul_of_nonpos_right
      (Nat.le_ceil (Real.log ε / Real.log ρ)) hL.le
    rwa [div_mul_cancel₀ _ hL.ne] at hceil
  rw [← Real.log_le_log_iff (pow_pos hρ0 _) hε0, Real.log_pow]
  exact hstep

/-- Expert-specific threshold budgets imply a total router-error budget. -/
theorem router_error_le_of_threshold {E : Type*} [Fintype E]
    {ρ : ℝ} {gate : E → ℝ} {residual : E → ℕ → ℝ} {passes : E → ℕ}
    {τ : E → ℝ} (hρ : 0 ≤ ρ) (hρpos : 0 < ρ)
    (hdecay : GeometricResidual ρ residual)
    (hgate : ∀ e, 0 < gate e) (hinitial : ∀ e, 0 < residual e 0)
    (hτ : ∀ e, 0 < τ e)
    (hpasses : ∀ e, passes e = thresholdPass ρ (τ e / (gate e * residual e 0)))
    (hρlt : ρ < 1) :
    routerError gate residual passes ≤ ∑ e, τ e := by
  apply router_error_le_of_budget hρ hdecay (fun e => (hgate e).le)
  intro e
  have hden : 0 < gate e * residual e 0 := mul_pos (hgate e) (hinitial e)
  have hε0 : 0 < τ e / (gate e * residual e 0) := div_pos (hτ e) hden
  have hpow : ρ ^ passes e ≤ τ e / (gate e * residual e 0) := by
    rw [hpasses e]
    exact threshold_pass_spec hρpos hρlt hε0
  calc
    gate e * (residual e 0 * ρ ^ passes e) =
        (gate e * residual e 0) * ρ ^ passes e := by ring
    _ ≤ (gate e * residual e 0) *
        (τ e / (gate e * residual e 0)) :=
      mul_le_mul_of_nonneg_left hpow hden.le
    _ = τ e := by
      rw [← mul_div_assoc, mul_div_cancel_left₀ _ (ne_of_gt hden)]

/-- Simplex-weighted errors are bounded by a uniform per-component cap. -/
theorem router_error_le_of_simplex {E : Type*} [Fintype E]
    {ρ ε : ℝ} {gate : E → ℝ} {residual : E → ℕ → ℝ} {passes : E → ℕ}
    (hρ : 0 ≤ ρ) (hdecay : GeometricResidual ρ residual)
    (hgate : ∀ e, 0 ≤ gate e) (hsum : ∑ e, gate e = 1)
    (hbound : ∀ e, residual e 0 * ρ ^ passes e ≤ ε) :
    routerError gate residual passes ≤ ε := by
  calc
    routerError gate residual passes ≤ ∑ e, gate e * ε := by
      apply router_error_le_of_budget hρ hdecay hgate
      intro e
      exact mul_le_mul_of_nonneg_left (hbound e) (hgate e)
    _ = (∑ e, gate e) * ε := by rw [← sum_mul]
    _ = ε := by rw [hsum, one_mul]

/-- The squared router-error proxy optimized by equal-cost pass allocation. -/
noncomputable def squaredRouterObjective {E : Type*} [Fintype E]
    (ρ : ℝ) (gate initial : E → ℝ) (passes : E → ℕ) : ℝ :=
  ∑ e, gate e ^ 2 * initial e ^ 2 * ρ ^ (2 * passes e)

/-- Add one nested pass to one expert. -/
def passUpdate {E : Type*} [DecidableEq E] (passes : E → ℕ) (expert : E) : E → ℕ :=
  fun e => if e = expert then passes e + 1 else passes e

/-- The objective decrease from one additional pass for one expert. -/
noncomputable def marginalGain {E : Type*}
    (ρ : ℝ) (gate initial : E → ℝ) (expert : E) (pass : ℕ) : ℝ :=
  gate expert ^ 2 * initial expert ^ 2 * ρ ^ (2 * pass) * (1 - ρ ^ 2)

/-- One extra pass reduces the squared objective by its geometric marginal gain. -/
theorem squared_objective_step {E : Type*} [Fintype E] [DecidableEq E]
    {ρ : ℝ} {gate initial : E → ℝ} {passes : E → ℕ} (expert : E) :
    squaredRouterObjective ρ gate initial passes -
        squaredRouterObjective ρ gate initial (passUpdate passes expert) =
      marginalGain ρ gate initial expert (passes expert) := by
  classical
  unfold squaredRouterObjective passUpdate marginalGain
  rw [← Finset.sum_sub_distrib]
  have hpoint (e : E) :
      gate e ^ 2 * initial e ^ 2 * ρ ^ (2 * passes e) -
          gate e ^ 2 * initial e ^ 2 * ρ ^
            (2 * (if e = expert then passes e + 1 else passes e)) =
        if e = expert then
          gate e ^ 2 * initial e ^ 2 * ρ ^ (2 * passes e) -
            gate e ^ 2 * initial e ^ 2 * ρ ^ (2 * (passes e + 1))
        else 0 := by
    by_cases h : e = expert <;> simp [h]
  simp_rw [hpoint]
  simp only [Finset.sum_ite_eq', Finset.mem_univ, if_true]
  have hexp : 2 * (passes expert + 1) = 2 * passes expert + 2 := by omega
  rw [hexp, pow_add]
  ring

/-- Increasing any pass count cannot increase the squared router objective. -/
theorem squared_objective_mono {E : Type*} [Fintype E]
    {ρ : ℝ} {gate initial : E → ℝ} {short long : E → ℕ}
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    (hpasses : ∀ e, short e ≤ long e) :
    squaredRouterObjective ρ gate initial long ≤
      squaredRouterObjective ρ gate initial short := by
  apply Finset.sum_le_sum
  intro e _
  have hpow : ρ ^ (2 * long e) ≤ ρ ^ (2 * short e) := by
    apply pow_le_pow_of_le_one hρ0 hρ1
    exact Nat.mul_le_mul_left 2 (hpasses e)
  have hcoeff : 0 ≤ gate e ^ 2 * initial e ^ 2 := by positivity
  exact mul_le_mul_of_nonneg_left hpow hcoeff

/-- With equal pass costs, the largest marginal gain is the best next pass. -/
theorem best_next_pass {E : Type*} [Fintype E] [DecidableEq E]
    {ρ : ℝ} {gate initial : E → ℝ} {passes : E → ℕ}
    (i j : E)
    (hmax : marginalGain ρ gate initial j (passes j) ≤
      marginalGain ρ gate initial i (passes i)) :
    squaredRouterObjective ρ gate initial (passUpdate passes i) ≤
      squaredRouterObjective ρ gate initial (passUpdate passes j) := by
  have hi := squared_objective_step (ρ := ρ) (gate := gate)
    (initial := initial) (passes := passes) i
  have hj := squared_objective_step (ρ := ρ) (gate := gate)
    (initial := initial) (passes := passes) j
  linarith

end Progressive.MoE
