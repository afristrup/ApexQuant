import Mathlib

open Finset

namespace Progressive.GreedyOptimal

/-- Total benefit of a pass allocation: expert `e` collects its first `p e` marginal benefits. -/
noncomputable def benefit {E : Type*} [Fintype E] (Δ : E → ℕ → ℝ) (p : E → ℕ) : ℝ :=
  ∑ e, ∑ j ∈ range (p e), Δ e j

/-! ### 1. Threshold allocations are optimal -/

/-- Single-expert exchange bound: moving from `b` to `a` passes changes the benefit by at most
`θ` per pass, when granted passes are worth at least `θ` and ungranted ones at most `θ`. -/
theorem prefix_sub_le {d : ℕ → ℝ} {θ : ℝ} {a b : ℕ}
    (hlo : ∀ j, j < b → θ ≤ d j) (hhi : ∀ j, b ≤ j → j < a → d j ≤ θ) :
    ∑ j ∈ range a, d j - ∑ j ∈ range b, d j ≤ θ * ((a : ℝ) - (b : ℝ)) := by
  rcases le_total a b with h | h
  · have key : ∀ n, a + n ≤ b →
        ∑ j ∈ range a, d j - ∑ j ∈ range (a + n), d j ≤
          θ * ((a : ℝ) - ((a + n : ℕ) : ℝ)) := by
      intro n
      induction n with
      | zero => simp
      | succ n ih =>
          intro hn
          have h1 := ih (by omega)
          have h2 := hlo (a + n) (by omega)
          rw [← Nat.add_assoc, sum_range_succ]
          push_cast at h1 ⊢
          linarith
    obtain ⟨n, rfl⟩ := Nat.exists_eq_add_of_le h
    exact key n le_rfl
  · have key : ∀ n, b + n ≤ a →
        ∑ j ∈ range (b + n), d j - ∑ j ∈ range b, d j ≤
          θ * (((b + n : ℕ) : ℝ) - (b : ℝ)) := by
      intro n
      induction n with
      | zero => simp
      | succ n ih =>
          intro hn
          have h1 := ih (by omega)
          have h2 := hhi (b + n) (by omega) (by omega)
          rw [← Nat.add_assoc, sum_range_succ]
          push_cast at h1 ⊢
          linarith
    obtain ⟨n, rfl⟩ := Nat.exists_eq_add_of_le h
    exact key n le_rfl

/-- Lagrangian bound: against a threshold allocation `pstar` with price `θ`, any allocation `p`
within the caps gains at most `θ` per extra pass spent. No sign condition on `θ` is needed. -/
theorem benefit_sub_le_threshold {E : Type*} [Fintype E]
    {Δ : E → ℕ → ℝ} {cap pstar p : E → ℕ} {θ : ℝ}
    (hlo : ∀ e j, j < pstar e → θ ≤ Δ e j)
    (hhi : ∀ e j, pstar e ≤ j → j < cap e → Δ e j ≤ θ)
    (hcap : ∀ e, p e ≤ cap e) :
    benefit Δ p - benefit Δ pstar ≤
      θ * (((∑ e, p e : ℕ) : ℝ) - ((∑ e, pstar e : ℕ) : ℝ)) := by
  unfold benefit
  rw [← sum_sub_distrib]
  push_cast
  rw [← sum_sub_distrib, mul_sum]
  exact sum_le_sum fun e _ =>
    prefix_sub_le (hlo e) fun j hj hja => hhi e j hj (lt_of_lt_of_le hja (hcap e))

/-- Capped threshold optimality: a threshold allocation beats every cap-respecting allocation
that spends no more passes. `pstar` itself need not be assumed within the caps. -/
theorem threshold_optimal_capped {E : Type*} [Fintype E]
    {Δ : E → ℕ → ℝ} {cap pstar p : E → ℕ} {θ : ℝ} (hθ : 0 ≤ θ)
    (hlo : ∀ e j, j < pstar e → θ ≤ Δ e j)
    (hhi : ∀ e j, pstar e ≤ j → j < cap e → Δ e j ≤ θ)
    (hcap : ∀ e, p e ≤ cap e)
    (hbudget : ∑ e, p e ≤ ∑ e, pstar e) :
    benefit Δ p ≤ benefit Δ pstar := by
  have h := benefit_sub_le_threshold hlo hhi hcap
  have hb : ((∑ e, p e : ℕ) : ℝ) ≤ ((∑ e, pstar e : ℕ) : ℝ) := by exact_mod_cast hbudget
  nlinarith [mul_nonneg hθ (sub_nonneg.mpr hb)]

/-- Threshold optimality without caps. -/
theorem threshold_optimal {E : Type*} [Fintype E]
    {Δ : E → ℕ → ℝ} {pstar p : E → ℕ} {θ : ℝ} (hθ : 0 ≤ θ)
    (hlo : ∀ e j, j < pstar e → θ ≤ Δ e j)
    (hhi : ∀ e j, pstar e ≤ j → Δ e j ≤ θ)
    (hbudget : ∑ e, p e ≤ ∑ e, pstar e) :
    benefit Δ p ≤ benefit Δ pstar :=
  threshold_optimal_capped (cap := p) hθ hlo (fun e j hj _ => hhi e j hj)
    (fun _ => le_rfl) hbudget

/-- Leftover budget: with price `θ = 0` a threshold allocation is optimal among all
cap-respecting allocations, whatever they spend. -/
theorem threshold_optimal_of_zero {E : Type*} [Fintype E]
    {Δ : E → ℕ → ℝ} {cap pstar p : E → ℕ}
    (hlo : ∀ e j, j < pstar e → 0 ≤ Δ e j)
    (hhi : ∀ e j, pstar e ≤ j → j < cap e → Δ e j ≤ 0)
    (hcap : ∀ e, p e ≤ cap e) :
    benefit Δ p ≤ benefit Δ pstar := by
  have h := benefit_sub_le_threshold hlo hhi hcap
  linarith

/-! ### 2. Greedy maintains the threshold property -/

/-- Greedy invariant: every granted pass is worth at least every expert's next available pass. -/
def GreedyInv {E : Type*} (Δ : E → ℕ → ℝ) (p : E → ℕ) : Prop :=
  ∀ e e' j, j < p e → Δ e' (p e') ≤ Δ e j

/-- The empty allocation satisfies the invariant. -/
theorem greedyInv_zero {E : Type*} (Δ : E → ℕ → ℝ) : GreedyInv Δ (fun _ => 0) :=
  fun _ _ _ hj => absurd hj (Nat.not_lt_zero _)

/-- One greedy step, granting a pass to an expert with the largest next gain, keeps the
invariant. -/
theorem greedyInv_step {E : Type*} [DecidableEq E]
    {Δ : E → ℕ → ℝ} {p : E → ℕ} {i : E}
    (hanti : ∀ e, Antitone (Δ e)) (hinv : GreedyInv Δ p)
    (hmax : ∀ e, Δ e (p e) ≤ Δ i (p i)) :
    GreedyInv Δ (Function.update p i (p i + 1)) := by
  intro e e' j hj
  have hmono : p e' ≤ Function.update p i (p i + 1) e' := by
    by_cases h : e' = i
    · subst h; simp
    · simp [h]
  have hstep : Δ e' (Function.update p i (p i + 1) e') ≤ Δ e' (p e') := hanti e' hmono
  by_cases hjp : j < p e
  · exact hstep.trans (hinv e e' j hjp)
  · have hei : e = i := by
      by_contra hne
      simp [hne] at hj
      exact hjp hj
    subst hei
    have hje : j = p e := by
      simp at hj
      omega
    subst hje
    exact hstep.trans (hmax e')

/-- The invariant yields the threshold property with price `θ = max_e Δ e (p e)`. -/
theorem greedyInv_threshold {E : Type*} [Fintype E] [Nonempty E]
    {Δ : E → ℕ → ℝ} {p : E → ℕ}
    (hnonneg : ∀ e j, 0 ≤ Δ e j) (hanti : ∀ e, Antitone (Δ e)) (hinv : GreedyInv Δ p) :
    ∃ θ : ℝ, 0 ≤ θ ∧ (∀ e j, j < p e → θ ≤ Δ e j) ∧ (∀ e j, p e ≤ j → Δ e j ≤ θ) := by
  refine ⟨univ.sup' univ_nonempty fun e => Δ e (p e), ?_, ?_, ?_⟩
  · obtain ⟨e⟩ := ‹Nonempty E›
    exact (hnonneg e (p e)).trans (le_sup' (fun e => Δ e (p e)) (mem_univ e))
  · intro e j hj
    exact sup'_le _ _ fun e' _ => hinv e e' j hj
  · intro e j hj
    exact (hanti e hj).trans (le_sup' (fun e => Δ e (p e)) (mem_univ e))

/-- Any allocation satisfying the greedy invariant is optimal for the budget it spends. -/
theorem greedyInv_optimal {E : Type*} [Fintype E] [Nonempty E]
    {Δ : E → ℕ → ℝ} {pstar p : E → ℕ}
    (hnonneg : ∀ e j, 0 ≤ Δ e j) (hanti : ∀ e, Antitone (Δ e)) (hinv : GreedyInv Δ pstar)
    (hbudget : ∑ e, p e ≤ ∑ e, pstar e) :
    benefit Δ p ≤ benefit Δ pstar := by
  obtain ⟨θ, hθ, hlo, hhi⟩ := greedyInv_threshold hnonneg hanti hinv
  exact threshold_optimal hθ hlo hhi hbudget

/-- Capped greedy invariant: every granted pass is worth at least the next pass of every
expert that is still below its cap. -/
def CappedGreedyInv {E : Type*} (Δ : E → ℕ → ℝ) (cap p : E → ℕ) : Prop :=
  ∀ e e' j, j < p e → p e' < cap e' → Δ e' (p e') ≤ Δ e j

/-- The empty allocation satisfies the capped invariant. -/
theorem cappedGreedyInv_zero {E : Type*} (Δ : E → ℕ → ℝ) (cap : E → ℕ) :
    CappedGreedyInv Δ cap (fun _ => 0) :=
  fun _ _ _ hj _ => absurd hj (Nat.not_lt_zero _)

/-- One capped greedy step, granting a pass to a best expert among those below cap, keeps the
capped invariant. -/
theorem cappedGreedyInv_step {E : Type*} [DecidableEq E]
    {Δ : E → ℕ → ℝ} {cap p : E → ℕ} {i : E}
    (hanti : ∀ e, Antitone (Δ e)) (hinv : CappedGreedyInv Δ cap p)
    (hmax : ∀ e, p e < cap e → Δ e (p e) ≤ Δ i (p i)) :
    CappedGreedyInv Δ cap (Function.update p i (p i + 1)) := by
  intro e e' j hj hcap'
  have hmono : p e' ≤ Function.update p i (p i + 1) e' := by
    by_cases h : e' = i
    · subst h; simp
    · simp [h]
  have hstep : Δ e' (Function.update p i (p i + 1) e') ≤ Δ e' (p e') := hanti e' hmono
  have hbelow : p e' < cap e' := lt_of_le_of_lt hmono hcap'
  by_cases hjp : j < p e
  · exact hstep.trans (hinv e e' j hjp hbelow)
  · have hei : e = i := by
      by_contra hne
      simp [hne] at hj
      exact hjp hj
    subst hei
    have hje : j = p e := by
      simp at hj
      omega
    subst hje
    exact hstep.trans (hmax e' hbelow)

/-- The capped invariant yields the capped threshold property. The price is the largest next
gain among experts below cap, or `0` when every expert is at its cap. -/
theorem cappedGreedyInv_threshold {E : Type*} [Fintype E]
    {Δ : E → ℕ → ℝ} {cap p : E → ℕ}
    (hnonneg : ∀ e j, 0 ≤ Δ e j) (hanti : ∀ e, Antitone (Δ e))
    (hinv : CappedGreedyInv Δ cap p) :
    ∃ θ : ℝ, 0 ≤ θ ∧ (∀ e j, j < p e → θ ≤ Δ e j) ∧
      (∀ e j, p e ≤ j → j < cap e → Δ e j ≤ θ) := by
  classical
  by_cases hS : (univ.filter fun e => p e < cap e).Nonempty
  · refine ⟨(univ.filter fun e => p e < cap e).sup' hS fun e => Δ e (p e), ?_, ?_, ?_⟩
    · obtain ⟨e, he⟩ := hS
      exact (hnonneg e (p e)).trans (le_sup' (fun e => Δ e (p e)) he)
    · intro e j hj
      exact sup'_le _ _ fun e' he' => hinv e e' j hj (mem_filter.mp he').2
    · intro e j hj hjc
      have he : e ∈ univ.filter fun e => p e < cap e :=
        mem_filter.mpr ⟨mem_univ e, lt_of_le_of_lt hj hjc⟩
      exact (hanti e hj).trans (le_sup' (fun e => Δ e (p e)) he)
  · refine ⟨0, le_rfl, fun e j _ => hnonneg e j, ?_⟩
    intro e j hj hjc
    exact absurd ⟨e, mem_filter.mpr ⟨mem_univ e, lt_of_le_of_lt hj hjc⟩⟩ hS

/-- Any allocation satisfying the capped greedy invariant is optimal among cap-respecting
allocations that spend no more passes. -/
theorem cappedGreedyInv_optimal {E : Type*} [Fintype E]
    {Δ : E → ℕ → ℝ} {cap pstar p : E → ℕ}
    (hnonneg : ∀ e j, 0 ≤ Δ e j) (hanti : ∀ e, Antitone (Δ e))
    (hinv : CappedGreedyInv Δ cap pstar) (hcap : ∀ e, p e ≤ cap e)
    (hbudget : ∑ e, p e ≤ ∑ e, pstar e) :
    benefit Δ p ≤ benefit Δ pstar := by
  obtain ⟨θ, hθ, hlo, hhi⟩ := cappedGreedyInv_threshold hnonneg hanti hinv
  exact threshold_optimal_capped hθ hlo hhi hcap hbudget

/-! ### 3. The recursive greedy allocation (uncapped) -/

/-- Some expert maximizes the next marginal benefit. -/
theorem exists_argmax {E : Type*} [Fintype E] [Nonempty E] (Δ : E → ℕ → ℝ) (p : E → ℕ) :
    ∃ i, ∀ e, Δ e (p e) ≤ Δ i (p i) := by
  obtain ⟨i, -, hi⟩ := exists_max_image univ (fun e => Δ e (p e)) univ_nonempty
  exact ⟨i, fun e => hi e (mem_univ e)⟩

/-- A chosen expert with the largest next marginal benefit. -/
noncomputable def argmax {E : Type*} [Fintype E] [Nonempty E] (Δ : E → ℕ → ℝ) (p : E → ℕ) : E :=
  Classical.choose (exists_argmax Δ p)

/-- The chosen expert is a maximizer. -/
theorem argmax_spec {E : Type*} [Fintype E] [Nonempty E] (Δ : E → ℕ → ℝ) (p : E → ℕ) :
    ∀ e, Δ e (p e) ≤ Δ (argmax Δ p) (p (argmax Δ p)) :=
  Classical.choose_spec (exists_argmax Δ p)

/-- Greedy allocation of `K` passes: repeatedly grant one pass to a best expert.
Caps are omitted in this recursive version; see `cappedGreedy` and `cappedGreedy_optimal`
for the capped recursion. -/
noncomputable def greedy {E : Type*} [Fintype E] [Nonempty E] [DecidableEq E]
    (Δ : E → ℕ → ℝ) : ℕ → E → ℕ
  | 0 => fun _ => 0
  | K + 1 =>
      Function.update (greedy Δ K) (argmax Δ (greedy Δ K))
        (greedy Δ K (argmax Δ (greedy Δ K)) + 1)

/-- Granting one more pass to one expert raises the spent budget by one. -/
theorem sum_update_succ {E : Type*} [Fintype E] [DecidableEq E] (p : E → ℕ) (i : E) :
    ∑ e, Function.update p i (p i + 1) e = ∑ e, p e + 1 := by
  have h : ∀ e, Function.update p i (p i + 1) e = p e + if e = i then 1 else 0 := by
    intro e
    by_cases he : e = i
    · subst he; simp
    · simp [he]
  simp [h, sum_add_distrib]

/-- Greedy spends exactly its budget. -/
theorem greedy_sum {E : Type*} [Fintype E] [Nonempty E] [DecidableEq E]
    (Δ : E → ℕ → ℝ) (K : ℕ) : ∑ e, greedy Δ K e = K := by
  induction K with
  | zero => simp [greedy]
  | succ K ih => rw [greedy, sum_update_succ, ih]

/-- Greedy satisfies the invariant at every budget. -/
theorem greedy_inv {E : Type*} [Fintype E] [Nonempty E] [DecidableEq E]
    {Δ : E → ℕ → ℝ} (hanti : ∀ e, Antitone (Δ e)) (K : ℕ) : GreedyInv Δ (greedy Δ K) := by
  induction K with
  | zero => exact greedyInv_zero Δ
  | succ K ih => exact greedyInv_step hanti ih (argmax_spec Δ (greedy Δ K))

/-- Greedy is optimal: no allocation of at most `K` passes has larger benefit. -/
theorem greedy_optimal {E : Type*} [Fintype E] [Nonempty E] [DecidableEq E]
    {Δ : E → ℕ → ℝ} (hnonneg : ∀ e j, 0 ≤ Δ e j) (hanti : ∀ e, Antitone (Δ e))
    {K : ℕ} {p : E → ℕ} (hp : ∑ e, p e ≤ K) :
    benefit Δ p ≤ benefit Δ (greedy Δ K) :=
  greedyInv_optimal hnonneg hanti (greedy_inv hanti K) (by rwa [greedy_sum])

/-! ### 4. Geometric instance -/

/-- Marginal benefit of pass `j + 1` under geometric residual decay. -/
noncomputable def geomGain {E : Type*} (a D0 γ : E → ℝ) (e : E) (j : ℕ) : ℝ :=
  a e * D0 e * (1 - γ e) * γ e ^ j

/-- Weighted geometric distortion of an allocation. -/
noncomputable def geomObjective {E : Type*} [Fintype E] (a D0 γ : E → ℝ) (p : E → ℕ) : ℝ :=
  ∑ e, a e * D0 e * γ e ^ p e

/-- Geometric gains are nonnegative. -/
theorem geomGain_nonneg {E : Type*} {a D0 γ : E → ℝ}
    (ha : ∀ e, 0 ≤ a e) (hD : ∀ e, 0 ≤ D0 e) (hγ0 : ∀ e, 0 < γ e) (hγ1 : ∀ e, γ e < 1)
    (e : E) (j : ℕ) : 0 ≤ geomGain a D0 γ e j :=
  mul_nonneg (mul_nonneg (mul_nonneg (ha e) (hD e)) (sub_nonneg.mpr (hγ1 e).le))
    (pow_nonneg (hγ0 e).le j)

/-- Geometric gains are antitone in the pass index. -/
theorem geomGain_antitone {E : Type*} {a D0 γ : E → ℝ}
    (ha : ∀ e, 0 ≤ a e) (hD : ∀ e, 0 ≤ D0 e) (hγ0 : ∀ e, 0 < γ e) (hγ1 : ∀ e, γ e < 1)
    (e : E) : Antitone (geomGain a D0 γ e) := fun _ _ hjk =>
  mul_le_mul_of_nonneg_left (pow_le_pow_of_le_one (hγ0 e).le (hγ1 e).le hjk)
    (mul_nonneg (mul_nonneg (ha e) (hD e)) (sub_nonneg.mpr (hγ1 e).le))

/-- Telescoping: the first `P` geometric gains sum to the distortion removed by `P` passes. -/
theorem geomGain_sum {E : Type*} (a D0 γ : E → ℝ) (e : E) (P : ℕ) :
    ∑ j ∈ range P, geomGain a D0 γ e j = a e * D0 e * (1 - γ e ^ P) := by
  induction P with
  | zero => simp
  | succ P ih =>
      rw [sum_range_succ, ih, geomGain]
      ring

/-- Benefit equals distortion removed relative to the empty allocation. -/
theorem geomObjective_zero_sub {E : Type*} [Fintype E] (a D0 γ : E → ℝ) (p : E → ℕ) :
    geomObjective a D0 γ (fun _ => 0) - geomObjective a D0 γ p =
      benefit (geomGain a D0 γ) p := by
  unfold geomObjective benefit
  rw [← sum_sub_distrib]
  refine sum_congr rfl fun e _ => ?_
  rw [geomGain_sum]
  ring

/-- Greedy minimizes the weighted geometric distortion among allocations of at most `K`
passes. -/
theorem greedy_minimizes_geometric {E : Type*} [Fintype E] [Nonempty E] [DecidableEq E]
    {a D0 γ : E → ℝ}
    (ha : ∀ e, 0 ≤ a e) (hD : ∀ e, 0 ≤ D0 e) (hγ0 : ∀ e, 0 < γ e) (hγ1 : ∀ e, γ e < 1)
    {K : ℕ} {p : E → ℕ} (hp : ∑ e, p e ≤ K) :
    geomObjective a D0 γ (greedy (geomGain a D0 γ) K) ≤ geomObjective a D0 γ p := by
  have h := greedy_optimal (geomGain_nonneg ha hD hγ0 hγ1) (geomGain_antitone ha hD hγ0 hγ1) hp
  rw [← geomObjective_zero_sub, ← geomObjective_zero_sub] at h
  linarith

/-! ### 5. Unequal costs break ratio greedy -/

/-- Item benefits of the two-item knapsack counterexample. -/
def knapBenefit : Fin 2 → ℕ := ![3, 4]

/-- Item costs of the two-item knapsack counterexample. -/
def knapCost : Fin 2 → ℕ := ![2, 3]

/-- With unequal costs and budget 3, ratio greedy picks item 0 (ratio 3/2 > 4/3), cannot then
afford item 1, and stops at value 3, while the feasible set `{1}` attains the optimum 4. -/
theorem ratio_greedy_suboptimal :
    (knapBenefit 1 : ℚ) / knapCost 1 < (knapBenefit 0 : ℚ) / knapCost 0 ∧
    knapCost 0 ≤ 3 ∧
    3 < knapCost 0 + knapCost 1 ∧
    ∑ i ∈ ({0} : Finset (Fin 2)), knapBenefit i = 3 ∧
    ∑ i ∈ ({1} : Finset (Fin 2)), knapCost i ≤ 3 ∧
    ∑ i ∈ ({1} : Finset (Fin 2)), knapBenefit i = 4 ∧
    (∀ S : Finset (Fin 2), ∑ i ∈ S, knapCost i ≤ 3 → ∑ i ∈ S, knapBenefit i ≤ 4) := by
  refine ⟨by norm_num [knapBenefit, knapCost], by decide, by decide, by decide, by decide,
    by decide, by decide⟩

/-! ### 6. The recursive capped greedy allocation -/

/-- If some expert is below its cap, some below-cap expert maximizes the next marginal
benefit among the below-cap experts. -/
theorem exists_cappedArgmax {E : Type*} [Fintype E] (Δ : E → ℕ → ℝ) (cap p : E → ℕ)
    (h : ∃ e, p e < cap e) :
    ∃ i, p i < cap i ∧ ∀ e, p e < cap e → Δ e (p e) ≤ Δ i (p i) := by
  classical
  obtain ⟨e0, he0⟩ := h
  obtain ⟨i, hi, hmax⟩ := exists_max_image (univ.filter fun e => p e < cap e)
    (fun e => Δ e (p e)) ⟨e0, mem_filter.mpr ⟨mem_univ e0, he0⟩⟩
  exact ⟨i, (mem_filter.mp hi).2, fun e he => hmax e (mem_filter.mpr ⟨mem_univ e, he⟩)⟩

/-- A chosen best expert among those below cap. -/
noncomputable def cappedArgmax {E : Type*} [Fintype E] (Δ : E → ℕ → ℝ) (cap p : E → ℕ)
    (h : ∃ e, p e < cap e) : E :=
  Classical.choose (exists_cappedArgmax Δ cap p h)

/-- The chosen expert is below its cap and maximizes the next gain among below-cap experts. -/
theorem cappedArgmax_spec {E : Type*} [Fintype E] (Δ : E → ℕ → ℝ) (cap p : E → ℕ)
    (h : ∃ e, p e < cap e) :
    p (cappedArgmax Δ cap p h) < cap (cappedArgmax Δ cap p h) ∧
      ∀ e, p e < cap e →
        Δ e (p e) ≤ Δ (cappedArgmax Δ cap p h) (p (cappedArgmax Δ cap p h)) :=
  Classical.choose_spec (exists_cappedArgmax Δ cap p h)

open Classical in
/-- One capped greedy step: grant a pass to a best below-cap expert, or do nothing when every
expert is at its cap. -/
noncomputable def cappedStep {E : Type*} [Fintype E] [DecidableEq E]
    (Δ : E → ℕ → ℝ) (cap p : E → ℕ) : E → ℕ :=
  if h : ∃ e, p e < cap e then
    Function.update p (cappedArgmax Δ cap p h) (p (cappedArgmax Δ cap p h) + 1)
  else p

/-- Capped greedy allocation of `K` passes: repeatedly grant one pass to a best expert among
those strictly below cap; once all experts are at cap the allocation stops changing. -/
noncomputable def cappedGreedy {E : Type*} [Fintype E] [DecidableEq E]
    (Δ : E → ℕ → ℝ) (cap : E → ℕ) : ℕ → E → ℕ
  | 0 => fun _ => 0
  | K + 1 => cappedStep Δ cap (cappedGreedy Δ cap K)

/-- A capped step keeps the allocation within the caps. -/
theorem cappedStep_le_cap {E : Type*} [Fintype E] [DecidableEq E]
    (Δ : E → ℕ → ℝ) {cap p : E → ℕ} (hp : ∀ e, p e ≤ cap e) (e : E) :
    cappedStep Δ cap p e ≤ cap e := by
  unfold cappedStep
  split_ifs with h
  · by_cases he : e = cappedArgmax Δ cap p h
    · subst he
      simpa using (cappedArgmax_spec Δ cap p h).1
    · simpa [he] using hp e
  · exact hp e

/-- Within the caps, some expert is below cap exactly when the caps are not exhausted. -/
theorem exists_lt_cap_iff {E : Type*} [Fintype E] {cap p : E → ℕ} (hp : ∀ e, p e ≤ cap e) :
    (∃ e, p e < cap e) ↔ ∑ e, p e < ∑ e, cap e := by
  constructor
  · rintro ⟨e, he⟩
    exact sum_lt_sum (fun e _ => hp e) ⟨e, mem_univ e, he⟩
  · intro hlt
    by_contra hne
    have heq : ∀ e, p e = cap e := fun e =>
      le_antisymm (hp e) (not_lt.mp fun hlt' => hne ⟨e, hlt'⟩)
    simp [heq] at hlt

/-- A capped step spends one more pass unless the caps are exhausted. -/
theorem cappedStep_sum {E : Type*} [Fintype E] [DecidableEq E]
    (Δ : E → ℕ → ℝ) {cap p : E → ℕ} (hp : ∀ e, p e ≤ cap e) :
    ∑ e, cappedStep Δ cap p e = min (∑ e, p e + 1) (∑ e, cap e) := by
  unfold cappedStep
  split_ifs with h
  · have hlt := (exists_lt_cap_iff hp).mp h
    rw [sum_update_succ]
    omega
  · have hge : ¬ ∑ e, p e < ∑ e, cap e := fun hlt => h ((exists_lt_cap_iff hp).mpr hlt)
    have hle : ∑ e, p e ≤ ∑ e, cap e := sum_le_sum fun e _ => hp e
    omega

/-- A capped step keeps the capped greedy invariant. -/
theorem cappedStep_inv {E : Type*} [Fintype E] [DecidableEq E]
    {Δ : E → ℕ → ℝ} {cap p : E → ℕ}
    (hanti : ∀ e, Antitone (Δ e)) (hinv : CappedGreedyInv Δ cap p) :
    CappedGreedyInv Δ cap (cappedStep Δ cap p) := by
  unfold cappedStep
  split_ifs with h
  · exact cappedGreedyInv_step hanti hinv (cappedArgmax_spec Δ cap p h).2
  · exact hinv

/-- Capped greedy respects the caps at every budget. -/
theorem cappedGreedy_le_cap {E : Type*} [Fintype E] [DecidableEq E]
    (Δ : E → ℕ → ℝ) (cap : E → ℕ) (K : ℕ) (e : E) : cappedGreedy Δ cap K e ≤ cap e := by
  induction K generalizing e with
  | zero => exact Nat.zero_le _
  | succ K ih => exact cappedStep_le_cap Δ ih e

/-- Capped greedy spends its budget, up to the total capacity. -/
theorem cappedGreedy_sum {E : Type*} [Fintype E] [DecidableEq E]
    (Δ : E → ℕ → ℝ) (cap : E → ℕ) (K : ℕ) :
    ∑ e, cappedGreedy Δ cap K e = min K (∑ e, cap e) := by
  induction K with
  | zero => simp [cappedGreedy]
  | succ K ih =>
      rw [cappedGreedy, cappedStep_sum Δ (cappedGreedy_le_cap Δ cap K), ih]
      omega

/-- Once the budget reaches the total capacity, capped greedy is exactly the cap vector. -/
theorem cappedGreedy_eq_cap {E : Type*} [Fintype E] [DecidableEq E]
    (Δ : E → ℕ → ℝ) (cap : E → ℕ) {K : ℕ} (hK : ∑ e, cap e ≤ K) :
    cappedGreedy Δ cap K = cap := by
  funext e
  by_contra hne
  have hlt : cappedGreedy Δ cap K e < cap e :=
    lt_of_le_of_ne (cappedGreedy_le_cap Δ cap K e) hne
  have hsum := (exists_lt_cap_iff (cappedGreedy_le_cap Δ cap K)).mp ⟨e, hlt⟩
  rw [cappedGreedy_sum] at hsum
  omega

/-- Capped greedy satisfies the capped invariant at every budget. -/
theorem cappedGreedy_inv {E : Type*} [Fintype E] [DecidableEq E]
    {Δ : E → ℕ → ℝ} (hanti : ∀ e, Antitone (Δ e)) (cap : E → ℕ) (K : ℕ) :
    CappedGreedyInv Δ cap (cappedGreedy Δ cap K) := by
  induction K with
  | zero => exact cappedGreedyInv_zero Δ cap
  | succ K ih => exact cappedStep_inv hanti ih

/-- Capped greedy is optimal: no cap-respecting allocation of at most `K` passes has larger
benefit. When `K` exceeds the total capacity the greedy allocation is the cap vector, and the
budget comparison needed by `cappedGreedyInv_optimal` still holds because every cap-respecting
allocation spends at most the total capacity. -/
theorem cappedGreedy_optimal {E : Type*} [Fintype E] [DecidableEq E]
    {Δ : E → ℕ → ℝ} (hnonneg : ∀ e j, 0 ≤ Δ e j) (hanti : ∀ e, Antitone (Δ e))
    {cap : E → ℕ} {K : ℕ} {p : E → ℕ} (hcap : ∀ e, p e ≤ cap e) (hp : ∑ e, p e ≤ K) :
    benefit Δ p ≤ benefit Δ (cappedGreedy Δ cap K) := by
  refine cappedGreedyInv_optimal hnonneg hanti (cappedGreedy_inv hanti cap K) hcap ?_
  rw [cappedGreedy_sum]
  exact le_min hp (sum_le_sum fun e _ => hcap e)

/-- Capped greedy minimizes the weighted geometric distortion among cap-respecting allocations
of at most `K` passes. -/
theorem cappedGreedy_minimizes_geometric {E : Type*} [Fintype E] [DecidableEq E]
    {a D0 γ : E → ℝ}
    (ha : ∀ e, 0 ≤ a e) (hD : ∀ e, 0 ≤ D0 e) (hγ0 : ∀ e, 0 < γ e) (hγ1 : ∀ e, γ e < 1)
    {cap : E → ℕ} {K : ℕ} {p : E → ℕ} (hcap : ∀ e, p e ≤ cap e) (hp : ∑ e, p e ≤ K) :
    geomObjective a D0 γ (cappedGreedy (geomGain a D0 γ) cap K) ≤ geomObjective a D0 γ p := by
  have h := cappedGreedy_optimal (geomGain_nonneg ha hD hγ0 hγ1)
    (geomGain_antitone ha hD hγ0 hγ1) hcap hp
  rw [← geomObjective_zero_sub, ← geomObjective_zero_sub] at h
  linarith

end Progressive.GreedyOptimal
