import Mathlib

open Finset

namespace Progressive.DynamicProgram

/-- An allocation `p` is feasible for the first `n` experts under budget `B`:
each option is within its cap and the total cost is within the budget. -/
def Feasible (cap : ℕ → ℕ) (C : ℕ → ℕ → ℕ) (n B : ℕ) (p : ℕ → ℕ) : Prop :=
  (∀ j, j < n → p j ≤ cap j) ∧ ∑ j ∈ range n, C j (p j) ≤ B

/-- Total value of an allocation over the first `n` experts. -/
def total (v : ℕ → ℕ → ℝ) (n : ℕ) (p : ℕ → ℕ) : ℝ :=
  ∑ j ∈ range n, v j (p j)

/-- The knapsack recursion. `F n b` is the least total value of the first `n`
experts under budget `b`, with `⊤` encoding infeasibility. -/
noncomputable def F (cap : ℕ → ℕ) (C : ℕ → ℕ → ℕ) (v : ℕ → ℕ → ℝ) :
    ℕ → ℕ → WithTop ℝ
  | 0, _ => 0
  | j + 1, b =>
      ((range (cap j + 1)).filter fun p => C j p ≤ b).inf
        fun p => F cap C v j (b - C j p) + (v j p : WithTop ℝ)

variable {cap : ℕ → ℕ} {C : ℕ → ℕ → ℕ} {v : ℕ → ℕ → ℝ}

/-- Lower bound: the recursion is below the value of every feasible
allocation. No assumption on the costs is needed. -/
theorem F_le_of_feasible (n B : ℕ) (p : ℕ → ℕ) (hp : Feasible cap C n B p) :
    F cap C v n B ≤ (total v n p : WithTop ℝ) := by
  induction n generalizing B with
  | zero => simp [F, total]
  | succ n ih =>
      obtain ⟨hcap, hcost⟩ := hp
      rw [sum_range_succ] at hcost
      have hmem : p n ∈ (range (cap n + 1)).filter fun q => C n q ≤ B := by
        have := hcap n (Nat.lt_succ_self n)
        simp only [mem_filter, mem_range]
        omega
      have hfeas : Feasible cap C n (B - C n (p n)) p :=
        ⟨fun j hj => hcap j (Nat.lt_succ_of_lt hj), by omega⟩
      calc F cap C v (n + 1) B
          ≤ F cap C v n (B - C n (p n)) + (v n (p n) : WithTop ℝ) :=
            Finset.inf_le (f := fun q => F cap C v n (B - C n q) + (v n q : WithTop ℝ)) hmem
        _ ≤ (total v n p : WithTop ℝ) + (v n (p n) : WithTop ℝ) :=
            add_le_add_left (ih _ hfeas) _
        _ = (total v (n + 1) p : WithTop ℝ) := by
            rw [total, total, sum_range_succ, WithTop.coe_add]

/-- Attainment: whenever the recursion is finite, some feasible allocation has
exactly that value. -/
theorem exists_feasible_of_ne_top (n B : ℕ) (hne : F cap C v n B ≠ ⊤) :
    ∃ p, Feasible cap C n B p ∧ F cap C v n B = (total v n p : WithTop ℝ) := by
  induction n generalizing B with
  | zero =>
      exact ⟨fun _ => 0, ⟨fun _ hj => absurd hj (Nat.not_lt_zero _), by simp⟩,
        by simp [F, total]⟩
  | succ n ih =>
      set s := (range (cap n + 1)).filter fun q => C n q ≤ B with hs
      have hF : F cap C v (n + 1) B =
          s.inf fun q => F cap C v n (B - C n q) + (v n q : WithTop ℝ) := rfl
      have hsne : s.Nonempty := by
        by_contra hempty
        rw [not_nonempty_iff_eq_empty] at hempty
        rw [hF, hempty, inf_empty] at hne
        exact hne rfl
      obtain ⟨q, hq, hqeq⟩ := exists_mem_eq_inf s hsne
        fun q => F cap C v n (B - C n q) + (v n q : WithTop ℝ)
      rw [← hF] at hqeq
      have hprev : F cap C v n (B - C n q) ≠ ⊤ := by
        intro htop
        rw [hqeq, htop, top_add] at hne
        exact hne rfl
      obtain ⟨p, ⟨hpcap, hpcost⟩, hpeq⟩ := ih (B - C n q) hprev
      rw [hs, mem_filter, mem_range] at hq
      have hupd : ∀ j ∈ range n, Function.update p n q j = p j := fun j hj =>
        Function.update_of_ne (Nat.ne_of_lt (mem_range.mp hj)) _ _
      have hsumC : ∑ j ∈ range n, C j (Function.update p n q j) =
          ∑ j ∈ range n, C j (p j) := sum_congr rfl fun j hj => by rw [hupd j hj]
      have hsumv : ∑ j ∈ range n, v j (Function.update p n q j) =
          ∑ j ∈ range n, v j (p j) := sum_congr rfl fun j hj => by rw [hupd j hj]
      refine ⟨Function.update p n q, ⟨fun j hj => ?_, ?_⟩, ?_⟩
      · rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hlt | rfl
        · rw [Function.update_of_ne (Nat.ne_of_lt hlt)]
          exact hpcap j hlt
        · rw [Function.update_self]
          omega
      · rw [sum_range_succ, Function.update_self, hsumC]
        omega
      · rw [hqeq, hpeq, total, total, sum_range_succ, Function.update_self, hsumv,
          WithTop.coe_add]

/-- The recursion is infeasible exactly when no allocation is feasible. -/
theorem F_eq_top_iff (n B : ℕ) :
    F cap C v n B = ⊤ ↔ ¬ ∃ p, Feasible cap C n B p := by
  constructor
  · rintro htop ⟨p, hp⟩
    have h := F_le_of_feasible (v := v) n B p hp
    rw [htop, top_le_iff] at h
    exact WithTop.coe_ne_top h
  · intro hno
    by_contra hne
    obtain ⟨p, hp, _⟩ := exists_feasible_of_ne_top n B hne
    exact hno ⟨p, hp⟩

/-- If the zero option of every expert is free, the zero allocation is
feasible under every budget. -/
theorem feasible_zero (hC0 : ∀ j, C j 0 = 0) (n B : ℕ) :
    Feasible cap C n B fun _ => 0 :=
  ⟨fun _ _ => Nat.zero_le _, by simp [hC0]⟩

/-- Exactness of the dynamic program, assuming the zero option is free
(`C j 0 = 0`), which makes every budget feasible: `F n B` is a real number,
it is attained by a feasible allocation, and that allocation is optimal. -/
theorem dp_exact (hC0 : ∀ j, C j 0 = 0) (n B : ℕ) :
    ∃ p, Feasible cap C n B p ∧ F cap C v n B = (total v n p : WithTop ℝ) ∧
      ∀ p', Feasible cap C n B p' → total v n p ≤ total v n p' := by
  have hne : F cap C v n B ≠ ⊤ := fun htop =>
    (F_eq_top_iff n B).mp htop ⟨_, feasible_zero hC0 n B⟩
  obtain ⟨p, hp, hpeq⟩ := exists_feasible_of_ne_top n B hne
  refine ⟨p, hp, hpeq, fun p' hp' => ?_⟩
  have h := F_le_of_feasible (v := v) n B p' hp'
  rw [hpeq] at h
  exact WithTop.coe_le_coe.mp h

/-- Exactness without any cost assumption: if some allocation is feasible, the
recursion returns the minimum of the total value over feasible allocations. -/
theorem dp_isLeast {n B : ℕ} (hfeas : ∃ p, Feasible cap C n B p) :
    ∃ m : ℝ, F cap C v n B = (m : WithTop ℝ) ∧
      IsLeast {t | ∃ p, Feasible cap C n B p ∧ t = total v n p} m := by
  have hne : F cap C v n B ≠ ⊤ := fun htop => (F_eq_top_iff n B).mp htop hfeas
  obtain ⟨p, hp, hpeq⟩ := exists_feasible_of_ne_top n B hne
  refine ⟨total v n p, hpeq, ⟨p, hp, rfl⟩, ?_⟩
  rintro t ⟨p', hp', rfl⟩
  have h := F_le_of_feasible (v := v) n B p' hp'
  rw [hpeq] at h
  exact WithTop.coe_le_coe.mp h

end Progressive.DynamicProgram
