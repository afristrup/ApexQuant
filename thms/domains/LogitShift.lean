import Mathlib

open Finset

namespace ApexQuant.DomainShift

/-- Finite token-logit vectors and token probability vectors. -/
def Logits (V : Type*) := V → ℝ
def TokenDistribution (V : Type*) := V → ℝ

/-- Total variation distance between two token distributions. -/
noncomputable def totalVariation {V : Type*} [Fintype V]
    (p q : TokenDistribution V) : ℝ :=
  (1 / 2 : ℝ) * ∑ v, |p v - q v|

/-- A finite-domain softmax implementation with an explicit Lipschitz contract. -/
def SoftmaxLipschitz {V : Type*} [Fintype V]
    (softmax : Logits V → TokenDistribution V) (L : ℝ) : Prop :=
  ∀ z z' δ, 0 ≤ δ →
    (∀ v, |z v - z' v| ≤ δ) →
    totalVariation (softmax z) (softmax z') ≤ L * δ


/-- The action of the finite softmax Jacobian at a probability vector. -/
def softmaxJacobianAction {V : Type*} [Fintype V]
    (p h : V → ℝ) : V → ℝ :=
  fun i => p i * (h i - ∑ j, p j * h j)

/-- The finite softmax Jacobian maps an `L∞` perturbation of size `B` to a
 total-variation-sized perturbation of at most `B`. -/
theorem softmax_jacobian_tv_le_one {V : Type*} [Fintype V]
    {p h : V → ℝ} {B : ℝ}
    (hp : ∀ i, 0 ≤ p i) (hsum : ∑ i, p i = 1)
    (_hB : 0 ≤ B) (hh : ∀ i, |h i| ≤ B) :
    (1 / 2 : ℝ) * ∑ i, |softmaxJacobianAction p h i| ≤ B := by
  have hmean : |∑ j, p j * h j| ≤ B := by
    calc
      |∑ j, p j * h j| ≤ ∑ j, |p j * h j| := by
        simpa using norm_sum_le (Finset.univ : Finset V) (fun j => p j * h j)
      _ = ∑ j, p j * |h j| := by
        apply Finset.sum_congr rfl
        intro j _
        rw [abs_mul, abs_of_nonneg (hp j)]
      _ ≤ ∑ j, p j * B := by
        apply Finset.sum_le_sum
        intro j _
        exact mul_le_mul_of_nonneg_left (hh j) (hp j)
      _ = B := by
        rw [← Finset.sum_mul, hsum, one_mul]
  have hterm : ∀ i, |softmaxJacobianAction p h i| ≤ p i * (2 * B) := by
    intro i
    rw [softmaxJacobianAction, abs_mul, abs_of_nonneg (hp i)]
    calc
      p i * |h i - ∑ j, p j * h j| ≤
          p i * (|h i| + |∑ j, p j * h j|) := by
        apply mul_le_mul_of_nonneg_left _ (hp i)
        simpa using abs_sub_le (h i) 0 (∑ j, p j * h j)
      _ ≤ p i * (2 * B) := by
        apply mul_le_mul_of_nonneg_left _ (hp i)
        linarith [hh i, hmean]
  calc
    (1 / 2 : ℝ) * ∑ i, |softmaxJacobianAction p h i| ≤
        (1 / 2 : ℝ) * ∑ i, p i * (2 * B) := by
      apply mul_le_mul_of_nonneg_left
      · exact Finset.sum_le_sum (fun i _ => hterm i)
      · norm_num
    _ = B := by
      rw [← Finset.sum_mul (Finset.univ : Finset V) p (2 * B), hsum]
      ring

/-- Weighted target-domain token-distribution shift. -/
noncomputable def weightedTokenShift {X V : Type*} [Fintype X] [Fintype V]
    (μ : X → ℝ) (softmax : Logits V → TokenDistribution V)
    (reference candidate : X → Logits V) : ℝ :=
  ∑ x, μ x * totalVariation (softmax (candidate x)) (softmax (reference x))

/-- A logit envelope transfers through any declared softmax Lipschitz bound. -/
theorem weighted_token_shift_le_of_logit_envelope
    {X V : Type*} [Fintype X] [Fintype V]
    {μ : X → ℝ} {softmax : Logits V → TokenDistribution V}
    {reference candidate : X → Logits V} {δ : X → ℝ} {L : ℝ}
    (hμ : ∀ x, 0 ≤ μ x)
    (hsoft : SoftmaxLipschitz softmax L)
    (hδ : ∀ x, 0 ≤ δ x)
    (hlogit : ∀ x v, |candidate x v - reference x v| ≤ δ x) :
    weightedTokenShift μ softmax reference candidate ≤
      ∑ x, μ x * (L * δ x) := by
  unfold weightedTokenShift
  apply Finset.sum_le_sum
  intro x _
  exact mul_le_mul_of_nonneg_left
    (hsoft (candidate x) (reference x) (δ x) (hδ x) (hlogit x)) (hμ x)

/-- For normalized target distributions and a uniform logit envelope, the
output-distribution shift is bounded by the softmax Lipschitz constant times
that envelope. -/
theorem weighted_token_shift_le_of_uniform_logit_envelope
    {X V : Type*} [Fintype X] [Fintype V]
    {μ : X → ℝ} {softmax : Logits V → TokenDistribution V}
    {reference candidate : X → Logits V} {L δ : ℝ}
    (hμ : ∀ x, 0 ≤ μ x) (hsum : ∑ x, μ x = 1)
    (hδ : 0 ≤ δ)
    (hsoft : SoftmaxLipschitz softmax L)
    (hlogit : ∀ x v, |candidate x v - reference x v| ≤ δ) :
    weightedTokenShift μ softmax reference candidate ≤ L * δ := by
  calc
    weightedTokenShift μ softmax reference candidate ≤
        ∑ x, μ x * (L * δ) := by
      apply weighted_token_shift_le_of_logit_envelope hμ hsoft
      · intro x
        exact hδ
      · intro x v
        exact hlogit x v
    _ = (∑ x, μ x) * (L * δ) := by
      simpa using (Finset.sum_mul (Finset.univ : Finset X) μ (L * δ)).symm
    _ = L * δ := by rw [hsum, one_mul]


/-- Finite categorical KL divergence. -/
noncomputable def klDivergence {V : Type*} [Fintype V]
    (p q : TokenDistribution V) : ℝ :=
  ∑ v, p v * Real.log (p v / q v)

/-- Jensen-Shannon divergence formed from the midpoint distribution. -/
noncomputable def jsDivergence {V : Type*} [Fintype V]
    (p q : TokenDistribution V) : ℝ :=
  (1 / 2 : ℝ) *
    (klDivergence p (fun v => (p v + q v) / 2) +
      klDivergence q (fun v => (p v + q v) / 2))

/-- A pointwise KL-versus-TV inequality transfers to the weighted target
 domain without requiring target samples during quantization. -/
theorem weighted_kl_shift_le_of_tv_bound
    {X V : Type*} [Fintype X] [Fintype V]
    {μ : X → ℝ} {softmax : Logits V → TokenDistribution V}
    {reference candidate : X → Logits V} {C : ℝ}
    (hμ : ∀ x, 0 ≤ μ x)
    (hbound : ∀ x,
      klDivergence (softmax (candidate x)) (softmax (reference x)) ≤
        C * totalVariation (softmax (candidate x)) (softmax (reference x))) :
    (∑ x, μ x * klDivergence (softmax (candidate x))
      (softmax (reference x))) ≤
      ∑ x, μ x * (C * totalVariation (softmax (candidate x))
        (softmax (reference x))) := by
  apply Finset.sum_le_sum
  intro x _
  exact mul_le_mul_of_nonneg_left (hbound x) (hμ x)

/-- The analogous transfer theorem for Jensen-Shannon divergence. -/
theorem weighted_js_shift_le_of_tv_bound
    {X V : Type*} [Fintype X] [Fintype V]
    {μ : X → ℝ} {softmax : Logits V → TokenDistribution V}
    {reference candidate : X → Logits V} {C : ℝ}
    (hμ : ∀ x, 0 ≤ μ x)
    (hbound : ∀ x,
      jsDivergence (softmax (candidate x)) (softmax (reference x)) ≤
        C * totalVariation (softmax (candidate x)) (softmax (reference x))) :
    (∑ x, μ x * jsDivergence (softmax (candidate x))
      (softmax (reference x))) ≤
      ∑ x, μ x * (C * totalVariation (softmax (candidate x))
        (softmax (reference x))) := by
  apply Finset.sum_le_sum
  intro x _
  exact mul_le_mul_of_nonneg_left (hbound x) (hμ x)



/-- The logarithm is `1 / η`-Lipschitz on positive reals bounded below by `η`. -/
theorem abs_log_div_le_of_floor {a b η : ℝ}
    (hη : 0 < η) (ha : η ≤ a) (hb : η ≤ b) :
    |Real.log (a / b)| ≤ |a - b| / η := by
  have ha0 : 0 < a := lt_of_lt_of_le hη ha
  have hb0 : 0 < b := lt_of_lt_of_le hη hb
  rcases le_total a b with hab | hba
  · have hdiff : 0 ≤ b - a := sub_nonneg.mpr hab
    have hratio : b / a - 1 = (b - a) / a := by
      field_simp
    have hbound : b / a - 1 ≤ (b - a) / η := by
      rw [hratio]
      exact div_le_div_of_nonneg_left hdiff hη ha
    have hlog : Real.log (b / a) ≤ (b - a) / η := by
      exact le_trans (Real.log_le_sub_one_of_pos (div_pos hb0 ha0)) hbound
    have hneg : Real.log (a / b) = -Real.log (b / a) := by
      rw [Real.log_div ha0.ne' hb0.ne', Real.log_div hb0.ne' ha0.ne']
      ring
    rw [abs_of_nonpos (sub_nonpos.mpr hab), abs_le]
    constructor
    · rw [hneg]
      have hneglog := neg_le_neg hlog
      simpa [sub_eq_add_neg] using hneglog
    · have hzero : Real.log (a / b) ≤ 0 := by
        exact Real.log_nonpos (le_of_lt (div_pos ha0 hb0))
          ((div_le_one₀ hb0).2 hab)
      exact hzero.trans (div_nonneg (neg_nonneg.mpr (sub_nonpos.mpr hab)) hη.le)
  · have hdiff : 0 ≤ a - b := sub_nonneg.mpr hba
    have hratio : a / b - 1 = (a - b) / b := by
      field_simp
    have hbound : a / b - 1 ≤ (a - b) / η := by
      rw [hratio]
      exact div_le_div_of_nonneg_left hdiff hη hb
    have hlog : Real.log (a / b) ≤ (a - b) / η := by
      exact le_trans (Real.log_le_sub_one_of_pos (div_pos ha0 hb0)) hbound
    have hnonneg : 0 ≤ Real.log (a / b) :=
      Real.log_nonneg ((one_le_div₀ hb0).2 hba)
    rw [abs_of_nonneg hnonneg, abs_of_nonneg hdiff]
    exact hlog


/-- Every token probability is bounded away from zero by `η`. -/
def ProbabilityFloor {V : Type*} (η : ℝ) (p : TokenDistribution V) : Prop :=
  0 < η ∧ ∀ v, η ≤ p v

/-- A probability vector has nonnegative entries and unit mass. -/
def ProbabilitySimplex {V : Type*} [Fintype V]
    (p : TokenDistribution V) : Prop :=
  (∀ v, 0 ≤ p v) ∧ ∑ v, p v = 1

/-- Each midpoint KL summand is bounded by its weighted absolute
probability difference, without a probability floor. -/
theorem js_term_le_weighted_abs {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    a * Real.log (a / ((a + b) / 2)) ≤
      (a / (a + b)) * |a - b| := by
  by_cases ha0 : a = 0
  · simp [ha0]
  have ha_pos : 0 < a := lt_of_le_of_ne ha (Ne.symm ha0)
  have hab_pos : 0 < a + b := add_pos_of_pos_of_nonneg ha_pos hb
  have hm_pos : 0 < (a + b) / 2 := by positivity
  have hlog := Real.log_le_sub_one_of_pos (div_pos ha_pos hm_pos)
  have hmul := mul_le_mul_of_nonneg_left hlog ha
  calc
    a * Real.log (a / ((a + b) / 2)) ≤
        a * (a / ((a + b) / 2) - 1) := hmul
    _ = (a / (a + b)) * (a - b) := by
      field_simp
      ring
    _ ≤ (a / (a + b)) * |a - b| := by
      exact mul_le_mul_of_nonneg_left (le_abs_self _) (div_nonneg ha (add_nonneg ha hb))

/-- The two midpoint KL summands are bounded by one absolute probability
difference. -/
theorem js_pair_le_abs {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    a * Real.log (a / ((a + b) / 2)) +
        b * Real.log (b / ((a + b) / 2)) ≤ |a - b| := by
  have ha_term := js_term_le_weighted_abs ha hb
  have hb_term :
      b * Real.log (b / ((a + b) / 2)) ≤
        (b / (a + b)) * |b - a| := by
    simpa [add_comm] using js_term_le_weighted_abs hb ha
  calc
    a * Real.log (a / ((a + b) / 2)) +
        b * Real.log (b / ((a + b) / 2)) ≤
        (a / (a + b)) * |a - b| +
          (b / (a + b)) * |b - a| := add_le_add ha_term hb_term
    _ = ((a / (a + b)) + (b / (a + b))) * |a - b| := by
      rw [abs_sub_comm b a]
      ring
    _ ≤ |a - b| := by
      by_cases hab : a + b = 0
      · simp [hab]
      · have hab_pos : 0 < a + b := lt_of_le_of_ne (add_nonneg ha hb) (Ne.symm hab)
        have hratio : a / (a + b) + b / (a + b) = 1 := by
          field_simp
        rw [hratio, one_mul]

/-- Jensen-Shannon divergence is bounded by total variation with no floor. -/
theorem js_le_tv {V : Type*} [Fintype V]
    {p q : TokenDistribution V}
    (hp : ProbabilitySimplex p) (hq : ProbabilitySimplex q) :
    jsDivergence p q ≤ totalVariation p q := by
  unfold jsDivergence klDivergence totalVariation
  have hpair : ∀ v,
      p v * Real.log (p v / ((p v + q v) / 2)) +
          q v * Real.log (q v / ((p v + q v) / 2)) ≤ |p v - q v| := by
    intro v
    exact js_pair_le_abs (hp.1 v) (hq.1 v)
  calc
    (1 / 2 : ℝ) *
        (∑ v, p v * Real.log (p v / ((p v + q v) / 2)) +
          ∑ v, q v * Real.log (q v / ((p v + q v) / 2))) =
        (1 / 2 : ℝ) * ∑ v,
          (p v * Real.log (p v / ((p v + q v) / 2)) +
            q v * Real.log (q v / ((p v + q v) / 2))) := by
      rw [Finset.sum_add_distrib]
    _ ≤ (1 / 2 : ℝ) * ∑ v, |p v - q v| := by
      apply mul_le_mul_of_nonneg_left
      · exact Finset.sum_le_sum (fun v _ => hpair v)
      · norm_num

/-- The floor and simplex hypotheses imply the KL summand bound. -/
theorem probability_log_term_le
    {V : Type*} [Fintype V]
    {p q : TokenDistribution V} {η : ℝ}
    (hp : ProbabilityFloor η p) (hq : ProbabilityFloor η q)
    (hsimplex : ProbabilitySimplex p) (v : V) :
    p v * Real.log (p v / q v) ≤ |p v - q v| / η := by
  have hp_le_one : p v ≤ 1 := by
    rw [← hsimplex.2]
    exact Finset.single_le_sum (fun i _ => hsimplex.1 i) (Finset.mem_univ v)
  have hlog := abs_log_div_le_of_floor hp.1 (hp.2 v) (hq.2 v)
  calc
    p v * Real.log (p v / q v) ≤ p v * |Real.log (p v / q v)| := by
      exact mul_le_mul_of_nonneg_left (le_abs_self _) (hsimplex.1 v)
    _ ≤ 1 * |Real.log (p v / q v)| := by
      exact mul_le_mul_of_nonneg_right hp_le_one (abs_nonneg _)
    _ ≤ |p v - q v| / η := by simpa using hlog

/-- With a probability floor, KL is linearly bounded by total variation. The
floor is explicit because no such KL upper bound exists uniformly when
probabilities may approach zero. -/
theorem kl_le_tv_of_probability_floor
    {V : Type*} [Fintype V]
    {p q : TokenDistribution V} {η : ℝ}
    (hp : ProbabilityFloor η p) (hq : ProbabilityFloor η q)
    (hsimplex : ProbabilitySimplex p) :
    klDivergence p q ≤ (2 / η) * totalVariation p q := by
  unfold klDivergence totalVariation
  have hsum :
      (∑ v, p v * Real.log (p v / q v)) ≤
        ∑ v, |p v - q v| / η := by
    exact Finset.sum_le_sum (fun v _ => probability_log_term_le hp hq hsimplex v)
  calc
    ∑ v, p v * Real.log (p v / q v) ≤
        ∑ v, |p v - q v| / η := hsum
    _ = (2 / η) * ((1 / 2 : ℝ) * ∑ v, |p v - q v|) := by
      simp_rw [div_eq_mul_inv]
      rw [← Finset.sum_mul (Finset.univ : Finset V)]
      ring

/-- Jensen-Shannon has a floor-conditioned total-variation bound. -/
theorem js_le_tv_of_probability_floor
    {V : Type*} [Fintype V]
    {p q : TokenDistribution V} {η : ℝ}
    (hp : ProbabilityFloor η p) (hq : ProbabilityFloor η q)
    (hsimplexp : ProbabilitySimplex p)
    (hsimplexq : ProbabilitySimplex q) :
    jsDivergence p q ≤ (1 / η) * totalVariation p q := by
  unfold jsDivergence
  have hpkl : klDivergence p (fun v => (p v + q v) / 2) ≤
      (1 / η) * totalVariation p q := by
    unfold klDivergence totalVariation
    calc
      ∑ v, p v * Real.log (p v / ((p v + q v) / 2)) ≤
          ∑ v, |p v - q v| / (2 * η) :=
        Finset.sum_le_sum (fun v _ => by
          have hm : ProbabilityFloor η (fun v => (p v + q v) / 2) := by
            constructor
            · exact hp.1
            · intro v
              linarith [hp.2 v, hq.2 v]
          have h := probability_log_term_le hp hm hsimplexp v
          calc
            p v * Real.log (p v / ((p v + q v) / 2)) ≤
                |p v - (p v + q v) / 2| / η := h
            _ = |p v - q v| / (2 * η) := by
              rw [show p v - (p v + q v) / 2 = (p v - q v) / 2 by ring,
                abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
              ring
        )
      _ = (1 / η) * ((1 / 2 : ℝ) * ∑ v, |p v - q v|) := by
        simp_rw [div_eq_mul_inv]
        rw [← Finset.sum_mul (Finset.univ : Finset V)]
        ring
  have hqkl : klDivergence q (fun v => (p v + q v) / 2) ≤
      (1 / η) * totalVariation p q := by
    unfold klDivergence totalVariation
    calc
      ∑ v, q v * Real.log (q v / ((p v + q v) / 2)) ≤
          ∑ v, |q v - p v| / (2 * η) :=
        Finset.sum_le_sum (fun v _ => by
          have hm : ProbabilityFloor η (fun v => (p v + q v) / 2) := by
            constructor
            · exact hp.1
            · intro v
              linarith [hp.2 v, hq.2 v]
          have h := probability_log_term_le hq hm hsimplexq v
          calc
            q v * Real.log (q v / ((p v + q v) / 2)) ≤
                |q v - (p v + q v) / 2| / η := h
            _ = |q v - p v| / (2 * η) := by
              rw [show q v - (p v + q v) / 2 = (q v - p v) / 2 by ring,
                abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
              ring
        )
      _ = (1 / η) * ((1 / 2 : ℝ) * ∑ v, |p v - q v|) := by
        simp_rw [div_eq_mul_inv]
        rw [← Finset.sum_mul (Finset.univ : Finset V)]
        have habs : (∑ v, |q v - p v|) = ∑ v, |p v - q v| := by
          apply Finset.sum_congr rfl
          intro v _
          rw [abs_sub_comm]
        rw [habs]
        ring
  calc
    (1 / 2 : ℝ) *
        (klDivergence p (fun v => (p v + q v) / 2) +
          klDivergence q (fun v => (p v + q v) / 2)) ≤
        (1 / 2 : ℝ) *
          ((1 / η) * totalVariation p q +
            (1 / η) * totalVariation p q) := by
      gcongr
    _ = (1 / η) * totalVariation p q := by ring

/-- A linear-logit perturbation is bounded by its operator norm. -/
theorem linear_logit_error_le
    {E F : Type*} [SeminormedAddCommGroup E] [SeminormedAddCommGroup F]
    [NormedSpace ℝ E] [NormedSpace ℝ F]
    (delta : E →L[ℝ] F) (activation : E) :
    ‖delta activation‖ ≤ ‖delta‖ * ‖activation‖ := by
  exact delta.le_opNorm activation

end ApexQuant.DomainShift
