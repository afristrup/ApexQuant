import Mathlib

open Finset MeasureTheory ProbabilityTheory

namespace ApexQuant.Bounds

/-! ### 1. Markov tail for the decay law -/

/-- Markov tail: a nonnegative integrable `X` with mean `m` exceeds `m / δ` with probability at
most `δ`. -/
theorem markov_tail {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} (hX : ∀ ω, 0 ≤ X ω) (hint : Integrable X μ) {m δ : ℝ}
    (hm : ∫ ω, X ω ∂μ = m) (hδ : 0 < δ) :
    μ.real {ω | m / δ < X ω} ≤ δ := by
  have hm0 : 0 ≤ m := hm ▸ integral_nonneg hX
  rcases hm0.eq_or_lt with h0 | hpos
  · have hae : X =ᵐ[μ] 0 := (integral_eq_zero_iff_of_nonneg hX hint).mp (hm.trans h0.symm)
    have hnull : μ {ω | m / δ < X ω} = 0 := by
      refine measure_mono_null (fun ω hω => ?_) (ae_iff.mp hae)
      have hω' : 0 < X ω := by simpa [← h0] using hω
      simpa using hω'.ne'
    simp [measureReal_def, hnull, hδ.le]
  · have hmono : μ.real {ω | m / δ < X ω} ≤ μ.real {ω | m / δ ≤ X ω} :=
      measureReal_mono fun ω (hω : m / δ < X ω) => le_of_lt hω
    have hmarkov := mul_meas_ge_le_integral_of_nonneg (Filter.Eventually.of_forall hX) hint (m / δ)
    rw [hm] at hmarkov
    have hq : 0 < m / δ := div_pos hpos hδ
    have hle : μ.real {ω | m / δ ≤ X ω} ≤ δ := by
      have h1 : m / δ * μ.real {ω | m / δ ≤ X ω} ≤ m / δ * δ := by
        rw [div_mul_cancel₀ m hδ.ne']
        exact hmarkov
      exact le_of_mul_le_mul_left h1 hq
    exact hmono.trans hle

/-- Paper form of the Markov tail: if `E X = ρ2P * w2` (think `X = ‖r_P‖²`), then
`sqrt X > sqrt (ρ2P / δ) * sqrt w2` with probability at most `δ`. -/
theorem markov_tail_sqrt {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} (hX : ∀ ω, 0 ≤ X ω) (hint : Integrable X μ)
    {ρ2P w2 δ : ℝ} (hρ : 0 ≤ ρ2P) (hw : 0 ≤ w2) (hm : ∫ ω, X ω ∂μ = ρ2P * w2)
    (hδ : 0 < δ) (_hδ1 : δ ≤ 1) :
    μ.real {ω | Real.sqrt (ρ2P / δ) * Real.sqrt w2 < Real.sqrt (X ω)} ≤ δ := by
  have hset : {ω | Real.sqrt (ρ2P / δ) * Real.sqrt w2 < Real.sqrt (X ω)} =
      {ω | ρ2P * w2 / δ < X ω} := by
    ext ω
    have hq : 0 ≤ ρ2P * w2 / δ := div_nonneg (mul_nonneg hρ hw) hδ.le
    have he : Real.sqrt (ρ2P / δ) * Real.sqrt w2 = Real.sqrt (ρ2P * w2 / δ) := by
      rw [← Real.sqrt_mul (div_nonneg hρ hδ.le)]
      congr 1
      ring
    simp only [Set.mem_setOf_eq, he]
    exact Real.sqrt_lt_sqrt_iff hq
  rw [hset]
  exact markov_tail hX hint hm hδ

/-! ### 2. Jensen for the square root -/

/-- Jensen: the mean of `sqrt X` is at most the square root of the mean of `X`. -/
theorem integral_sqrt_le_sqrt_integral {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} (hX : ∀ ω, 0 ≤ X ω) (hint : Integrable X μ) :
    ∫ ω, Real.sqrt (X ω) ∂μ ≤ Real.sqrt (∫ ω, X ω ∂μ) := by
  have hsqrt : Integrable (Real.sqrt ∘ X) μ := by
    refine Integrable.mono' (hint.add (integrable_const (1 : ℝ)))
      (Real.continuous_sqrt.comp_aestronglyMeasurable hint.aestronglyMeasurable)
      (Filter.Eventually.of_forall fun ω => ?_)
    simp only [Function.comp_apply, Pi.add_apply, Real.norm_eq_abs,
      abs_of_nonneg (Real.sqrt_nonneg _)]
    nlinarith [Real.sq_sqrt (hX ω), Real.sqrt_nonneg (X ω), sq_nonneg (Real.sqrt (X ω) - 1)]
  exact Real.strictConcaveOn_sqrt.concaveOn.le_map_integral
    Real.continuous_sqrt.continuousOn isClosed_Ici
    (Filter.Eventually.of_forall fun ω => hX ω) hint hsqrt

/-! ### 3. Mixed-stage decay -/

/-- Mixed-stage decay: per-stage contraction factors multiply. -/
theorem mixed_stage_decay {a ρsq : ℕ → ℝ} (h : ∀ p, a (p + 1) = ρsq p * a p) (P : ℕ) :
    a P = (∏ p ∈ range P, ρsq p) * a 0 := by
  induction P with
  | zero => simp
  | succ P ih =>
      rw [h, ih, prod_range_succ]
      ring

/-- Constant-rate decay: `a P = ρ2 ^ P * a 0`. -/
theorem constant_stage_decay {a : ℕ → ℝ} {ρ2 : ℝ} (h : ∀ p, a (p + 1) = ρ2 * a p) (P : ℕ) :
    a P = ρ2 ^ P * a 0 := by
  have := mixed_stage_decay (ρsq := fun _ => ρ2) h P
  simpa using this

/-! ### 4. Pass rule -/

/-- Pass rule: `P ≥ log ε / log ρ` passes bring a `ρ`-contraction below `ε`. -/
theorem pass_rule {ρ ε : ℝ} (hρ0 : 0 < ρ) (hρ1 : ρ < 1) (hε0 : 0 < ε) (_hε1 : ε < 1)
    {P : ℕ} (hP : Real.log ε / Real.log ρ ≤ P) : ρ ^ P ≤ ε := by
  have hlog : Real.log ρ < 0 := Real.log_neg hρ0 hρ1
  have h1 : (P : ℝ) * Real.log ρ ≤ Real.log ε := (div_le_iff_of_neg hlog).mp hP
  have h2 : Real.log (ρ ^ P) ≤ Real.log ε := by
    rw [Real.log_pow]
    exact h1
  exact (Real.log_le_log_iff (pow_pos hρ0 P) hε0).mp h2

/-! ### 5. Rounded scales -/

/-- Rounded scale, pointwise: perturbing the scale `s` by a relative error `η` with `|η| ≤ εs`
costs at most `s * εs * ‖v‖` on top of the shape error. -/
theorem rounded_scale_norm_le {H : Type*} [NormedAddCommGroup H] [NormedSpace ℝ H]
    (u v : H) {s η εs : ℝ} (hs : 0 ≤ s) (hη : |η| ≤ εs) :
    ‖s • (u - v) - (s * η) • v‖ ≤ s * (‖u - v‖ + εs * ‖v‖) := by
  have heq : s • (u - v) - (s * η) • v = s • ((u - v) - η • v) := by
    simp only [smul_sub, mul_smul]
  rw [heq, norm_smul, Real.norm_eq_abs, abs_of_nonneg hs]
  refine mul_le_mul_of_nonneg_left ?_ hs
  calc ‖(u - v) - η • v‖ ≤ ‖u - v‖ + ‖η • v‖ := norm_sub_le _ _
    _ = ‖u - v‖ + |η| * ‖v‖ := by rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ‖u - v‖ + εs * ‖v‖ := by
        have := mul_le_mul_of_nonneg_right hη (norm_nonneg v)
        linarith

/-- Cauchy-Schwarz in `L2` for real functions. -/
theorem abs_integral_mul_le_sqrt_mul_sqrt {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    {f g : Ω → ℝ} (hf : MemLp f 2 μ) (hg : MemLp g 2 μ) :
    |∫ ω, f ω * g ω ∂μ| ≤
      Real.sqrt (∫ ω, f ω ^ 2 ∂μ) * Real.sqrt (∫ ω, g ω ^ 2 ∂μ) := by
  have hf2 : Integrable (fun ω => f ω ^ 2) μ := hf.integrable_sq
  have hg2 : Integrable (fun ω => g ω ^ 2) μ := hg.integrable_sq
  have hfg : Integrable (fun ω => f ω * g ω) μ := hf.integrable_mul hg
  have hquad : ∀ t : ℝ, 0 ≤ (∫ ω, f ω ^ 2 ∂μ) * (t * t) +
      (-2 * ∫ ω, f ω * g ω ∂μ) * t + ∫ ω, g ω ^ 2 ∂μ := by
    intro t
    have hexp : ∫ ω, (t * f ω - g ω) ^ 2 ∂μ = (∫ ω, f ω ^ 2 ∂μ) * (t * t) +
        (-2 * ∫ ω, f ω * g ω ∂μ) * t + ∫ ω, g ω ^ 2 ∂μ := by
      have hfun : (fun ω => (t * f ω - g ω) ^ 2) =
          fun ω => (t * t) * f ω ^ 2 + (-2 * t) * (f ω * g ω) + g ω ^ 2 := by
        funext ω
        ring
      have i1 : Integrable (fun ω => (t * t) * f ω ^ 2) μ := hf2.const_mul _
      have i2 : Integrable (fun ω => (-2 * t) * (f ω * g ω)) μ := hfg.const_mul _
      have i3 : Integrable (fun ω => (t * t) * f ω ^ 2 + (-2 * t) * (f ω * g ω)) μ :=
        i1.add i2
      rw [hfun, integral_add i3 hg2, integral_add i1 i2, integral_const_mul,
        integral_const_mul]
      ring
    rw [← hexp]
    exact integral_nonneg fun ω => sq_nonneg _
  have hdisc := discrim_le_zero hquad
  rw [discrim] at hdisc
  rw [← Real.sqrt_mul (integral_nonneg fun ω => sq_nonneg (f ω))]
  refine Real.abs_le_sqrt ?_
  nlinarith

/-- Rounded scale, second moment for real envelopes: with `D = E f²` and `M = E g²`,
`E (f + εs g)² ≤ (sqrt D + εs sqrt M)²` (Minkowski in `L2`). -/
theorem integral_add_mul_sq_le {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    {f g : Ω → ℝ} (hf : MemLp f 2 μ) (hg : MemLp g 2 μ) {εs : ℝ} (hε : 0 ≤ εs) :
    ∫ ω, (f ω + εs * g ω) ^ 2 ∂μ ≤
      (Real.sqrt (∫ ω, f ω ^ 2 ∂μ) + εs * Real.sqrt (∫ ω, g ω ^ 2 ∂μ)) ^ 2 := by
  have hf2 : Integrable (fun ω => f ω ^ 2) μ := hf.integrable_sq
  have hg2 : Integrable (fun ω => g ω ^ 2) μ := hg.integrable_sq
  have hfg : Integrable (fun ω => f ω * g ω) μ := hf.integrable_mul hg
  have hcs := (le_abs_self _).trans (abs_integral_mul_le_sqrt_mul_sqrt hf hg)
  have hD := Real.sq_sqrt (integral_nonneg fun ω => sq_nonneg (f ω) : 0 ≤ ∫ ω, f ω ^ 2 ∂μ)
  have hM := Real.sq_sqrt (integral_nonneg fun ω => sq_nonneg (g ω) : 0 ≤ ∫ ω, g ω ^ 2 ∂μ)
  have hfun : (fun ω => (f ω + εs * g ω) ^ 2) =
      fun ω => f ω ^ 2 + (2 * εs) * (f ω * g ω) + (εs * εs) * g ω ^ 2 := by
    funext ω
    ring
  have i1 : Integrable (fun ω => (2 * εs) * (f ω * g ω)) μ := hfg.const_mul _
  have i2 : Integrable (fun ω => (εs * εs) * g ω ^ 2) μ := hg2.const_mul _
  have i3 : Integrable (fun ω => f ω ^ 2 + (2 * εs) * (f ω * g ω)) μ := hf2.add i1
  rw [hfun, integral_add i3 i2, integral_add hf2 i1, integral_const_mul, integral_const_mul]
  nlinarith [mul_le_mul_of_nonneg_left hcs (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) hε)]

/-- Rounded scale, second-moment form: with `D = E ‖U - q U‖²` and `M = E ‖q U‖²`,
`E (‖U - q U‖ + εs ‖q U‖)² ≤ (sqrt D + εs sqrt M)²`. -/
theorem rounded_scale_second_moment {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    {U : Ω → H} {q : H → H}
    (hUm : AEStronglyMeasurable (fun ω => U ω - q (U ω)) μ)
    (hqm : AEStronglyMeasurable (fun ω => q (U ω)) μ)
    (hD : Integrable (fun ω => ‖U ω - q (U ω)‖ ^ 2) μ)
    (hM : Integrable (fun ω => ‖q (U ω)‖ ^ 2) μ) {εs : ℝ} (hε : 0 ≤ εs) :
    ∫ ω, (‖U ω - q (U ω)‖ + εs * ‖q (U ω)‖) ^ 2 ∂μ ≤
      (Real.sqrt (∫ ω, ‖U ω - q (U ω)‖ ^ 2 ∂μ) +
        εs * Real.sqrt (∫ ω, ‖q (U ω)‖ ^ 2 ∂μ)) ^ 2 :=
  integral_add_mul_sq_le ((memLp_two_iff_integrable_sq hUm.norm).mpr hD)
    ((memLp_two_iff_integrable_sq hqm.norm).mpr hM) hε

/-! ### 6. Two-layer expert bound -/

/-- Two-layer expert certificate: quantizing both layers of `x ↦ W2 (σ (W1 x))` with an
`L`-Lipschitz activation fixing `0` costs at most `L (U2 ‖W1‖ + ‖W2h‖ U1) ‖x‖`. -/
theorem two_layer_bound {E F G : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F]
    [NormedAddCommGroup G] [NormedSpace ℝ G]
    (W1 W1h : E →L[ℝ] F) (W2 W2h : F →L[ℝ] G) (σ : F → F) {L U1 U2 : ℝ}
    (hσ : ∀ x y, ‖σ x - σ y‖ ≤ L * ‖x - y‖) (hσ0 : σ 0 = 0)
    (h1 : ‖W1 - W1h‖ ≤ U1) (h2 : ‖W2 - W2h‖ ≤ U2) (x : E) :
    ‖W2 (σ (W1 x)) - W2h (σ (W1h x))‖ ≤ L * (U2 * ‖W1‖ + ‖W2h‖ * U1) * ‖x‖ := by
  by_cases hL : 0 ≤ L
  · have hσnorm : ‖σ (W1 x)‖ ≤ L * (‖W1‖ * ‖x‖) := by
      have h := hσ (W1 x) 0
      rw [hσ0, sub_zero, sub_zero] at h
      exact h.trans (mul_le_mul_of_nonneg_left (W1.le_opNorm x) hL)
    have hdiff : ‖σ (W1 x) - σ (W1h x)‖ ≤ L * (U1 * ‖x‖) := by
      refine (hσ _ _).trans (mul_le_mul_of_nonneg_left ?_ hL)
      have h : ‖W1 x - W1h x‖ ≤ ‖W1 - W1h‖ * ‖x‖ := (W1 - W1h).le_opNorm x
      exact h.trans (mul_le_mul_of_nonneg_right h1 (norm_nonneg x))
    have hU2 : 0 ≤ U2 := (norm_nonneg _).trans h2
    have ha : ‖(W2 - W2h) (σ (W1 x))‖ ≤ U2 * (L * (‖W1‖ * ‖x‖)) :=
      ((W2 - W2h).le_opNorm _).trans
        (mul_le_mul h2 hσnorm (norm_nonneg _) hU2)
    have hb : ‖W2h (σ (W1 x) - σ (W1h x))‖ ≤ ‖W2h‖ * (L * (U1 * ‖x‖)) :=
      (W2h.le_opNorm _).trans (mul_le_mul_of_nonneg_left hdiff (norm_nonneg _))
    have heq : W2 (σ (W1 x)) - W2h (σ (W1h x)) =
        (W2 - W2h) (σ (W1 x)) + W2h (σ (W1 x) - σ (W1h x)) := by
      simp [map_sub]
    rw [heq]
    calc ‖(W2 - W2h) (σ (W1 x)) + W2h (σ (W1 x) - σ (W1h x))‖
        ≤ ‖(W2 - W2h) (σ (W1 x))‖ + ‖W2h (σ (W1 x) - σ (W1h x))‖ := norm_add_le _ _
      _ ≤ U2 * (L * (‖W1‖ * ‖x‖)) + ‖W2h‖ * (L * (U1 * ‖x‖)) := add_le_add ha hb
      _ = L * (U2 * ‖W1‖ + ‖W2h‖ * U1) * ‖x‖ := by ring
  · have hF : ∀ y : F, y = 0 := by
      intro y
      have h := (norm_nonneg _).trans (hσ y 0)
      rw [sub_zero] at h
      by_contra hy
      exact hL (nonneg_of_mul_nonneg_left h (norm_pos_iff.mpr hy))
    have hW1 : W1 = 0 := ContinuousLinearMap.ext fun y => hF _
    have hW2h : W2h = 0 := ContinuousLinearMap.ext fun y => by
      simp [hF y]
    rw [hF (σ (W1 x)), hF (σ (W1h x)), hW1, hW2h]
    simp

/-! ### 7. Linear layer certificate -/

/-- Linear layer certificate in operator norm. -/
theorem linear_layer_bound {E F : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F]
    (W Wh : E →L[ℝ] F) {U : ℝ} (hU : ‖W - Wh‖ ≤ U) (x : E) :
    ‖(W - Wh) x‖ ≤ U * ‖x‖ :=
  ((W - Wh).le_opNorm x).trans (mul_le_mul_of_nonneg_right hU (norm_nonneg x))

/-- Frobenius norm of a real matrix. -/
noncomputable def frob {m n : Type*} [Fintype m] [Fintype n] (A : Matrix m n ℝ) : ℝ :=
  Real.sqrt (∑ i, ∑ j, A i j ^ 2)

/-- Linear layer certificate in Frobenius norm: `‖A x‖₂ ≤ ‖A‖_F ‖x‖₂`, by Cauchy-Schwarz row
by row. -/
theorem mulVec_norm_le_frob {m n : Type*} [Fintype m] [Fintype n]
    (A : Matrix m n ℝ) (x : n → ℝ) :
    Real.sqrt (∑ i, (A.mulVec x) i ^ 2) ≤ frob A * Real.sqrt (∑ j, x j ^ 2) := by
  unfold frob
  rw [← Real.sqrt_mul (sum_nonneg fun i _ => sum_nonneg fun j _ => sq_nonneg (A i j))]
  refine Real.sqrt_le_sqrt ?_
  rw [sum_mul]
  refine sum_le_sum fun i _ => ?_
  exact sum_mul_sq_le_sq_mul_sq univ (fun j => A i j) x

/-! ### 8. Shared-sign variance -/

/-- Covariance Cauchy-Schwarz. -/
theorem covariance_le_sqrt_variance_mul {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X Y : Ω → ℝ} (hX : MemLp X 2 μ) (hY : MemLp Y 2 μ) :
    cov[X, Y; μ] ≤ Real.sqrt (Var[X; μ]) * Real.sqrt (Var[Y; μ]) := by
  rw [variance_eq_integral hX.aemeasurable, variance_eq_integral hY.aemeasurable, covariance]
  exact (le_abs_self _).trans
    (abs_integral_mul_le_sqrt_mul_sqrt (hX.sub (memLp_const _)) (hY.sub (memLp_const _)))

/-- Shared-sign variance, Minkowski form: the standard deviation of a sum is at most the sum
of the standard deviations, with no independence assumption. -/
theorem variance_sum_le_sq_sum_sqrt {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {G : Type*} [Fintype G] {L : G → Ω → ℝ}
    (hL : ∀ g, MemLp (L g) 2 μ) :
    Var[fun ω => ∑ g, L g ω; μ] ≤ (∑ g, Real.sqrt (Var[L g; μ])) ^ 2 := by
  rw [variance_fun_sum hL, sq, sum_mul_sum]
  exact sum_le_sum fun g _ => sum_le_sum fun g' _ =>
    covariance_le_sqrt_variance_mul (hL g) (hL g')

/-- Shared-sign variance, counting form: the variance of a sum of `card G` terms is at most
`card G` times the sum of the variances. -/
theorem variance_sum_le_card_mul_sum {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {G : Type*} [Fintype G] {L : G → Ω → ℝ}
    (hL : ∀ g, MemLp (L g) 2 μ) :
    Var[fun ω => ∑ g, L g ω; μ] ≤ (Fintype.card G : ℝ) * ∑ g, Var[L g; μ] := by
  refine (variance_sum_le_sq_sum_sqrt hL).trans ?_
  have h := sq_sum_le_card_mul_sum_sq (s := (univ : Finset G))
    (f := fun g => Real.sqrt (Var[L g; μ]))
  simp only [Real.sq_sqrt (variance_nonneg _ _), card_univ] at h
  exact h

/-! ### 9. Shrinkage -/

/-- Shrinkage of the mean residual: `m (p + 1) = β • m p` gives `m P = β ^ P • m 0`. -/
theorem shrinkage_iterate {V : Type*} [AddCommGroup V] [Module ℝ V] {m : ℕ → V} {β : ℝ}
    (h : ∀ p, m (p + 1) = β • m p) (P : ℕ) : m P = β ^ P • m 0 := by
  induction P with
  | zero => simp
  | succ P ih => rw [h, ih, smul_smul, pow_succ']

/-- Shrinkage bias: starting from `m 0 = w`, the decoded mean is `(1 - β ^ P) • w`. -/
theorem shrinkage_bias {V : Type*} [AddCommGroup V] [Module ℝ V] {m : ℕ → V} {β : ℝ} {w : V}
    (h : ∀ p, m (p + 1) = β • m p) (h0 : m 0 = w) (P : ℕ) :
    w - m P = (1 - β ^ P) • w := by
  rw [shrinkage_iterate h P, h0, sub_smul, one_smul]

/-! ### 10. Triangle envelope for a routed sum -/

/-- Routed-sum envelope in attention-value form: weight error times the value bound plus the
weighted value errors. -/
theorem routed_sum_envelope {J H : Type*} [Fintype J] [NormedAddCommGroup H] [NormedSpace ℝ H]
    {v vh : J → H} {α αh : J → ℝ} {V : ℝ} {Uv : J → ℝ}
    (hαh : ∀ j, 0 ≤ αh j) (hV : ∀ j, ‖v j‖ ≤ V) (hU : ∀ j, ‖v j - vh j‖ ≤ Uv j) :
    ‖∑ j, α j • v j - ∑ j, αh j • vh j‖ ≤
      V * ∑ j, |α j - αh j| + ∑ j, αh j * Uv j := by
  have heq : ∑ j, α j • v j - ∑ j, αh j • vh j =
      ∑ j, ((α j - αh j) • v j + αh j • (v j - vh j)) := by
    rw [← sum_sub_distrib]
    refine sum_congr rfl fun j _ => ?_
    rw [sub_smul, smul_sub]
    abel
  rw [heq, mul_sum, ← sum_add_distrib]
  refine (norm_sum_le _ _).trans (sum_le_sum fun j _ => ?_)
  refine (norm_add_le _ _).trans (add_le_add ?_ ?_)
  · rw [norm_smul, Real.norm_eq_abs, mul_comm]
    exact mul_le_mul_of_nonneg_right (hV j) (abs_nonneg _)
  · rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hαh j)]
    exact mul_le_mul_of_nonneg_left (hU j) (hαh j)

/-- Routed-sum envelope with total variation `TV = (1/2) ∑ |α j - αh j|`: the weight term is
`2 V TV`. -/
theorem routed_sum_envelope_tv {J H : Type*} [Fintype J] [NormedAddCommGroup H]
    [NormedSpace ℝ H] {v vh : J → H} {α αh : J → ℝ} {V : ℝ} {Uv : J → ℝ}
    (hαh : ∀ j, 0 ≤ αh j) (hV : ∀ j, ‖v j‖ ≤ V) (hU : ∀ j, ‖v j - vh j‖ ≤ Uv j) :
    ‖∑ j, α j • v j - ∑ j, αh j • vh j‖ ≤
      2 * V * ((1 / 2) * ∑ j, |α j - αh j|) + ∑ j, αh j * Uv j := by
  have h := routed_sum_envelope (α := α) hαh hV hU
  linarith

end ApexQuant.Bounds
