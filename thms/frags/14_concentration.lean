/-! ## §13  Concentration: what a per-pass variance bound actually buys

Proposition 2 (§3) is an identity in *expectation*.  The paper's Remark ("Concentration")
claims more: that `err(P) = ρ^P` holds "essentially deterministically", on the grounds that the
per-pass ratio `‖r_{p+1}‖²/‖r_p‖²` has mean `ρ²` and variance `O(1/d)`.

That variance estimate is not formalised here; §25 later derives it, with an explicit constant
and for every nearest-codeword stage, from the Poincaré inequality on the sphere taken as a
hypothesis, and item 5 of the header records what remains.  What is formalised here is the half the prose
leaves implicit: the passage from a *per-pass* deviation bound to a *`P`-pass* one.

That passage is not free, and writing it down is the point of this section.  The per-pass ratios
multiply, so relative deviations compound as `(1 ± ε)^P`; the failure events union, so the
confidence degrades as `Pδ`.  Both prices are invisible in the statement `err(P) = ρ^P`, and
both are what "essentially deterministically" has to mean if it is to mean anything. -/

section Concentration

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The pass-`p` squared normalised stage error `‖u_p − q(u_p)‖²`.  This is the paper's per-pass
ratio `‖r_{p+1}‖²/‖r_p‖²`, defined without the division so that it is total: by
`norm_resid_succ_sq` the two agree whenever `r_p ≠ 0`, and at `r_p = 0` the ratio is undefined
while this is not. -/
noncomputable def stageErrSq (S : Scheme E) (w : E) (p : ℕ) : ℝ :=
  ‖unitVec S w p - S.q (unitVec S w p)‖ ^ 2

theorem stageErrSq_nonneg (S : Scheme E) (w : E) (p : ℕ) : 0 ≤ stageErrSq S w p := sq_nonneg _

/-- **The product form of the recursion.**  `‖r_P‖² = ‖w‖² · ∏_{p<P} ‖u_p − q(u_p)‖²`.

Deterministic and hypothesis-free: this is `norm_resid_succ_sq` iterated, and it is the reason
concentration of the decay law is a statement about a *product* rather than a sum. -/
theorem norm_resid_sq_eq_prod (S : Scheme E) (w : E) (P : ℕ) :
    ‖resid S w P‖ ^ 2 = ‖w‖ ^ 2 * ∏ p ∈ range P, stageErrSq S w p := by
  induction P with
  | zero => simp
  | succ P ih =>
      rw [norm_resid_succ_sq, ih, Finset.prod_range_succ]
      simp only [stageErrSq]
      ring

/-- `DecayWithin S w ρ ε P`: the `P`-pass squared error lies inside the band the decay law
predicts, widened by the compounded relative factor `(1 ± ε)^P`. -/
def DecayWithin (S : Scheme E) (w : E) (ρ ε : ℝ) (P : ℕ) : Prop :=
  (1 - ε) ^ P * ρ ^ (2 * P) * ‖w‖ ^ 2 ≤ ‖resid S w P‖ ^ 2 ∧
    ‖resid S w P‖ ^ 2 ≤ (1 + ε) ^ P * ρ ^ (2 * P) * ‖w‖ ^ 2

/-- **The deterministic sandwich.**  If each of the first `P` passes has its ratio within a
relative `ε` of `ρ²`, then the `P`-pass squared error is within a relative `(1 ± ε)^P` of
`ρ^{2P}‖w‖²`.

No probability is used: this is the pure compounding step.  The exponent `P` on `(1 ± ε)` is not
an artefact of the proof, the ratios genuinely multiply, which is why a per-pass bound that
looks tight can still be loose after several passes. -/
theorem decayWithin_of_forall {S : Scheme E} {w : E} {ρ ε : ℝ} {P : ℕ}
    (hε1 : ε ≤ 1)
    (h : ∀ p ∈ range P, |stageErrSq S w p - ρ ^ 2| ≤ ε * ρ ^ 2) :
    DecayWithin S w ρ ε P := by
  have hlow : ∀ p ∈ range P, (1 - ε) * ρ ^ 2 ≤ stageErrSq S w p := by
    intro p hp
    have h' := (abs_le.mp (h p hp)).1
    nlinarith [h']
  have hhigh : ∀ p ∈ range P, stageErrSq S w p ≤ (1 + ε) * ρ ^ 2 := by
    intro p hp
    have h' := (abs_le.mp (h p hp)).2
    nlinarith [h']
  have h0 : (0 : ℝ) ≤ (1 - ε) * ρ ^ 2 := mul_nonneg (by linarith) (sq_nonneg _)
  have hprod_low : ((1 - ε) * ρ ^ 2) ^ P ≤ ∏ p ∈ range P, stageErrSq S w p := by
    calc ((1 - ε) * ρ ^ 2) ^ P = ∏ _p ∈ range P, (1 - ε) * ρ ^ 2 := by
          rw [Finset.prod_const, Finset.card_range]
      _ ≤ ∏ p ∈ range P, stageErrSq S w p := Finset.prod_le_prod (fun _ _ => h0) hlow
  have hprod_high : ∏ p ∈ range P, stageErrSq S w p ≤ ((1 + ε) * ρ ^ 2) ^ P := by
    calc ∏ p ∈ range P, stageErrSq S w p ≤ ∏ _p ∈ range P, (1 + ε) * ρ ^ 2 :=
          Finset.prod_le_prod (fun p _ => stageErrSq_nonneg S w p) hhigh
      _ = ((1 + ε) * ρ ^ 2) ^ P := by rw [Finset.prod_const, Finset.card_range]
  have hexp : ∀ c : ℝ, (c * ρ ^ 2) ^ P = c ^ P * ρ ^ (2 * P) := by
    intro c; rw [mul_pow, pow_mul]
  constructor
  · rw [norm_resid_sq_eq_prod, ← hexp (1 - ε), mul_comm (‖w‖ ^ 2)]
    exact mul_le_mul_of_nonneg_right hprod_low (sq_nonneg _)
  · rw [norm_resid_sq_eq_prod, ← hexp (1 + ε), mul_comm (‖w‖ ^ 2)]
    exact mul_le_mul_of_nonneg_right hprod_high (sq_nonneg _)

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- **The union bound over passes.**  If each of the first `P` passes misses its relative-`ε`
band with probability at most `δ`, then the `P`-pass decay law holds to relative `(1 ± ε)^P`
with probability at least `1 − Pδ`.

This is the precise content of "essentially deterministically": *both* prices are explicit, the
accuracy `(1 ± ε)^P` and the confidence `Pδ`.  Neither appears in `err(P) = ρ^P`. -/
theorem meas_not_decayWithin_le {S : Ω → Scheme E} {W : Ω → E} {ρ ε : ℝ} {δ : ℝ≥0∞} {P : ℕ}
    (hε1 : ε ≤ 1)
    (h : ∀ p < P, μ {ω | ¬ |stageErrSq (S ω) (W ω) p - ρ ^ 2| ≤ ε * ρ ^ 2} ≤ δ) :
    μ {ω | ¬ DecayWithin (S ω) (W ω) ρ ε P} ≤ P * δ := by
  have hsub : {ω | ¬ DecayWithin (S ω) (W ω) ρ ε P}
      ⊆ ⋃ p ∈ range P, {ω | ¬ |stageErrSq (S ω) (W ω) p - ρ ^ 2| ≤ ε * ρ ^ 2} := by
    intro ω hω
    rcases Classical.em (∀ p ∈ range P, |stageErrSq (S ω) (W ω) p - ρ ^ 2| ≤ ε * ρ ^ 2) with
      hall | hall
    · exact absurd (decayWithin_of_forall hε1 hall) hω
    · push Not at hall
      obtain ⟨p, hp, hbad⟩ := hall
      exact Set.mem_iUnion₂_of_mem hp (not_le.mpr hbad)
  calc μ {ω | ¬ DecayWithin (S ω) (W ω) ρ ε P}
      ≤ μ (⋃ p ∈ range P, {ω | ¬ |stageErrSq (S ω) (W ω) p - ρ ^ 2| ≤ ε * ρ ^ 2}) :=
        measure_mono hsub
    _ ≤ ∑ p ∈ range P, μ {ω | ¬ |stageErrSq (S ω) (W ω) p - ρ ^ 2| ≤ ε * ρ ^ 2} :=
        measure_biUnion_finset_le _ _
    _ ≤ ∑ _p ∈ range P, δ := Finset.sum_le_sum fun p hp => h p (mem_range.mp hp)
    _ = P * δ := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]

/-- **The Chebyshev instantiation.**  A per-pass variance bound `Var ≤ v` supplies
`δ = v/(ερ²)²` in `meas_not_decayWithin_le`.

The paper's Remark asserts `v = O(1/d)`; that assertion is a computation about the spherical
marginal and is *not* proved here.  What is proved is that such a bound suffices, and exactly
what it yields: substituting `v = C/d` gives failure probability at most `P·C/(d ε² ρ⁴)`, so the
pass count enters linearly and the tolerance quadratically. -/
theorem meas_not_decayWithin_le_of_variance [IsFiniteMeasure μ]
    {S : Ω → Scheme E} {W : Ω → E} {ρ ε v : ℝ} {P : ℕ}
    (hε0 : 0 < ε) (hε1 : ε ≤ 1) (hρ : ρ ≠ 0)
    (hmem : ∀ p < P, MemLp (fun ω => stageErrSq (S ω) (W ω) p) 2 μ)
    (hmean : ∀ p < P, ∫ ω, stageErrSq (S ω) (W ω) p ∂μ = ρ ^ 2)
    (hvar : ∀ p < P, variance (fun ω => stageErrSq (S ω) (W ω) p) μ ≤ v) :
    μ {ω | ¬ DecayWithin (S ω) (W ω) ρ ε P} ≤ P * ENNReal.ofReal (v / (ε * ρ ^ 2) ^ 2) := by
  refine meas_not_decayWithin_le hε1 fun p hp => ?_
  have hρ2 : (0 : ℝ) < ρ ^ 2 := lt_of_le_of_ne (sq_nonneg ρ) (Ne.symm (pow_ne_zero 2 hρ))
  have hc : (0 : ℝ) < ε * ρ ^ 2 := mul_pos hε0 hρ2
  have hsub : {ω | ¬ |stageErrSq (S ω) (W ω) p - ρ ^ 2| ≤ ε * ρ ^ 2}
      ⊆ {ω | ε * ρ ^ 2 ≤
          |stageErrSq (S ω) (W ω) p - ∫ ω, stageErrSq (S ω) (W ω) p ∂μ|} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    rw [hmean p hp]
    exact le_of_lt (not_le.mp hω)
  calc μ {ω | ¬ |stageErrSq (S ω) (W ω) p - ρ ^ 2| ≤ ε * ρ ^ 2}
      ≤ μ {ω | ε * ρ ^ 2 ≤
          |stageErrSq (S ω) (W ω) p - ∫ ω, stageErrSq (S ω) (W ω) p ∂μ|} := measure_mono hsub
    _ ≤ ENNReal.ofReal (variance (fun ω => stageErrSq (S ω) (W ω) p) μ / (ε * ρ ^ 2) ^ 2) :=
        meas_ge_le_variance_div_sq (hmem p hp) hc
    _ ≤ ENNReal.ofReal (v / (ε * ρ ^ 2) ^ 2) := by
        gcongr
        exact hvar p hp

end Concentration
