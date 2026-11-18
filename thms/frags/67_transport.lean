
/-! ## §64  The transport route, and why the threshold route is the sharper of the two

§55 charges the scalar distortion gap to `sup_x |cdfGapInt μ ν a x|`, the once-integrated CDF
difference at the codebook's thresholds, and §56 multiplies it by `d`.  There is a second route,
and it looks cleaner.  If the stage error `ψ t = (t − q t)²` is Lipschitz with constant `L` on a
set carrying both laws, then `|E_μ ψ − E_ν ψ| ≤ L · W₁(μ, ν)`, and §48c's identity turns that into

  `|hadRate H (blockStage q) v − ρ²| ≤ d · L · W₁(coordLaw H v i, f_d)`.

`abs_hadRate_sub_rhoSq_le_wOne` proves exactly that.  `W₁` is built here from nothing, as the
infimum of the `L¹` costs of the integrable couplings (`wOne`), and the inequality is the easy half
of Kantorovich-Rubinstein duality: `abs_integral_sub_integral_le_wCost` bounds the gap by the cost
of *any* coupling and `le_mul_wOne_of_forall` passes to the infimum.  The stage error is **not**
globally Lipschitz, since `(t − q t)²` grows quadratically away from the codewords, so the
hypothesis has to be Lipschitz-on-a-set; `ae_mem_of_coupling` is what makes that usable, since
every coupling of two laws supported in `S` is supported in `S × S`.  §61 supplies the constant,
`L = 2M` from a sup bound `M` on `|t − q t|` over the common support, and
`abs_hadRate_sub_rhoSq_le_wOne_of_sup` is the composition; on `[−1, 1]` and for the shipped
codebook §61's `lipschitzOnWith_stageErr_Icc` evaluates to exactly the `L = 1.86650` the
arithmetic below uses.

**The comparison, and it is negative.**  The two routes are in different currencies and are not
comparable until one is converted into the other.  `abs_cdfGapInt_le_wOne` converts, and one way
only:

  `sup_x |cdfGapInt μ ν a x| ≤ W₁(μ, ν)`,

because `cdfGapInt μ ν a x` is the gap between the two laws' expectations of the clamp
`t ↦ (x − t)⁺ − (a − t)⁺` (§57's `cdfGapInt_eq_integral_clamp_sub`), that clamp is `1`-Lipschitz,
and `W₁` is an infimum over exactly such test functions.  The threshold currency is therefore never
the more expensive of the two, and the comparison collapses to the two prefactors,
`2 · spread cbk · d` against `L · d`.  `abs_hadRate_sub_rhoSq_le_spread_wOne` prints the threshold
bound in the transport currency so that the two right-hand sides can be set side by side, and
`threshold_le_transport_of_two_mul_spread_le` is the criterion: the threshold route is at least as
good exactly when `2 · spread cbk ≤ L`.

**Why `L` is large.**  It is governed by the distance from the support edge to the outermost
codeword, not by the spacing of the codewords.  `le_of_lipschitzOn_chord` shows that any admissible
`L` on `[−r, r]` is at least `r + t − 2c` for every `t` in the outermost cell, `c` the largest
codeword, hence at least `r − c` and, as `t ↑ r`, at least `2(r − c)`, which is the sup of
`|ψ′| = 2|t − q t|`.  The threshold route's `2 · spread cbk = 2(c_K − c_0)` sees only the codewords.
That is the whole story, and it is the same term an integration by parts finds: writing `G` for the
once-integrated CDF difference, a second integration by parts bounds the gap by
`sup|G| · TV(ψ′)`, and `TV(ψ′) = 4r + 2 · spread cbk` over `[−r, r]`, the `4r` being an interval
width that §50's summation by parts does not have, because it runs over the `K` thresholds and its
weights telescope to the spread.  At `d = 512` that crude route costs `2184.7` where transport
costs `955.65`, so transport is the better of the two losers.

**The arithmetic at `d = 512`, for the shipped two-bit codebook.**  Codewords `±0.4528/√d` and
`±1.5104/√d`, so `spread cbk = 3.0208/√512 = 0.13350`; both laws live on `[−1, 1]`, so
`sup_{|t| ≤ 1} |t − q t| = 1 − 1.5104/√512 = 0.93325`, attained at the support edge, the interior
half-gaps being `0.02337` and `0.02001`, forty times smaller.  The best Lipschitz constant is
`L = 1.86650` and the two prefactors are

  threshold `2 · spread cbk · d = 136.71`,  transport `L · d = 955.65`,

a ratio of `6.9905`, which is `(√d − 1.5104)/3.0208` and so grows like `√d`.
`threshold_le_transport_512` records `6.95` of that `6.9905` in checkable form, `6.95` being what
the chord at `t = c + (99/100)(1 − c)` delivers; `le_lipschitz_512` is the step that costs
something.  The two routes tie at `√d = 4.5312`, that is `d ≈ 21`, and below that the transport
route wins; that is the regime in which the codebook spans its own support, which is not the regime
of this paper.

**In absolute terms.**  At `d = 512` and a flat direction, §60's completed threshold bound is
`0.650`, against the free bound `1` that holds whenever `0` is a codeword, and against
`ρ² = 0.1171`, which it does not reach.  If `W₁` were as small as the **proved** bound on
`sup_x |cdfGapInt|` there, `C = 0.0047521`, which is the smallest value `abs_cdfGapInt_le_wOne`
permits it to be, the transport bound would be `955.65 × 0.0047521 = 4.541`: worse than the free
bound, and `38.8` times `ρ²`.  There is no dimension and no codebook in this development at which
the transport route is the one to use.

**What is not established.**  That the stage error is Lipschitz at all is proved in §61 and not
here; every statement below carries `hLip` as a hypothesis, and only
`abs_hadRate_sub_rhoSq_le_wOne_of_sup` discharges it.  What is missing is the other half: no `W₁`
bound is proved for the pair that matters, so the transport route is conditional on a quantity
this development does not bound.  `wOne_sphericalCoordLaw_gaussian_le` repackages
§57's coupling leg as the transport bound it already was,
`W₁(f_d, N(0, 1/d)) ≤ √(1 + 1/(4(d−1)))/d`, which is `0.0019536` at `d = 512`; but §58's Fourier
leg produces a `cdfGapInt` bound and not a `W₁` bound, and the implication proved here runs the
other way, so the composition §59 performs has no transport analogue and
`abs_hadRate_sub_rhoSq_le_wOne` is discharged at no direction at all.  Nothing here weakens or
replaces §55, §56, §59, §60 or §63: every theorem of those sections stands, and the conclusion of
this one is that they are the ones to use.  §63 answers the same question in the threshold
currency, where the `K` numbers are both necessary and sufficient; this section answers it in the
transport currency and finds the answer more expensive.  Neither `ρ²` nor any codebook the paper
ships is analysed here; `3.0208/√512` and `1.5104/√512` enter as hypotheses on an abstract
`Codebook`, and no claim is made that any deployed rotation attains `ρ²`. -/

noncomputable section TransportRoute

open Hadamard Orthogonal CoordLaw

namespace Transport

open Threshold

/-! ### Couplings and the Wasserstein-1 distance -/

/-- A **coupling** of two laws on `ℝ`: a law on `ℝ × ℝ` with the two as its marginals. -/
def IsCoupling (γ : Measure (ℝ × ℝ)) (μ ν : Measure ℝ) : Prop :=
  Measure.map Prod.fst γ = μ ∧ Measure.map Prod.snd γ = ν

/-- The `L¹` transport cost of a coupling. -/
def wCost (γ : Measure (ℝ × ℝ)) : ℝ := ∫ p, |p.1 - p.2| ∂γ

/-- The set of `L¹` costs of integrable couplings of `μ` and `ν`. -/
def couplingCosts (μ ν : Measure ℝ) : Set ℝ :=
  {r | ∃ γ : Measure (ℝ × ℝ), IsProbabilityMeasure γ ∧ IsCoupling γ μ ν ∧
    Integrable (fun p : ℝ × ℝ => |p.1 - p.2|) γ ∧ wCost γ = r}

/-- **The Wasserstein-1 distance**, as the infimum of the `L¹` costs of the integrable couplings.
When no integrable coupling exists the set is empty and `sInf` returns `0`, so every statement
below that reads `wOne` as a distance carries a nonemptiness hypothesis explicitly. -/
def wOne (μ ν : Measure ℝ) : ℝ := sInf (couplingCosts μ ν)

theorem wCost_nonneg {γ : Measure (ℝ × ℝ)} : 0 ≤ wCost γ :=
  integral_nonneg fun _ => abs_nonneg _

theorem bddBelow_couplingCosts (μ ν : Measure ℝ) : BddBelow (couplingCosts μ ν) := by
  refine ⟨0, ?_⟩
  rintro r ⟨γ, _, _, _, rfl⟩
  exact wCost_nonneg

theorem wOne_nonneg (μ ν : Measure ℝ) : 0 ≤ wOne μ ν := by
  rcases Set.eq_empty_or_nonempty (couplingCosts μ ν) with h | h
  · rw [wOne, h, Real.sInf_empty]
  · exact le_csInf h fun r hr => by obtain ⟨γ, _, _, _, rfl⟩ := hr; exact wCost_nonneg

theorem wOne_le_wCost {μ ν : Measure ℝ} {γ : Measure (ℝ × ℝ)} [IsProbabilityMeasure γ]
    (hγ : IsCoupling γ μ ν) (hint : Integrable (fun p : ℝ × ℝ => |p.1 - p.2|) γ) :
    wOne μ ν ≤ wCost γ :=
  csInf_le (bddBelow_couplingCosts μ ν) ⟨γ, inferInstance, hγ, hint, rfl⟩

/-- A marginal of a coupling transports integrability of the identity. -/
theorem integrable_fst_of_coupling {μ ν : Measure ℝ} {γ : Measure (ℝ × ℝ)}
    (hγ : IsCoupling γ μ ν) (hμ : Integrable (fun t : ℝ => t) μ) :
    Integrable (fun p : ℝ × ℝ => p.1) γ := by
  have hasm : AEStronglyMeasurable (fun t : ℝ => t) (Measure.map Prod.fst γ) := by
    rw [hγ.1]; exact hμ.aestronglyMeasurable
  exact (integrable_map_measure hasm measurable_fst.aemeasurable).mp (by rw [hγ.1]; exact hμ)

/-- The second marginal, likewise. -/
theorem integrable_snd_of_coupling {μ ν : Measure ℝ} {γ : Measure (ℝ × ℝ)}
    (hγ : IsCoupling γ μ ν) (hν : Integrable (fun t : ℝ => t) ν) :
    Integrable (fun p : ℝ × ℝ => p.2) γ := by
  have hasm : AEStronglyMeasurable (fun t : ℝ => t) (Measure.map Prod.snd γ) := by
    rw [hγ.2]; exact hν.aestronglyMeasurable
  exact (integrable_map_measure hasm measurable_snd.aemeasurable).mp (by rw [hγ.2]; exact hν)

/-- The product law is a coupling: the independent one. -/
theorem isCoupling_prod (μ ν : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] :
    IsCoupling (μ.prod ν) μ ν := by
  constructor
  · rw [Measure.map_fst_prod, measure_univ, one_smul]
  · rw [Measure.map_snd_prod, measure_univ, one_smul]

/-- **Finite first moments make `wOne` an infimum over a nonempty set**, the independent coupling
witnessing it.  Without this the `sInf` is the junk value `0` and `wOne` is not a distance. -/
theorem couplingCosts_nonempty {μ ν : Measure ℝ} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hμ : Integrable (fun t : ℝ => t) μ) (hν : Integrable (fun t : ℝ => t) ν) :
    (couplingCosts μ ν).Nonempty := by
  refine ⟨wCost (μ.prod ν), μ.prod ν, inferInstance, isCoupling_prod μ ν, ?_, rfl⟩
  exact ((integrable_fst_of_coupling (isCoupling_prod μ ν) hμ).sub
    (integrable_snd_of_coupling (isCoupling_prod μ ν) hν)).abs

/-- **Every coupling of laws supported in `S` is supported in `S × S`.**  This is what lets a
Lipschitz hypothesis that holds only on the common support be used under the coupling, and it is
the reason the theorems below need no global Lipschitz hypothesis, which the stage error does not
satisfy: `(t − q t)²` grows quadratically far from the codewords. -/
theorem ae_mem_of_coupling {S : Set ℝ} (hS : MeasurableSet S) {μ ν : Measure ℝ}
    {γ : Measure (ℝ × ℝ)} (hγ : IsCoupling γ μ ν) (hμS : μ Sᶜ = 0) (hνS : ν Sᶜ = 0) :
    ∀ᵐ p ∂γ, p.1 ∈ S ∧ p.2 ∈ S := by
  have h1 : γ (Prod.fst ⁻¹' Sᶜ) = 0 := by
    rw [← Measure.map_apply measurable_fst hS.compl, hγ.1, hμS]
  have h2 : γ (Prod.snd ⁻¹' Sᶜ) = 0 := by
    rw [← Measure.map_apply measurable_snd hS.compl, hγ.2, hνS]
  have hu := measure_union_null h1 h2
  rw [ae_iff]
  refine measure_mono_null (fun p hp => ?_) hu
  simp only [Set.mem_setOf_eq, not_and_or] at hp
  rcases hp with hp | hp
  · exact Or.inl hp
  · exact Or.inr hp

/-! ### The transport inequality -/

/-- **The transport bound, against one coupling.**  If `ψ` is `L`-Lipschitz on a measurable set
`S` carrying both laws, then the two expectations of `ψ` differ by at most `L` times the `L¹` cost
of any coupling.  This is the easy half of Kantorovich-Rubinstein duality and nothing more: it
produces no coupling, and it says nothing about how small the cost of the best one is. -/
theorem abs_integral_sub_integral_le_wCost {ψ : ℝ → ℝ} {L : ℝ} {S : Set ℝ}
    (hS : MeasurableSet S) (hLip : ∀ s ∈ S, ∀ t ∈ S, |ψ s - ψ t| ≤ L * |s - t|)
    {μ ν : Measure ℝ} (hμS : μ Sᶜ = 0) (hνS : ν Sᶜ = 0)
    {γ : Measure (ℝ × ℝ)} (hγ : IsCoupling γ μ ν)
    (hψμ : Integrable ψ μ) (hψν : Integrable ψ ν)
    (hcost : Integrable (fun p : ℝ × ℝ => |p.1 - p.2|) γ) :
    |(∫ t, ψ t ∂μ) - ∫ t, ψ t ∂ν| ≤ L * wCost γ := by
  have hasm1 : AEStronglyMeasurable ψ (Measure.map Prod.fst γ) := by
    rw [hγ.1]; exact hψμ.aestronglyMeasurable
  have hasm2 : AEStronglyMeasurable ψ (Measure.map Prod.snd γ) := by
    rw [hγ.2]; exact hψν.aestronglyMeasurable
  have hm1 : (∫ t, ψ t ∂μ) = ∫ p, ψ p.1 ∂γ := by
    rw [← hγ.1]; exact integral_map measurable_fst.aemeasurable hasm1
  have hm2 : (∫ t, ψ t ∂ν) = ∫ p, ψ p.2 ∂γ := by
    rw [← hγ.2]; exact integral_map measurable_snd.aemeasurable hasm2
  have hi1 : Integrable (fun p : ℝ × ℝ => ψ p.1) γ :=
    (integrable_map_measure hasm1 measurable_fst.aemeasurable).mp (by rw [hγ.1]; exact hψμ)
  have hi2 : Integrable (fun p : ℝ × ℝ => ψ p.2) γ :=
    (integrable_map_measure hasm2 measurable_snd.aemeasurable).mp (by rw [hγ.2]; exact hψν)
  rw [hm1, hm2, ← integral_sub hi1 hi2]
  calc |∫ p, (ψ p.1 - ψ p.2) ∂γ| ≤ ∫ p, |ψ p.1 - ψ p.2| ∂γ := abs_integral_le_integral_abs
    _ ≤ ∫ p, L * |p.1 - p.2| ∂γ := by
        refine integral_mono_ae (hi1.sub hi2).abs (hcost.const_mul L) ?_
        filter_upwards [ae_mem_of_coupling hS hγ hμS hνS] with p hp
        exact hLip p.1 hp.1 p.2 hp.2
    _ = L * wCost γ := by rw [wCost, integral_const_mul]

/-- Passing from every coupling to the infimum, which needs `0 ≤ L` and a nonempty coupling
set. -/
theorem le_mul_wOne_of_forall {μ ν : Measure ℝ} {L y : ℝ} (hL0 : 0 ≤ L)
    (hne : (couplingCosts μ ν).Nonempty)
    (h : ∀ γ : Measure (ℝ × ℝ), IsProbabilityMeasure γ → IsCoupling γ μ ν →
      Integrable (fun p : ℝ × ℝ => |p.1 - p.2|) γ → y ≤ L * wCost γ) :
    y ≤ L * wOne μ ν := by
  rcases eq_or_lt_of_le hL0 with hL | hL
  · obtain ⟨r, γ, hp, hc, hi, hr⟩ := hne
    have hy := h γ hp hc hi
    rw [← hL] at hy ⊢
    simpa using hy
  · rw [← div_le_iff₀' hL]
    refine le_csInf hne ?_
    rintro r ⟨γ, hp, hc, hi, rfl⟩
    rw [div_le_iff₀' hL]
    exact h γ hp hc hi

/-- **The transport bound, against `W₁`.**  `|E_μ ψ − E_ν ψ| ≤ L · W₁(μ, ν)` for a `ψ` that is
`L`-Lipschitz on a common support.  This is the statement the transport route rests on; the
Lipschitz hypothesis is taken, not proved, here. -/
theorem abs_integral_sub_integral_le_wOne {ψ : ℝ → ℝ} {L : ℝ} {S : Set ℝ}
    (hS : MeasurableSet S) (hL0 : 0 ≤ L)
    (hLip : ∀ s ∈ S, ∀ t ∈ S, |ψ s - ψ t| ≤ L * |s - t|)
    {μ ν : Measure ℝ} (hμS : μ Sᶜ = 0) (hνS : ν Sᶜ = 0)
    (hne : (couplingCosts μ ν).Nonempty)
    (hψμ : Integrable ψ μ) (hψν : Integrable ψ ν) :
    |(∫ t, ψ t ∂μ) - ∫ t, ψ t ∂ν| ≤ L * wOne μ ν :=
  le_mul_wOne_of_forall hL0 hne fun _γ _ hc hi =>
    abs_integral_sub_integral_le_wCost hS hLip hμS hνS hc hψμ hψν hi

/-- A function Lipschitz on a bounded interval carrying all the mass is integrable there.  This
discharges the integrability hypotheses of the transport bound from the Lipschitz hypothesis
itself, so no separate moment assumption is needed below. -/
theorem integrable_of_lipschitzOn_Icc {ψ : ℝ → ℝ} (hψ : Measurable ψ) {L r : ℝ} (hL0 : 0 ≤ L)
    (hr : 0 ≤ r) (hLip : ∀ s ∈ Set.Icc (-r) r, ∀ t ∈ Set.Icc (-r) r, |ψ s - ψ t| ≤ L * |s - t|)
    {μ : Measure ℝ} [IsFiniteMeasure μ] (hsupp : μ (Set.Icc (-r) r)ᶜ = 0) : Integrable ψ μ := by
  refine (integrable_const (|ψ 0| + L * r)).mono' hψ.aestronglyMeasurable ?_
  have hae : ∀ᵐ t ∂μ, t ∈ Set.Icc (-r) r := by
    rw [ae_iff]
    exact measure_mono_null (fun t ht => ht) hsupp
  filter_upwards [hae] with t ht
  have h0 : (0 : ℝ) ∈ Set.Icc (-r) r := ⟨by linarith, hr⟩
  have hd := hLip t ht 0 h0
  rw [sub_zero] at hd
  have habs : |t| ≤ r := abs_le.mpr ⟨ht.1, ht.2⟩
  have : |ψ t| ≤ |ψ 0| + L * r := by
    have h1 : |ψ t| - |ψ 0| ≤ |ψ t - ψ 0| := abs_sub_abs_le_abs_sub _ _
    nlinarith [hd, h1, abs_nonneg (ψ t), abs_nonneg (ψ 0)]
  rwa [Real.norm_eq_abs]

/-! ### The threshold currency is dominated by the transport currency -/

/-- **The bridge.**  `|cdfGapInt μ ν a x|` is at most the `L¹` cost of any coupling, for every
`a` and `x`.  This is §57's `abs_cdfGapInt_le_integral_abs_sub_of_coupling` read on the coupling
measure itself, plus the observation that `cdfGapInt` vanishes when `x < a`. -/
theorem abs_cdfGapInt_le_wCost {μ ν : Measure ℝ} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    {γ : Measure (ℝ × ℝ)} [IsProbabilityMeasure γ] (hγ : IsCoupling γ μ ν)
    (hμ : Integrable (fun t : ℝ => t) μ) (hν : Integrable (fun t : ℝ => t) ν) (a x : ℝ) :
    |cdfGapInt μ ν a x| ≤ wCost γ := by
  rcases le_or_gt a x with hax | hax
  · rw [wCost]
    exact abs_cdfGapInt_le_integral_abs_sub_of_coupling γ measurable_fst.aemeasurable
      measurable_snd.aemeasurable (integrable_fst_of_coupling hγ hμ)
      (integrable_snd_of_coupling hγ hν) hγ.1 hγ.2 hax
  · have he : Set.Ioc a x = (∅ : Set ℝ) := Set.Ioc_eq_empty (by linarith)
    rw [cdfGapInt, he, Measure.restrict_empty, integral_zero_measure, abs_zero]
    exact wCost_nonneg

/-- **The two currencies compared.**  The quantity the threshold identity charges for,
`sup_x |cdfGapInt μ ν a x|`, is at most `W₁(μ, ν)`, the quantity the transport route charges for.
So a bound in the threshold currency is never harder to obtain than one in the transport
currency, and the comparison of the two routes reduces to the comparison of their prefactors. -/
theorem abs_cdfGapInt_le_wOne {μ ν : Measure ℝ} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hμ : Integrable (fun t : ℝ => t) μ) (hν : Integrable (fun t : ℝ => t) ν)
    (hne : (couplingCosts μ ν).Nonempty) (a x : ℝ) : |cdfGapInt μ ν a x| ≤ wOne μ ν := by
  have h := le_mul_wOne_of_forall (L := 1) (y := |cdfGapInt μ ν a x|) zero_le_one hne
    (fun γ hp hc _ => by rw [one_mul]; exact @abs_cdfGapInt_le_wCost μ ν _ _ γ hp hc hμ hν a x)
  rwa [one_mul] at h

/-! ### One leg is available in the transport currency, and the other is not -/

/-- **`W₁(f_d, N(0, 1/d)) ≤ √(1 + 1/(4(d−1)))/d`.**  §57's coupling leg is stated as an `L¹` cost,
so it is already a transport bound, not merely a `cdfGapInt` bound: the map
`z ↦ (z/‖z‖, z/√d)` pushes the Gaussian on `Ed d` to a coupling of the two laws whose cost is the
quantity §57 estimates.  Nothing new is estimated here; only the packaging changes.

This is the **only** leg the transport route can use.  §58's Fourier leg bounds
`|cdfGapInt (coordLaw H v i) (N(0, 1/d)) a x|` by Esseen smoothing, and that is a bound on the
once-integrated CDF difference, which does **not** imply a bound on `W₁ = ∫ |cdf μ − cdf ν|`: the
implication proved here, `abs_cdfGapInt_le_wOne`, runs the other way.  So the composition §59
performs has no transport analogue in this development, and
`abs_hadRate_sub_rhoSq_le_wOne` cannot presently be discharged at any direction. -/
theorem wOne_sphericalCoordLaw_gaussian_le {d : ℕ} (hd : 2 ≤ d) :
    wOne (sphericalCoordLaw d) (gaussianReal 0 (Real.toNNReal (1 / (d : ℝ))))
      ≤ Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) := by
  haveI : Nontrivial (Ed d) := Module.nontrivial_of_finrank_pos (R := ℝ) (by
    have : (0 : ℕ) < d := by omega
    simpa [Ed] using this)
  have hi : (0 : ℕ) < d := by omega
  set i : Fin d := ⟨0, hi⟩ with hidef
  have hmX : Measurable (fun z : Ed d => (dir z).ofLp i) := measurable_dir_coord i
  have hcoord : Measurable (fun z : Ed d => z.ofLp i) :=
    (measurable_pi_apply i).comp (WithLp.measurable_ofLp 2 _)
  have hmY : Measurable (fun z : Ed d => z.ofLp i / Real.sqrt (d : ℝ)) := hcoord.div_const _
  have hXlaw : Measure.map (fun z : Ed d => (dir z).ofLp i) (gaussianE (Ed d))
      = sphericalCoordLaw d := gaussianE_map_dir_coord hd i
  have hYlaw : Measure.map (fun z : Ed d => z.ofLp i / Real.sqrt (d : ℝ)) (gaussianE (Ed d))
      = gaussianReal 0 (Real.toNNReal (1 / (d : ℝ))) := by
    have h1 : (fun z : Ed d => z.ofLp i / Real.sqrt (d : ℝ))
        = (fun t : ℝ => t / Real.sqrt (d : ℝ)) ∘ (fun z : Ed d => z.ofLp i) := rfl
    have hdiv : Measurable (fun t : ℝ => t / Real.sqrt (d : ℝ)) := measurable_id.div_const _
    rw [h1, ← Measure.map_map hdiv hcoord, gaussianE_map_ofLp i, gaussianReal_map_div_sqrt hd]
  have hIX : Integrable (fun z : Ed d => (dir z).ofLp i) (gaussianE (Ed d)) := by
    refine (integrable_const (1 : ℝ)).mono' hmX.aestronglyMeasurable ?_
    filter_upwards with z
    rw [Real.norm_eq_abs]
    exact abs_dir_coord_le_one i z
  have hIY : Integrable (fun z : Ed d => z.ofLp i / Real.sqrt (d : ℝ)) (gaussianE (Ed d)) := by
    refine ((integrable_norm_gaussianE (E := Ed d)).div_const
      (Real.sqrt (d : ℝ))).mono' hmY.aestronglyMeasurable ?_
    filter_upwards with z
    rw [Real.norm_eq_abs, abs_div, abs_of_nonneg (Real.sqrt_nonneg _)]
    gcongr
    simpa using PiLp.norm_apply_le z i
  set F : Ed d → ℝ × ℝ :=
    fun z => ((dir z).ofLp i, z.ofLp i / Real.sqrt (d : ℝ)) with hFdef
  have hmF : Measurable F := hmX.prodMk hmY
  set γ : Measure (ℝ × ℝ) := Measure.map F (gaussianE (Ed d)) with hγdef
  haveI : IsProbabilityMeasure γ := by
    rw [hγdef]; exact Measure.isProbabilityMeasure_map hmF.aemeasurable
  have hcpl : IsCoupling γ (sphericalCoordLaw d)
      (gaussianReal 0 (Real.toNNReal (1 / (d : ℝ)))) := by
    constructor
    · rw [hγdef, Measure.map_map measurable_fst hmF]; exact hXlaw
    · rw [hγdef, Measure.map_map measurable_snd hmF]; exact hYlaw
  have hasm : AEStronglyMeasurable (fun p : ℝ × ℝ => |p.1 - p.2|) (Measure.map F (gaussianE (Ed d)))
      := (continuous_abs.comp (continuous_fst.sub continuous_snd)).aestronglyMeasurable
  have hcost : Integrable (fun p : ℝ × ℝ => |p.1 - p.2|) γ := by
    rw [hγdef]
    exact (integrable_map_measure hasm hmF.aemeasurable).mpr (hIX.sub hIY).abs
  have hwc : wCost γ
      = ∫ z, |(dir z).ofLp i - z.ofLp i / Real.sqrt (d : ℝ)| ∂(gaussianE (Ed d)) := by
    rw [wCost, hγdef]
    exact integral_map hmF.aemeasurable hasm
  calc wOne (sphericalCoordLaw d) (gaussianReal 0 (Real.toNNReal (1 / (d : ℝ))))
      ≤ wCost γ := wOne_le_wCost hcpl hcost
    _ = ∫ z, |(dir z).ofLp i - z.ofLp i / Real.sqrt (d : ℝ)| ∂(gaussianE (Ed d)) := hwc
    _ ≤ Real.sqrt (1 + 1 / (4 * ((d : ℝ) - 1))) / (d : ℝ) :=
        integral_abs_dir_coord_sub_coord_div_sqrt_le hd i

/-! ### The stage error, and the deployed per-pass rate -/

/-- **The transport bound for the stage error.**  If the stage error `t ↦ (t − q t)²` is
`L`-Lipschitz on a set carrying both laws, the distortion gap is at most `L · W₁`.  Continuity of
the stage error at the thresholds, which is what makes such an `L` exist for a nearest-codeword
map even though `q` itself jumps, is **not** proved here: `hLip` is a hypothesis. -/
theorem abs_distortion_sub_distortion_le_wOne {q : ℝ → ℝ} {L : ℝ} {S : Set ℝ}
    (hS : MeasurableSet S) (hL0 : 0 ≤ L)
    (hLip : ∀ s ∈ S, ∀ t ∈ S, |(s - q s) ^ 2 - (t - q t) ^ 2| ≤ L * |s - t|)
    {μ ν : Measure ℝ} (hμS : μ Sᶜ = 0) (hνS : ν Sᶜ = 0)
    (hne : (couplingCosts μ ν).Nonempty)
    (hIμ : Integrable (fun t : ℝ => (t - q t) ^ 2) μ)
    (hIν : Integrable (fun t : ℝ => (t - q t) ^ 2) ν) :
    |distortion q μ - distortion q ν| ≤ L * wOne μ ν :=
  abs_integral_sub_integral_le_wOne hS hL0 hLip hμS hνS hne hIμ hIν

/-- **The deployed per-pass rate, by transport.**  For `d ≥ 2`, a flat `H`, a unit direction `v`
and a measurable `q` whose stage error is `L`-Lipschitz on `[−1, 1]`,

  `|hadRate H (blockStage q) v − ρ²| ≤ d · L · W₁(coordLaw H v i, f_d)`.

This is the transport route in full.  It is a different currency from §55's threshold route, not
a sharpening of it: §56 charges `2 · spread cbk · d` per unit of `sup_x |cdfGapInt|`, this charges
`L · d` per unit of `W₁`, and `abs_cdfGapInt_le_wOne` shows the first currency is the cheaper of
the two.  The comparison of the two prefactors is below and it goes against this theorem.

Nothing here supplies `L`, and nothing here bounds `W₁` for the pair in question.  In particular
this development proves no `W₁` bound between `coordLaw H v i` and `f_d`: §58's Fourier leg bounds
the once-integrated CDF difference, which does not imply a `W₁` bound, so the transport route
cannot presently be closed even at a flat direction. -/
theorem abs_hadRate_sub_rhoSq_le_wOne {d : ℕ} (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) {q : ℝ → ℝ} (hqm : Measurable q) {L : ℝ} (hL0 : 0 ≤ L)
    (hLip : ∀ s ∈ Set.Icc (-1 : ℝ) 1, ∀ t ∈ Set.Icc (-1 : ℝ) 1,
      |(s - q s) ^ 2 - (t - q t) ^ 2| ≤ L * |s - t|) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ (d : ℝ) * L * wOne (coordLaw H v i) (sphericalCoordLaw d) := by
  haveI := isProbabilityMeasure_sphericalCoordLaw hd
  have hψ : Measurable fun t : ℝ => (t - q t) ^ 2 := ScalarGap.measurable_sub_cb_sq hqm
  have hμS : (coordLaw H v i) (Set.Icc (-1 : ℝ) 1)ᶜ = 0 := by
    have h := coordLaw_compl_Icc_eq_zero H v i
    rwa [hv] at h
  have hνS := sphericalCoordLaw_compl_Icc_eq_zero d
  have hIμ := integrable_of_lipschitzOn_Icc hψ hL0 zero_le_one hLip hμS
  have hIν := integrable_of_lipschitzOn_Icc hψ hL0 zero_le_one hLip hνS
  have hne := couplingCosts_nonempty (integrable_id_coordLaw H v i)
    (integrable_id_sphericalCoordLaw hd)
  have hgap := abs_distortion_sub_distortion_le_wOne measurableSet_Icc hL0 hLip hμS hνS hne hIμ hIν
  have hid : distortion q (radLaw d v) = distortion q (coordLaw H v i) := by
    rw [coordLaw_eq_radLaw hH v i]
  have hd0 : (0 : ℝ) ≤ (d : ℝ) := Nat.cast_nonneg d
  rw [ScalarGap.hadRate_sub_rhoSq hH hqm v, hid, abs_mul, abs_of_nonneg hd0]
  calc (d : ℝ) * |distortion q (coordLaw H v i) - distortion q (sphericalCoordLaw d)|
      ≤ (d : ℝ) * (L * wOne (coordLaw H v i) (sphericalCoordLaw d)) :=
        mul_le_mul_of_nonneg_left hgap hd0
    _ = (d : ℝ) * L * wOne (coordLaw H v i) (sphericalCoordLaw d) := by ring

/-- **The transport bound with §61's Lipschitz constant put in.**  For a nearest-codeword map
whose error is at most `M` on the common support `[−1, 1]`, §61's `abs_stageErr_sub_le_of_sup`
gives `L = 2M`, and the transport bound reads

  `|hadRate H (blockStage q) v − ρ²| ≤ d · 2M · W₁(coordLaw H v i, f_d)`.

This is the transport route with nothing left assumed about the stage error.  It is still not a
bound on anything, because `W₁` between those two laws is not bounded anywhere in this
development; and for a codebook at the coordinate scale `M` is `1 − c_K`, the distance from the
support edge to the outermost codeword, which is what the comparison below charges for. -/
theorem abs_hadRate_sub_rhoSq_le_wOne_of_sup {d : ℕ} (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) {C : Set ℝ} {q : ℝ → ℝ} (hq : IsNearest C q)
    (hqm : Measurable q) {M : ℝ} (hM0 : 0 ≤ M)
    (hM : ∀ u ∈ Set.Icc (-1 : ℝ) 1, |u - q u| ≤ M) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ (d : ℝ) * (2 * M) * wOne (coordLaw H v i) (sphericalCoordLaw d) :=
  abs_hadRate_sub_rhoSq_le_wOne hd hH v hv i hqm (by linarith)
    (fun s hs t ht => abs_stageErr_sub_le_of_sup hq hM hs ht)

/-! ### The comparison: the threshold route, in the same currency -/

/-- **The threshold route, priced in `W₁`.**  §56's `abs_hadRate_sub_rhoSq_le_of_cdfGapInt` with
its hypothesis `hC` discharged by the bridge.  Both routes now read
`prefactor · W₁(coordLaw H v i, f_d)`, with prefactor `2 · spread cbk · d` here and `L · d`
there, so they may be compared as numbers. -/
theorem abs_hadRate_sub_rhoSq_le_spread_wOne {K d : ℕ} (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ 2 * wOne (coordLaw H v i) (sphericalCoordLaw d) * spread cbk * (d : ℝ) := by
  haveI := isProbabilityMeasure_sphericalCoordLaw hd
  have hne := couplingCosts_nonempty (integrable_id_coordLaw H v i)
    (integrable_id_sphericalCoordLaw hd)
  exact abs_hadRate_sub_rhoSq_le_of_cdfGapInt hd hH v hv i cbk hq hqm ha hb hthr
    (fun x => abs_cdfGapInt_le_wOne (integrable_id_coordLaw H v i)
      (integrable_id_sphericalCoordLaw hd) hne a x)

/-- Comparing the two prefactors at equal currency: if the transport route's Lipschitz constant
exceeds `F` times twice the spread, its bound exceeds `F` times the threshold route's. -/
theorem mul_threshold_le_transport {d L W spr F : ℝ} (hd : 0 ≤ d) (hW : 0 ≤ W)
    (h : F * (2 * spr) ≤ L) : F * (2 * W * spr * d) ≤ d * L * W :=
  calc F * (2 * W * spr * d) = F * (2 * spr) * (W * d) := by ring
    _ ≤ L * (W * d) := mul_le_mul_of_nonneg_right h (mul_nonneg hW hd)
    _ = d * L * W := by ring

/-! ### How large the transport route's Lipschitz constant has to be -/

/-- A nearest-codeword map sends everything at or above the largest codeword to it. -/
theorem eq_top_of_isNearest {C : Set ℝ} {q : ℝ → ℝ} (hq : IsNearest C q) {c : ℝ} (hc : c ∈ C)
    (hub : ∀ u ∈ C, u ≤ c) {t : ℝ} (ht : c ≤ t) : q t = c := by
  obtain ⟨hmem, hmin⟩ := hq t
  have h1 : ‖t - q t‖ ≤ ‖t - c‖ := hmin c hc
  have h2 : q t ≤ c := hub _ hmem
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (by linarith : (0 : ℝ) ≤ t - q t),
    abs_of_nonneg (by linarith : (0 : ℝ) ≤ t - c)] at h1
  linarith

/-- The same for an ordered codebook, whose largest codeword is the last one. -/
theorem isNearest_top {K : ℕ} (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) {t : ℝ} (ht : cbk.c (Fin.last K) ≤ t) :
    q t = cbk.c (Fin.last K) :=
  eq_top_of_isNearest hq ⟨Fin.last K, rfl⟩
    (by rintro u ⟨k, rfl⟩; exact cbk.mono.monotone (Fin.le_last k)) ht

/-- **Every Lipschitz constant for the stage error on `[−r, r]` is at least `r + t − 2c`**, for
any `t` in the outermost cell, `c` being the largest codeword.  Letting `t ↑ r` this approaches
`2(r − c)`, the sup of `|ψ′| = 2|t − q t|` on the interval, so the transport route's constant is
governed by the distance from the support edge to the outermost codeword and not by the spacing
of the codewords.  This is a **lower** bound on any admissible `L`: it is what makes the
comparison below a statement against the transport route rather than a wish. -/
theorem le_of_lipschitzOn_chord {q : ℝ → ℝ} {L r c t : ℝ} (hrc : -r ≤ c) (hct : c ≤ t)
    (htr : t < r) (hqr : q r = c) (hqt : q t = c)
    (hLip : ∀ s ∈ Set.Icc (-r) r, ∀ u ∈ Set.Icc (-r) r,
      |(s - q s) ^ 2 - (u - q u) ^ 2| ≤ L * |s - u|) : r + t - 2 * c ≤ L := by
  have hrt : 0 < r - t := by linarith
  have hmr : -r ≤ t := le_trans hrc hct
  have h := hLip r ⟨by linarith, le_refl r⟩ t ⟨hmr, htr.le⟩
  rw [hqr, hqt, abs_of_pos (by linarith : (0 : ℝ) < r - t)] at h
  have h1 : (r - t) * (r + t - 2 * c) ≤ L * (r - t) :=
    calc (r - t) * (r + t - 2 * c) = (r - c) ^ 2 - (t - c) ^ 2 := by ring
      _ ≤ |(r - c) ^ 2 - (t - c) ^ 2| := le_abs_self _
      _ ≤ L * (r - t) := h
  nlinarith [h1, hrt]

/-- **The criterion, at equal currency.**  The threshold route's printed bound is at most the
transport route's exactly when twice the spread is at most the Lipschitz constant.  The `F = 1`
case of `mul_threshold_le_transport`, stated separately because it is the statement of what the
comparison turns on: `2 · spread cbk` against `L`. -/
theorem threshold_le_transport_of_two_mul_spread_le {K : ℕ} (cbk : Codebook K) {d L W : ℝ}
    (hd : 0 ≤ d) (hW : 0 ≤ W) (h : 2 * spread cbk ≤ L) :
    2 * W * spread cbk * d ≤ d * L * W := by
  have := mul_threshold_le_transport (F := 1) (spr := spread cbk) (L := L) (W := W) (d := d)
    hd hW (by linarith)
  linarith

/-- **When the codebook geometry decides the comparison.**  For a nearest-codeword map of an
ordered codebook whose largest codeword `c` lies inside the support `[−r, r]`, every Lipschitz
constant of the stage error on `[−r, r]` is at least `r − c`.  So if the spread is at most half
the distance from the top codeword to the support edge, the threshold route dominates the
transport route at equal currency, by `threshold_le_transport_of_two_mul_spread_le`.

For a codebook placed at the coordinate scale, `c = 1.5104/√d` and spread `3.0208/√d` against
`r = 1`, the condition `2 · spread ≤ r − c` reads `√d ≥ 7.552`, that is `d ≥ 58`; the sharper
chord of `le_of_lipschitzOn_chord` at `t ↑ r` gives `L ≥ 2(r − c)` in the limit and lowers this to
`√d ≥ 4.5312`, that is `d ≥ 21`.  Neither is a hypothesis anyone would want to drop: the regime
in which the transport route would win is the one where the codebook spans its own support. -/
theorem two_mul_spread_le_lipschitz {K : ℕ} (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) {L r : ℝ} (hrc : -r ≤ cbk.c (Fin.last K))
    (hcr : cbk.c (Fin.last K) < r)
    (hLip : ∀ s ∈ Set.Icc (-r) r, ∀ u ∈ Set.Icc (-r) r,
      |(s - q s) ^ 2 - (u - q u) ^ 2| ≤ L * |s - u|)
    (hgeom : 2 * spread cbk ≤ r - cbk.c (Fin.last K)) : 2 * spread cbk ≤ L := by
  have hqr : q r = cbk.c (Fin.last K) := isNearest_top cbk hq hcr.le
  have hqc : q (cbk.c (Fin.last K)) = cbk.c (Fin.last K) := isNearest_top cbk hq le_rfl
  have h := le_of_lipschitzOn_chord hrc le_rfl hcr hqr hqc hLip
  linarith

/-- `22.627 ≤ √512`; the true value is `22.62741…`.  §60's `sqrt_512_ge` gives only `22.6`, which
is not enough for the comparison below. -/
theorem sqrt_512_ge_sharp : (22.627 : ℝ) ≤ Real.sqrt 512 := by
  have h : (22.627 : ℝ) = Real.sqrt (22.627 ^ 2) := (Real.sqrt_sq (by norm_num)).symm
  rw [h]
  exact Real.sqrt_le_sqrt (by norm_num)

/-- **The transport route's Lipschitz constant is at least `6.95` times twice the spread**, at
`d = 512` and for the shipped two-bit codebook's geometry: largest codeword `1.5104/√512`, spread
at most `3.0208/√512`, support `[−1, 1]`.  The chord from `t = c + (99/100)(1 − c)` to the support
edge `t = 1` already forces `L ≥ (199/100)(1 − 1.5104/√512) = 1.8572`, against
`2 · spread = 0.26699`.  The exact ratio at the sup of `|ψ′|` is
`(√512 − 1.5104)/3.0208 = 6.9905`; `6.95` is what this chord delivers.

The ratio grows like `√d/3.0208`: it is `1` only at `√d = 4.5312`, that is `d ≈ 21`. -/
theorem le_lipschitz_512 {K : ℕ} (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) {L : ℝ}
    (hLip : ∀ s ∈ Set.Icc (-1 : ℝ) 1, ∀ t ∈ Set.Icc (-1 : ℝ) 1,
      |(s - q s) ^ 2 - (t - q t) ^ 2| ≤ L * |s - t|)
    (hspread : spread cbk ≤ 3.0208 / Real.sqrt 512)
    (htop : cbk.c (Fin.last K) = 1.5104 / Real.sqrt 512) :
    695 / 100 * (2 * spread cbk) ≤ L := by
  have hs : (22.627 : ℝ) ≤ Real.sqrt 512 := sqrt_512_ge_sharp
  set s := Real.sqrt 512 with hsdef
  have hs0 : (0 : ℝ) < s := by linarith
  set c := (1.5104 : ℝ) / s with hcdef
  have hc0 : (0 : ℝ) < c := by rw [hcdef]; positivity
  have hc1 : c ≤ 1 / 10 := by
    rw [hcdef, div_le_div_iff₀ hs0 (by norm_num)]
    linarith
  set t := c + 99 / 100 * (1 - c) with htdef
  have hct : c ≤ t := by rw [htdef]; nlinarith
  have htr : t < 1 := by rw [htdef]; nlinarith
  have hqr : q 1 = c := by rw [← htop]; exact isNearest_top cbk hq (by rw [htop]; linarith)
  have hqt : q t = c := by rw [← htop]; exact isNearest_top cbk hq (by rw [htop]; linarith)
  have hchord := le_of_lipschitzOn_chord (r := 1) (by linarith) hct htr hqr hqt hLip
  have heq : (1 : ℝ) + t - 2 * c = 199 / 100 * (1 - c) := by rw [htdef]; ring
  rw [heq] at hchord
  have hnum : (0 : ℝ) ≤ 1.99 * s - 44.994816 := by linarith
  have hkey : (695 : ℝ) / 100 * (2 * (3.0208 / s)) ≤ 199 / 100 * (1 - c) := by
    rw [hcdef, ← sub_nonneg]
    have hid : (199 : ℝ) / 100 * (1 - 1.5104 / s) - 695 / 100 * (2 * (3.0208 / s))
        = (1.99 * s - 44.994816) / s := by field_simp; ring
    rw [hid]
    exact div_nonneg hnum hs0.le
  have hmono : (695 : ℝ) / 100 * (2 * spread cbk) ≤ 695 / 100 * (2 * (3.0208 / s)) := by
    linarith [hspread]
  linarith

/-- **The comparison, at the dimension the paper ships.**  At `d = 512`, for the shipped two-bit
codebook's geometry and *any* pair of laws, the threshold route's bound is at most `1/6.95` of the
transport route's:

  `6.95 · (2 · W₁ · spread cbk · 512) ≤ 512 · L · W₁`.

Both sides are the printed right-hand sides of `abs_hadRate_sub_rhoSq_le_spread_wOne` and
`abs_hadRate_sub_rhoSq_le_wOne` at the same `W₁`, so this is a comparison of the two routes and
not of two different quantities.  It goes against the transport route. -/
theorem threshold_le_transport_512 {K : ℕ} (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) {L : ℝ}
    (hLip : ∀ s ∈ Set.Icc (-1 : ℝ) 1, ∀ t ∈ Set.Icc (-1 : ℝ) 1,
      |(s - q s) ^ 2 - (t - q t) ^ 2| ≤ L * |s - t|)
    (hspread : spread cbk ≤ 3.0208 / Real.sqrt 512)
    (htop : cbk.c (Fin.last K) = 1.5104 / Real.sqrt 512) (μ ν : Measure ℝ) :
    695 / 100 * (2 * wOne μ ν * spread cbk * (512 : ℝ)) ≤ (512 : ℝ) * L * wOne μ ν :=
  mul_threshold_le_transport (by norm_num) (wOne_nonneg μ ν)
    (le_lipschitz_512 cbk hq hLip hspread htop)

end Transport

end TransportRoute
