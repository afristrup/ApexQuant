
/-! ## §63  The threshold form read as a characterisation, not only as a bound

§55's `distortion_sub_distortion_eq_threshold` is an **identity**, so for a fixed ordered codebook
and a fixed nearest-codeword map the distortion gap of two laws depends on the pair of laws
through exactly `K` real numbers: the once-integrated CDF differences
`cdfGapInt μ ν a (thr cbk k)` at the `K` thresholds.  §55 reads that identity in one direction
only, as an upper bound.  This section reads it in both, and records what therefore cannot be
read off it at all.

**What the `K` numbers determine.**  `distortion_sub_distortion_congr_of_cdfGapInt` is the sharp
form of the question.  Two pairs of laws that agree in integrated CDF at the `K` thresholds have
the *same* distortion gap, whatever they do anywhere else.  The statement allows two different
nearest maps of the same codebook and two different enclosing intervals, because the right-hand
side of the identity names neither.  Nothing about the laws away from the thresholds enters, and
that is the whole content: the gap is a function on `ℝ^K`, evaluated at the threshold vector.
`distortion_eq_distortion_of_cdfGapInt_eq_zero` is the case at zero, where the conclusion is an
equality of distortions with no bound and no smallness hypothesis anywhere in the statement.

**Upper, in total variation.**  `codewordTV c = Σ_k |c_{k+1} − c_k|` is the quantity the identity
actually produces, and `abs_distortion_sub_distortion_le_codewordTV` is the bound in that form:
`|D(q, μ) − D(q, ν)| ≤ 2 · codewordTV cbk.c · M` for any `M` bounding the `K` threshold numbers.
Monotonicity of the codebook is what collapses `codewordTV` to `spread`, and only that
(`codewordTV_eq_spread`); the weighted-sum lemma
`abs_sum_fin_weighted_le_codewordTV` needs no order hypothesis on `c` at all, unlike §50's
`abs_sum_fin_weighted_le`.  `abs_distortion_sub_distortion_le_codewordTV_sup'` is the same bound
with `M` taken to be the largest of the `K` numbers, which needs `K ≠ 0` to have a largest.
`exists_abs_cdfGapInt_gt` is the contrapositive, and is the **necessity** half: a gap exceeding
`2 · codewordTV cbk.c · M` forces *some* threshold to carry an integrated CDF difference
exceeding `M`.  Smallness at the `K` thresholds is therefore not merely sufficient for a small
gap, it is necessary, at the cost of the factor `2 · codewordTV cbk.c`.

**Lower.**  `le_abs_distortion_sub_distortion` is the largest-single-term-minus-the-rest bound,
`2 · (w_j · |G_j| − Σ_{k ≠ j} w_k · |G_k|) ≤ |D(q, μ) − D(q, ν)|` for every `j`, where
`w_k = c_{k+1} − c_k` and `G_k` is the `k`-th threshold number.  It is an equality-free
consequence of the triangle inequality applied to the identity, and it is stated for every `j`
rather than for a maximiser, so it needs no `K ≠ 0`.
`le_abs_distortion_sub_distortion_of_rest_le` is the usable form: if the other `K − 1` numbers
are bounded by `ε`, the rest is bounded by `(codewordTV cbk.c − w_j) · ε` and one large threshold
term forces a large gap.  `distortion_ne_distortion_of_rest_le` is the conclusion in the form
that bites, a strict separation of the two distortions.

**Cancellation, and what therefore cannot hold.**  No lower bound in terms of the largest of the
`K` numbers alone can hold unconditionally, because the weighted sum can vanish while the numbers
do not.  `exists_weighted_sum_eq_zero` exhibits the cancellation for any codebook with at least
two thresholds, and `not_exists_pos_lower_bound_of_max` is the refutation: there is no `κ > 0`
with `κ · |G_k| ≤ |Σ_j w_j G_j|` for all `G` and all `k`.  Both are stated about the right-hand
side of the identity **as a function of the `K` numbers**, not about a pair of laws: no measures
are constructed here, and it is *not* claimed that every vector of `K` numbers is realised by a
pair of laws meeting §55's hypotheses.  That realisation question is not addressed anywhere in
this development, so what is refuted is a lower bound on the identity's weighted sum, which is
the only object the gap is ever a function of.  This is why the two-sided statement
`abs_distortion_sub_distortion_sandwich` brackets the gap by a *signed* lower expression that can
be negative, and not by a multiple of the largest threshold number.

**What is not claimed.**  Nothing here bounds `cdfGapInt` for any particular pair of laws, so
every statement below is conditional on numbers it does not supply.  In particular nothing here
says that the deployed randomised Hadamard rotation attains the Haar rate `ρ²`, nothing here
evaluates `cdfGapInt` for the deployed coordinate law against the spherical one, and no codebook
the paper ships is mentioned.  No continuity, Lipschitz or piecewise-smoothness property of the
stage error `t ↦ (t − q t)²` is used or proved anywhere below: the identity it all rests on needs
only §52's `sq_sub_eq_of_isNearest` and the midpoint placement of the thresholds, and the
characterisation is extracted from that identity by the triangle inequality and nothing else.
The factor `2 · codewordTV cbk.c` relating the two sides is not claimed to be optimal. -/

namespace Threshold

variable {K : ℕ}

/-! ### The total variation of a codeword sequence -/

/-- **The total variation of a codeword sequence**, `Σ_k |c_{k+1} − c_k|`.  This, and not the
spread, is the quantity §55's identity produces when its summands are bounded one at a time: the
identity weights the `k`-th threshold number by `c_{k+1} − c_k`, and only monotonicity turns the
sum of the absolute values of those weights into `c_K − c_0`.  No order hypothesis on `c`. -/
noncomputable def codewordTV (c : Fin (K + 1) → ℝ) : ℝ :=
  ∑ k : Fin K, |c k.succ - c k.castSucc|

/-- The total variation is nonnegative, with no hypothesis on `c`.  At `K = 0` it is zero. -/
theorem codewordTV_nonneg (c : Fin (K + 1) → ℝ) : 0 ≤ codewordTV c :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- For a monotone sequence the absolute values may be dropped. -/
theorem codewordTV_eq_sum_of_monotone {c : Fin (K + 1) → ℝ} (hc : Monotone c) :
    codewordTV c = ∑ k : Fin K, (c k.succ - c k.castSucc) :=
  Finset.sum_congr rfl fun k _ => abs_of_nonneg (sub_castSucc_nonneg hc k)

/-- **For an ordered codebook the total variation is the spread.**  This is the only place the
strict order of `Codebook` is spent in the upper bounds below, and it is what identifies the
general form of the characterisation with §55's `spread`-shaped bound. -/
theorem codewordTV_eq_spread (cbk : Codebook K) : codewordTV cbk.c = spread cbk := by
  rw [codewordTV_eq_sum_of_monotone cbk.mono.monotone, spread, sum_fin_telescope]

/-- Deleting one increment from a monotone sequence leaves the rest of the total variation. -/
theorem sum_erase_eq_codewordTV_sub {c : Fin (K + 1) → ℝ} (hc : Monotone c) (j : Fin K) :
    ∑ k ∈ Finset.univ.erase j, (c k.succ - c k.castSucc)
      = codewordTV c - (c j.succ - c j.castSucc) := by
  have h := Finset.sum_erase_add (Finset.univ : Finset (Fin K))
    (fun k : Fin K => c k.succ - c k.castSucc) (Finset.mem_univ j)
  rw [codewordTV_eq_sum_of_monotone hc]
  linarith

/-! ### The two-sided weighted-sum bounds, as pure algebra -/

/-- **The upper bound in total-variation form.**  A uniform bound on the summands bounds the
weighted sum by the total variation of the weights, for an *arbitrary* sequence `c`.  §50's
`abs_sum_fin_weighted_le` is the monotone specialisation; the proof here is the same calculation
with `|c_{k+1} − c_k|` in place of `c_{k+1} − c_k`, which is what removes the order hypothesis.
No sign hypothesis on `M`; at `K = 0` both sides are zero. -/
theorem abs_sum_fin_weighted_le_codewordTV (c : Fin (K + 1) → ℝ) (G : Fin K → ℝ) {M : ℝ}
    (hG : ∀ k, |G k| ≤ M) :
    |∑ k : Fin K, (c k.succ - c k.castSucc) * G k| ≤ codewordTV c * M := by
  calc |∑ k : Fin K, (c k.succ - c k.castSucc) * G k|
      ≤ ∑ k : Fin K, |(c k.succ - c k.castSucc) * G k| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k : Fin K, |c k.succ - c k.castSucc| * M := by
        refine Finset.sum_le_sum fun k _ => ?_
        rw [abs_mul]
        exact mul_le_mul_of_nonneg_left (hG k) (abs_nonneg _)
    _ = codewordTV c * M := by rw [← Finset.sum_mul, codewordTV]

/-- **The lower bound: one term minus the rest.**  For every index `j`, the weighted sum is at
least the `j`-th term in absolute value minus the sum of the absolute values of the others.  This
is the triangle inequality on the splitting of the sum at `j`, and it is the whole of what the
identity supports from below: the right-hand side can be negative, in which case the statement is
vacuous, and `not_exists_pos_lower_bound_of_max` shows that no bound in terms of the largest
`|G k|` alone can replace it. -/
theorem sub_sum_erase_le_abs_sum_fin_weighted (c : Fin (K + 1) → ℝ) (G : Fin K → ℝ) (j : Fin K) :
    |c j.succ - c j.castSucc| * |G j|
        - ∑ k ∈ Finset.univ.erase j, |c k.succ - c k.castSucc| * |G k|
      ≤ |∑ k : Fin K, (c k.succ - c k.castSucc) * G k| := by
  set w : Fin K → ℝ := fun k => c k.succ - c k.castSucc with hwdef
  have hsplit : w j * G j
      = (∑ k : Fin K, w k * G k) - ∑ k ∈ Finset.univ.erase j, w k * G k := by
    rw [← Finset.add_sum_erase Finset.univ (fun k => w k * G k) (Finset.mem_univ j)]
    ring
  have hR : |∑ k ∈ Finset.univ.erase j, w k * G k|
      ≤ ∑ k ∈ Finset.univ.erase j, |w k| * |G k| :=
    (Finset.abs_sum_le_sum_abs _ _).trans
      (Finset.sum_le_sum fun k _ => le_of_eq (abs_mul _ _))
  have h1 : |w j * G j|
      ≤ |∑ k : Fin K, w k * G k| + |∑ k ∈ Finset.univ.erase j, w k * G k| := by
    rw [hsplit]; exact abs_sub _ _
  rw [← abs_mul]
  linarith

/-! ### Cancellation: no lower bound in terms of the largest threshold number -/

/-- Every increment of an ordered codebook is strictly positive. -/
theorem sub_castSucc_pos (cbk : Codebook K) (k : Fin K) :
    0 < cbk.c k.succ - cbk.c k.castSucc :=
  sub_pos.mpr (cbk.mono Fin.castSucc_lt_succ)

/-- **The threshold numbers can cancel.**  For a codebook with at least two thresholds there is a
nonzero vector of threshold numbers whose weighted sum, the right-hand side of §55's identity, is
exactly zero: put the weight of the second threshold at the first, and minus the weight of the
first at the second.  Both entries are nonzero because the increments of an ordered codebook are
strictly positive.

**What this is a statement about.**  It is a statement about the identity's right-hand side as a
function of the `K` threshold numbers, and *not* about a pair of probability laws: no measures
are constructed, and it is not claimed here that this vector of threshold numbers is realised by
any pair of laws satisfying §55's support, mean and second-moment hypotheses.  The realisation
question is not addressed anywhere in this development. -/
theorem exists_weighted_sum_eq_zero (cbk : Codebook K) (hK : 2 ≤ K) :
    ∃ G : Fin K → ℝ, (∑ k : Fin K, (cbk.c k.succ - cbk.c k.castSucc) * G k = 0)
      ∧ ∃ k : Fin K, G k ≠ 0 := by
  classical
  obtain ⟨j0, j1, hne⟩ : ∃ j0 j1 : Fin K, j0 ≠ j1 :=
    ⟨⟨0, by omega⟩, ⟨1, by omega⟩, by simp [Fin.ext_iff]⟩
  refine ⟨fun k => if k = j0 then cbk.c j1.succ - cbk.c j1.castSucc
      else if k = j1 then -(cbk.c j0.succ - cbk.c j0.castSucc) else 0, ?_, j0, ?_⟩
  · show ∑ k : Fin K, (cbk.c k.succ - cbk.c k.castSucc)
        * (if k = j0 then cbk.c j1.succ - cbk.c j1.castSucc
            else if k = j1 then -(cbk.c j0.succ - cbk.c j0.castSucc) else 0) = 0
    rw [Finset.sum_eq_add_of_mem j0 j1 (Finset.mem_univ _) (Finset.mem_univ _) hne
      (fun k _ hk => by rw [if_neg hk.1, if_neg hk.2, mul_zero])]
    rw [if_pos rfl, if_neg hne.symm, if_pos rfl]
    ring
  · show (if j0 = j0 then cbk.c j1.succ - cbk.c j1.castSucc
        else if j0 = j1 then -(cbk.c j0.succ - cbk.c j0.castSucc) else 0) ≠ 0
    rw [if_pos rfl]
    exact ne_of_gt (sub_castSucc_pos cbk j1)

/-- **No lower bound in terms of the largest threshold number alone.**  For a codebook with at
least two thresholds there is no positive `κ` with `κ · |G k| ≤ |Σ_j w_j G_j|` for every vector
`G` of threshold numbers and every index `k`, because `exists_weighted_sum_eq_zero` makes the
right-hand side vanish while the left-hand side does not.  This is why the necessity half of the
characterisation is stated as `exists_abs_cdfGapInt_gt`, which locates *some* large threshold
number given a large gap, rather than as a converse bound on a prescribed one, and why
`le_abs_distortion_sub_distortion` has to subtract the other terms.

As in `exists_weighted_sum_eq_zero`, this is about the identity's right-hand side as a function
of the `K` numbers and not about a pair of laws. -/
theorem not_exists_pos_lower_bound_of_max (cbk : Codebook K) (hK : 2 ≤ K) :
    ¬ ∃ κ : ℝ, 0 < κ ∧ ∀ G : Fin K → ℝ, ∀ k : Fin K,
        κ * |G k| ≤ |∑ j : Fin K, (cbk.c j.succ - cbk.c j.castSucc) * G j| := by
  rintro ⟨κ, hκ, hbd⟩
  obtain ⟨G, hsum, k, hk⟩ := exists_weighted_sum_eq_zero cbk hK
  have h := hbd G k
  rw [hsum, abs_zero] at h
  exact absurd h (not_le.mpr (mul_pos hκ (abs_pos.mpr hk)))

/-! ### The gap depends on the laws only through the `K` threshold numbers -/

/-- **The sharp form: nothing away from the thresholds matters.**  Two pairs of laws that agree
in once-integrated CDF at the `K` thresholds have the same distortion gap.  The two pairs may
live on different enclosing intervals and may be read by different nearest-codeword maps of the
same codebook, because §55's identity names neither the interval's right endpoint nor the map on
its right-hand side.  This is the characterisation the upper and lower bounds below quantify: the
gap is a function of `K` real numbers, and the two laws enter only through those numbers.

Each pair must satisfy §55's hypotheses separately.  Nothing is asserted about a pair that fails
them, and nothing is asserted about the size of either gap. -/
theorem distortion_sub_distortion_congr_of_cdfGapInt (cbk : Codebook K) {q q' : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hq' : IsNearest (Set.range cbk.c) q')
    (μ ν μ' ν' : Measure ℝ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    [IsProbabilityMeasure μ'] [IsProbabilityMeasure ν'] (a b a' b' : ℝ)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hthr' : ∀ k : Fin K, thr cbk k ∈ Set.Icc a' b')
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hsupp_mu' : μ' (Set.Ioc a' b')ᶜ = 0) (hsupp_nu' : ν' (Set.Ioc a' b')ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν)
    (hmean' : ∫ t, t ∂μ' = ∫ t, t ∂ν') (hsq' : ∫ t, t ^ 2 ∂μ' = ∫ t, t ^ 2 ∂ν')
    (hG : ∀ k : Fin K, cdfGapInt μ ν a (thr cbk k) = cdfGapInt μ' ν' a' (thr cbk k)) :
    distortion q μ - distortion q ν = distortion q' μ' - distortion q' ν' := by
  rw [distortion_sub_distortion_eq_threshold cbk hq μ ν a b hthr hsupp_mu hsupp_nu hmean hsq,
    distortion_sub_distortion_eq_threshold cbk hq' μ' ν' a' b' hthr' hsupp_mu' hsupp_nu'
      hmean' hsq']
  exact congrArg _ (Finset.sum_congr rfl fun k _ => by rw [hG k])

/-- **Agreement at the `K` thresholds forces equal distortions.**  The case of
`distortion_sub_distortion_congr_of_cdfGapInt` at the pair `(μ, μ)`, stated directly: if the
once-integrated CDF difference vanishes at every threshold then the two distortions are equal.
No bound, no smallness hypothesis, and no hypothesis at all on the laws away from the `K`
thresholds beyond §55's support, mean and second-moment requirements. -/
theorem distortion_eq_distortion_of_cdfGapInt_eq_zero (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a b : ℝ) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν)
    (hG : ∀ k : Fin K, cdfGapInt μ ν a (thr cbk k) = 0) :
    distortion q μ = distortion q ν := by
  have h := distortion_sub_distortion_eq_threshold cbk hq μ ν a b hthr hsupp_mu hsupp_nu hmean hsq
  have hz : ∑ k : Fin K, (cbk.c k.succ - cbk.c k.castSucc) * cdfGapInt μ ν a (thr cbk k) = 0 :=
    Finset.sum_eq_zero fun k _ => by rw [hG k, mul_zero]
  rw [hz, mul_zero] at h
  linarith

/-! ### The upper half, and its contrapositive -/

/-- **The upper bound in total-variation form.**  Under §55's hypotheses, any `M` bounding the
`K` threshold numbers gives `|D(q, μ) − D(q, ν)| ≤ 2 · codewordTV cbk.c · M`.  By
`codewordTV_eq_spread` this is §55's `abs_distortion_sub_distortion_le_of_cdfGapInt` with the
total variation in place of the spread; it is stated this way because the total variation is what
the identity produces and is the form that survives dropping monotonicity of the weights.
Conditional on `M`: no bound on `cdfGapInt` is proved here for any pair of laws. -/
theorem abs_distortion_sub_distortion_le_codewordTV (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a b : ℝ) {M : ℝ} (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν)
    (hM : ∀ k : Fin K, |cdfGapInt μ ν a (thr cbk k)| ≤ M) :
    |distortion q μ - distortion q ν| ≤ 2 * codewordTV cbk.c * M := by
  have hw := abs_sum_fin_weighted_le_codewordTV cbk.c
    (fun k : Fin K => cdfGapInt μ ν a (thr cbk k)) hM
  rw [distortion_sub_distortion_eq_threshold cbk hq μ ν a b hthr hsupp_mu hsupp_nu hmean hsq,
    abs_mul]
  have habs : |(-2 : ℝ)| = 2 := by norm_num
  rw [habs]
  linarith

/-- **The upper bound at the largest of the `K` threshold numbers.**  The same bound with `M`
taken to be `max_k |cdfGapInt μ ν a (thr cbk k)|`.  The nonemptiness hypothesis is what gives the
maximum a value, and it is exactly `K ≠ 0`; at `K = 0` the gap is zero by §55's identity and
there is nothing to state. -/
theorem abs_distortion_sub_distortion_le_codewordTV_sup' (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a b : ℝ) (hne : (Finset.univ : Finset (Fin K)).Nonempty)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν) :
    |distortion q μ - distortion q ν|
      ≤ 2 * codewordTV cbk.c
          * Finset.univ.sup' hne fun k => |cdfGapInt μ ν a (thr cbk k)| :=
  abs_distortion_sub_distortion_le_codewordTV cbk hq μ ν a b hthr hsupp_mu hsupp_nu hmean hsq
    fun k => Finset.le_sup' (fun k => |cdfGapInt μ ν a (thr cbk k)|) (Finset.mem_univ k)

/-- **Necessity: a large gap forces a large threshold number.**  The contrapositive of
`abs_distortion_sub_distortion_le_codewordTV`.  If the distortion gap exceeds
`2 · codewordTV cbk.c · M` then some threshold carries an integrated CDF difference exceeding
`M`.  Together with the upper bound this is the characterisation in its crudest form: closeness
of the two laws at the `K` thresholds is necessary as well as sufficient for a small gap, with
the factor `2 · codewordTV cbk.c` between the two directions.  Which threshold is large is not
determined, and `not_exists_pos_lower_bound_of_max` shows it cannot be: a prescribed threshold
number can be large while the gap is zero. -/
theorem exists_abs_cdfGapInt_gt (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a b : ℝ) {M : ℝ} (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν)
    (hgt : 2 * codewordTV cbk.c * M < |distortion q μ - distortion q ν|) :
    ∃ k : Fin K, M < |cdfGapInt μ ν a (thr cbk k)| := by
  by_contra hcon
  have hM : ∀ k : Fin K, |cdfGapInt μ ν a (thr cbk k)| ≤ M := fun k =>
    not_lt.mp fun h => hcon ⟨k, h⟩
  exact absurd (abs_distortion_sub_distortion_le_codewordTV cbk hq μ ν a b hthr hsupp_mu
    hsupp_nu hmean hsq hM) (not_le.mpr hgt)

/-! ### The lower half -/

/-- **One threshold term minus the rest bounds the gap from below.**  For every index `j`,

  `2 · (w_j · |G_j| − Σ_{k ≠ j} w_k · |G_k|) ≤ |D(q, μ) − D(q, ν)|`,

where `w_k = c_{k+1} − c_k` and `G_k = cdfGapInt μ ν a (thr cbk k)`.  The left-hand side is
signed and is often negative, in which case the statement says nothing; that is not a defect of
the proof but the content of `not_exists_pos_lower_bound_of_max`, since the terms can cancel
exactly.  Monotonicity of the codebook is used only to drop the absolute values on the weights. -/
theorem le_abs_distortion_sub_distortion (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a b : ℝ) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν) (j : Fin K) :
    2 * ((cbk.c j.succ - cbk.c j.castSucc) * |cdfGapInt μ ν a (thr cbk j)|
          - ∑ k ∈ Finset.univ.erase j,
              (cbk.c k.succ - cbk.c k.castSucc) * |cdfGapInt μ ν a (thr cbk k)|)
      ≤ |distortion q μ - distortion q ν| := by
  have habs1 : ∀ k : Fin K, |cbk.c k.succ - cbk.c k.castSucc| = cbk.c k.succ - cbk.c k.castSucc :=
    fun k => abs_of_nonneg (sub_castSucc_nonneg cbk.mono.monotone k)
  have h := sub_sum_erase_le_abs_sum_fin_weighted cbk.c
    (fun k : Fin K => cdfGapInt μ ν a (thr cbk k)) j
  simp only [habs1] at h
  rw [distortion_sub_distortion_eq_threshold cbk hq μ ν a b hthr hsupp_mu hsupp_nu hmean hsq,
    abs_mul]
  have habs : |(-2 : ℝ)| = 2 := by norm_num
  rw [habs]
  linarith

/-- **The usable lower bound: one large threshold number against `K − 1` small ones.**  If every
threshold other than `j` carries an integrated CDF difference at most `ε`, then

  `2 · (w_j · |G_j| − (codewordTV cbk.c − w_j) · ε) ≤ |D(q, μ) − D(q, ν)|`.

The rest of the total variation, `codewordTV cbk.c − w_j`, is exactly the weight available to
cancel the `j`-th term, by `sum_erase_eq_codewordTV_sub`.  The hypothesis is on the other
thresholds only; nothing is assumed about `G_j`. -/
theorem le_abs_distortion_sub_distortion_of_rest_le (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a b : ℝ) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν) (j : Fin K) {ε : ℝ}
    (hrest : ∀ k : Fin K, k ≠ j → |cdfGapInt μ ν a (thr cbk k)| ≤ ε) :
    2 * ((cbk.c j.succ - cbk.c j.castSucc) * |cdfGapInt μ ν a (thr cbk j)|
          - (codewordTV cbk.c - (cbk.c j.succ - cbk.c j.castSucc)) * ε)
      ≤ |distortion q μ - distortion q ν| := by
  have hbd : ∑ k ∈ Finset.univ.erase j,
        (cbk.c k.succ - cbk.c k.castSucc) * |cdfGapInt μ ν a (thr cbk k)|
      ≤ (codewordTV cbk.c - (cbk.c j.succ - cbk.c j.castSucc)) * ε := by
    rw [← sum_erase_eq_codewordTV_sub cbk.mono.monotone j, Finset.sum_mul]
    refine Finset.sum_le_sum fun k hk => ?_
    exact mul_le_mul_of_nonneg_left (hrest k (Finset.ne_of_mem_erase hk))
      (sub_castSucc_nonneg cbk.mono.monotone k)
  have h := le_abs_distortion_sub_distortion cbk hq μ ν a b hthr hsupp_mu hsupp_nu hmean hsq j
  linarith

/-- **One threshold that the others cannot cancel separates the two distortions.**  If the other
`K − 1` threshold numbers are at most `ε` and the `j`-th term strictly exceeds the weight the
others could contribute, then `D(q, μ) ≠ D(q, ν)`.  This is the lower half in the form that
bites: the gap is not merely bounded below by a possibly negative quantity, it is nonzero. -/
theorem distortion_ne_distortion_of_rest_le (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a b : ℝ) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν) (j : Fin K) {ε : ℝ}
    (hrest : ∀ k : Fin K, k ≠ j → |cdfGapInt μ ν a (thr cbk k)| ≤ ε)
    (hdom : (codewordTV cbk.c - (cbk.c j.succ - cbk.c j.castSucc)) * ε
      < (cbk.c j.succ - cbk.c j.castSucc) * |cdfGapInt μ ν a (thr cbk j)|) :
    distortion q μ ≠ distortion q ν := by
  intro heq
  have h := le_abs_distortion_sub_distortion_of_rest_le cbk hq μ ν a b hthr hsupp_mu hsupp_nu
    hmean hsq j hrest
  rw [heq, sub_self, abs_zero] at h
  linarith

/-! ### The characterisation -/

/-- **The characterisation.**  Under §55's hypotheses the distortion gap is pinned, from above
and from below, by the `K` once-integrated CDF differences at the thresholds and by nothing else:

  `2 · (w_j · |G_j| − Σ_{k ≠ j} w_k · |G_k|) ≤ |D(q, μ) − D(q, ν)| ≤ 2 · TV · max_k |G_k|`

for every `j`, where `w_k = c_{k+1} − c_k`, `TV = codewordTV cbk.c` and
`G_k = cdfGapInt μ ν a (thr cbk k)`.  The two sides are separated by a factor determined by the
codebook alone, its total variation, which is its spread when it is ordered
(`codewordTV_eq_spread`).  Together with
`distortion_sub_distortion_congr_of_cdfGapInt`, which says the gap is a function of the `K`
numbers alone, this is the answer to "for which pairs of laws is the gap small": exactly those
whose integrated CDFs agree at the `K` thresholds, to within that factor, and the behaviour of
the laws anywhere else is irrelevant.

**Read the lower bound with care.**  Its left-hand side is signed, and
`not_exists_pos_lower_bound_of_max` shows that this is unavoidable: no lower bound in terms of
`max_k |G_k|` alone can hold, because the `K` terms can cancel exactly.  So the sandwich is not a
two-sided estimate by a single quantity, and the factor `2 · TV` is not claimed to be optimal.
As everywhere in this section, no `G_k` is evaluated for any particular pair of laws. -/
theorem abs_distortion_sub_distortion_sandwich (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a b : ℝ) (hne : (Finset.univ : Finset (Fin K)).Nonempty)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν) :
    (∀ j : Fin K,
        2 * ((cbk.c j.succ - cbk.c j.castSucc) * |cdfGapInt μ ν a (thr cbk j)|
              - ∑ k ∈ Finset.univ.erase j,
                  (cbk.c k.succ - cbk.c k.castSucc) * |cdfGapInt μ ν a (thr cbk k)|)
          ≤ |distortion q μ - distortion q ν|)
      ∧ |distortion q μ - distortion q ν|
          ≤ 2 * codewordTV cbk.c
              * Finset.univ.sup' hne fun k => |cdfGapInt μ ν a (thr cbk k)| :=
  ⟨fun j => le_abs_distortion_sub_distortion cbk hq μ ν a b hthr hsupp_mu hsupp_nu hmean hsq j,
    abs_distortion_sub_distortion_le_codewordTV_sup' cbk hq μ ν a b hne hthr hsupp_mu hsupp_nu
      hmean hsq⟩

/-- **The characterisation with the spread in place of the total variation.**  The same sandwich
read through `codewordTV_eq_spread`, so that the upper half matches §55's
`abs_distortion_sub_distortion_le_of_cdfGapInt` verbatim in shape: the factor separating the two
halves is `2 · spread cbk`.  Stated separately because the total-variation form is the general
one and the spread form is what the ordered codebooks the paper ships make of it. -/
theorem abs_distortion_sub_distortion_le_spread_sup' (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a b : ℝ) (hne : (Finset.univ : Finset (Fin K)).Nonempty)
    (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν) :
    |distortion q μ - distortion q ν|
      ≤ 2 * spread cbk * Finset.univ.sup' hne fun k => |cdfGapInt μ ν a (thr cbk k)| := by
  rw [← codewordTV_eq_spread cbk]
  exact abs_distortion_sub_distortion_le_codewordTV_sup' cbk hq μ ν a b hne hthr hsupp_mu
    hsupp_nu hmean hsq

end Threshold
