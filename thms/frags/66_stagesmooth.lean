
/-! ## §61  The stage error is smooth even though the quantiser is not

A nearest-codeword map `q` jumps: at a threshold it switches codewords, and the jump is the full
cell width, not a small one.  The quantity the rate integrates is not `q` but the stage error
`ψ t = (t − q t)²`, and that one does **not** jump.  This section makes "smoothness" a defined
notion for this development and proves the three facts that go with it.  It proves no bound on a
distortion gap and no bound on the deployed rate, and it is composed with §55's threshold
identity nowhere.

**The thresholds here really are midpoints.**  `thr cbk k = (c_k + c_{k+1})/2` by definition in
§52, so nothing below is conditional on a hypothesis that they are.  The midpoint content is
isolated as `sq_thr_sub_castSucc`: at `τ_k` the two candidate codewords are equidistant, since
`τ_k − c_k = −(τ_k − c_{k+1})`, so the two one-sided candidate values of `ψ` are equal squares.
`stageErr_thr_eq_castSucc` and `stageErr_thr_eq_succ` are that fact read at `τ_k`: the stage
error there is the same number whichever codeword is used, so the tie convention of `cellIdx` is
invisible to it, exactly as §52's `sq_sub_eq_of_isNearest` already made it invisible everywhere.

**Continuity is proved without any case analysis, and that is the point.**  For a nearest map,
`|t − q t| = dist(t, C)` (§16's `norm_sub_eq_infDist`), so `ψ = dist(·, C)²` with `C` the codeword
set, and `dist(·, C)` is `1`-Lipschitz for every set whatsoever.  `continuous_stageErr` is that
observation.  No cell decomposition, no one-sided limits, no ordering of the codebook and no
midpoint hypothesis enter it; the midpoint lemmas above explain *why* the two candidate values
agree, but the proof of continuity does not consume them.  That the quantiser itself is
genuinely discontinuous is recorded beside it: `stepQuant_jump` says the jump at `τ_k` is at
least `c_{k+1} − c_k` at **every** point above `τ_k`, however close, and
`not_continuousAt_stepQuant` turns that into the negation.

**The Lipschitz constant is twice the sup of the error.**  §25's `abs_stageErr_sub_le` already
gives the two-point bound `|ψ x − ψ y| ≤ |x − y|·(|x − q x| + |y − q y|)` in the vector setting;
`abs_stageErr_sub_le_of_sup` reads it on a set on which `|t − q t| ≤ M`, giving the constant
`2M`, and `lipschitzOnWith_stageErr` packages that as `LipschitzOnWith`.  The constant is not
pulled out of the air: `hasDerivAt_stageErr_of_isNearest` shows that off the thresholds `ψ` is
differentiable with `ψ' = 2(t − q t)`, so `|ψ'| = 2|t − q t|` (`abs_deriv_stageErr`) and `2M` is
the sup of `|ψ'|` rather than an overestimate of it.

**A sup to put in.**  `abs_sub_stepQuant_le_half_of_gap`: on `[c_0, c_K]`, if every cell has
width at most `g`, the error is at most `g/2`, half the widest cell.  Off that interval the error
is the overshoot, so on `[a, b]` the sup is at most
`max (g/2) (max (c_0 − a) (b − c_K))` (`abs_sub_nearest_le_of_mem_Icc`), and
`lipschitzOnWith_stageErr_Icc` is the Lipschitz statement with twice that.  This is an upper
bound for the sup, not the sup itself, and it is not claimed to be attained.

**The relation to `ρ²`, stated without overstating it.**  `ρ² = d·E_ν[ψ]` is `d` times the
**mean** of `ψ`, and the Lipschitz constant `L = 2·sup|t − q t|` is twice its **sup**.  The
inequality between them that is free runs one way only, mean `≤` sup: `rhoSq_le_of_abs_sub_le`
gives `ρ² ≤ d·M²`, equivalently `ρ² ≤ d·L²/4` (`rhoSq_le_of_lipschitz_const`).  **No inequality
in the other direction is proved here, and none is free.**  Nothing in this development bounds
`M`, or `L`, or `sup|t − q t|`, by any function of `ρ²`; a law may put almost all of its mass
where the error is tiny and a vanishing amount where it is large, making `ρ²` small and `L`
large.  `le_rhoSq_of_mass` records exactly what an argument in that direction has to supply: a
set of mass `ν(s)` on which the error is at least `m` forces `ρ² ≥ d·ν(s)·m²`, so a converse
needs a lower bound on the mass near the sup and this section provides none.

**What is not established.**  That `ψ` fails to be differentiable at a threshold: it has a corner
there, since the one-sided slopes are `±(c_{k+1} − c_k)`, but no theorem below asserts the
failure and none is proved.  Any smoothness beyond continuity everywhere and differentiability
off the finitely many thresholds; in particular `ψ` is not claimed to be `C¹`, and no second
derivative is discussed.  Any global Lipschitz bound: `ψ` grows quadratically away from the
codebook, so every Lipschitz statement here is on a bounded set and carries its `M` as a
hypothesis.  Anything about a particular codebook: `K` is arbitrary, no codeword is given a
value, and no numerical constant for any shipped codebook appears.  And nothing here compares
two laws, so no statement below bears on `|hadRate − ρ²|`. -/

namespace Threshold

variable {K : ℕ}

/-! ### The stage error as a function of the point -/

/-- **The stage error** `ψ t = (t − q t)²`: the integrand of `distortion`, and the object this
section proves smooth.  It is a function of the point `t` alone once `q` is fixed. -/
noncomputable def stageErr (q : ℝ → ℝ) (t : ℝ) : ℝ := (t - q t) ^ 2

theorem stageErr_apply (q : ℝ → ℝ) (t : ℝ) : stageErr q t = (t - q t) ^ 2 := rfl

theorem stageErr_nonneg (q : ℝ → ℝ) (t : ℝ) : 0 ≤ stageErr q t := sq_nonneg _

/-- The scalar distortion is the mean of the stage error.  This is a definitional rewriting and
carries no integrability hypothesis; it is stated so the rest of the section can speak of `ψ`. -/
theorem distortion_eq_integral_stageErr (q : ℝ → ℝ) (ν : Measure ℝ) :
    distortion q ν = ∫ t, stageErr q t ∂ν := rfl

/-- **`ρ²` is `d` times the mean of `ψ`.**  Definitional, and the statement the sup-versus-mean
discussion below is about. -/
theorem rhoSq_eq_mul_integral_stageErr (d : ℕ) (q : ℝ → ℝ) (ν : Measure ℝ) :
    rhoSq d q ν = (d : ℝ) * ∫ t, stageErr q t ∂ν := rfl

/-- The stage error is the same **function** for any two nearest maps of the same codebook, not
merely the same at each point separately.  This is §52's `sq_sub_eq_of_isNearest` as an equality
of functions, which is what the continuity and derivative statements below need. -/
theorem stageErr_congr_of_isNearest {C : Set ℝ} {q q' : ℝ → ℝ} (hq : IsNearest C q)
    (hq' : IsNearest C q') : stageErr q = stageErr q' :=
  funext fun t => sq_sub_eq_of_isNearest hq hq' t

/-- The error of a nearest map is the distance to the codebook, in scalar form. -/
theorem abs_sub_eq_infDist {C : Set ℝ} {q : ℝ → ℝ} (hq : IsNearest C q) (t : ℝ) :
    |t - q t| = Metric.infDist t C := by
  rw [← Real.norm_eq_abs]
  exact norm_sub_eq_infDist hq t

/-- **The stage error is the squared distance to the codebook.**  This is the identity the whole
section runs on: it replaces a piecewise formula, one branch per cell, by a single expression in
which the codebook enters only as a set. -/
theorem stageErr_eq_infDist_sq {C : Set ℝ} {q : ℝ → ℝ} (hq : IsNearest C q) (t : ℝ) :
    stageErr q t = Metric.infDist t C ^ 2 := by
  rw [stageErr, ← sq_abs (t - q t), abs_sub_eq_infDist hq]

/-- **The stage error is continuous everywhere, thresholds included.**  `dist(·, C)` is
`1`-Lipschitz for every set `C`, so its square is continuous; no ordering of the codebook, no
cell decomposition and no midpoint hypothesis is used, and `C` need not even be finite.  The
midpoint lemmas below say *why* the two one-sided candidate values agree at a threshold, but this
proof does not consume them. -/
theorem continuous_stageErr {C : Set ℝ} {q : ℝ → ℝ} (hq : IsNearest C q) :
    Continuous (stageErr q) := by
  have hfun : stageErr q = fun t : ℝ => Metric.infDist t C ^ 2 :=
    funext fun t => stageErr_eq_infDist_sq hq t
  rw [hfun]
  exact (Metric.continuous_infDist_pt C).pow 2

/-! ### The midpoint identity, which is why there is nothing to jump -/

/-- **The midpoint identity.**  At `τ_k` the displacement to the lower codeword is the negative of
the displacement to the upper one.  This is a two-line consequence of `thr` being defined as the
midpoint in §52, and it is the entire content of the continuity claim at a threshold. -/
theorem thr_sub_castSucc_eq_neg (cbk : Codebook K) (k : Fin K) :
    thr cbk k - cbk.c k.castSucc = -(thr cbk k - cbk.c k.succ) := by
  unfold thr; ring

/-- Hence the two candidate squared errors at a threshold are equal. -/
theorem sq_thr_sub_castSucc (cbk : Codebook K) (k : Fin K) :
    (thr cbk k - cbk.c k.castSucc) ^ 2 = (thr cbk k - cbk.c k.succ) ^ 2 := by
  rw [thr_sub_castSucc_eq_neg]; ring

/-- The thresholds strictly below `τ_k` are exactly those of smaller index. -/
theorem belowThr_thr (cbk : Codebook K) (k : Fin K) :
    belowThr cbk (thr cbk k) = Finset.Iio k := by
  ext j
  rw [mem_belowThr, Finset.mem_Iio]
  exact (thr_strictMono cbk).lt_iff_lt

theorem cellIdx_thr (cbk : Codebook K) (k : Fin K) :
    cellIdx cbk (thr cbk k) = k.castSucc := by
  refine Fin.ext ?_
  rw [cellIdx_val, belowThr_thr, Fin.card_Iio, Fin.val_castSucc]

/-- `stepQuant` breaks the tie at a threshold downwards, taking the lower codeword.  Which way it
breaks is immaterial to `ψ`, by `stageErr_thr_eq_succ`. -/
theorem stepQuant_thr (cbk : Codebook K) (k : Fin K) :
    stepQuant cbk (thr cbk k) = cbk.c k.castSucc := by
  rw [stepQuant, cellIdx_thr]

/-- The stage error at a threshold, computed with the **lower** codeword. -/
theorem stageErr_thr_eq_castSucc {cbk : Codebook K} {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (k : Fin K) :
    stageErr q (thr cbk k) = (thr cbk k - cbk.c k.castSucc) ^ 2 := by
  rw [stageErr_congr_of_isNearest hq (isNearest_stepQuant cbk), stageErr, stepQuant_thr]

/-- **The same number, computed with the upper codeword.**  The two one-sided candidate values of
`ψ` at `τ_k` agree, so there is nothing for `ψ` to jump by, whatever `q` does there. -/
theorem stageErr_thr_eq_succ {cbk : Codebook K} {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (k : Fin K) :
    stageErr q (thr cbk k) = (thr cbk k - cbk.c k.succ) ^ 2 := by
  rw [stageErr_thr_eq_castSucc hq, sq_thr_sub_castSucc]

/-- Above a threshold the quantiser has already moved to the upper codeword or beyond. -/
theorem le_stepQuant_of_thr_lt (cbk : Codebook K) {k : Fin K} {t : ℝ} (ht : thr cbk k < t) :
    cbk.c k.succ ≤ stepQuant cbk t := by
  have h := (thr_lt_iff_lt_cellIdx cbk t k).mp ht
  exact cbk.mono.monotone (by rw [Fin.le_def, Fin.val_succ]; omega)

/-- **The quantiser really does jump, and by the full cell width.**  At `τ_k` its value is `c_k`,
and at *every* point above `τ_k`, however close, it is at least `c_{k+1}`.  The gap is bounded
below uniformly, which is what `not_continuousAt_stepQuant` turns into a negation. -/
theorem stepQuant_jump (cbk : Codebook K) {k : Fin K} {t : ℝ} (ht : thr cbk k < t) :
    cbk.c k.succ - cbk.c k.castSucc ≤ stepQuant cbk t - stepQuant cbk (thr cbk k) := by
  rw [stepQuant_thr]
  linarith [le_stepQuant_of_thr_lt cbk ht]

/-- **`stepQuant` is discontinuous at every threshold**, against `continuous_stageErr`.  This is
the precise sense in which `q` jumps but the quantity the rate integrates does not. -/
theorem not_continuousAt_stepQuant (cbk : Codebook K) (k : Fin K) :
    ¬ ContinuousAt (stepQuant cbk) (thr cbk k) := by
  intro hc
  have hlim := hc.continuousWithinAt (s := Set.Ioi (thr cbk k))
  have hev : ∀ᶠ t in nhdsWithin (thr cbk k) (Set.Ioi (thr cbk k)),
      cbk.c k.succ ≤ stepQuant cbk t :=
    Filter.eventually_of_mem self_mem_nhdsWithin fun t ht => le_stepQuant_of_thr_lt cbk ht
  have hge := ge_of_tendsto hlim hev
  rw [stepQuant_thr] at hge
  exact absurd (cbk.mono Fin.castSucc_lt_succ) (not_lt.mpr hge)

/-! ### Lipschitz on a bounded set, with constant twice the sup of the error -/

/-- The two-point bound with the slope function `t ↦ |t − q t|`, which is §25's
`abs_stageErr_sub_le` read on the line. -/
theorem abs_stageErr_sub_le_slope {C : Set ℝ} {q : ℝ → ℝ} (hq : IsNearest C q) (x y : ℝ) :
    |stageErr q x - stageErr q y| ≤ |x - y| * (|x - q x| + |y - q y|) := by
  have h := abs_stageErr_sub_le hq x y
  simpa [stageErr, Real.norm_eq_abs, sq_abs] using h

/-- **The Lipschitz estimate, in explicit two-point form.**  On any set on which the error is at
most `M`, the stage error is Lipschitz with constant `2M`.  `M` is carried as a hypothesis: no
sup is computed here, and none is claimed to be attained. -/
theorem abs_stageErr_sub_le_of_sup {C : Set ℝ} {q : ℝ → ℝ} (hq : IsNearest C q) {s : Set ℝ}
    {M : ℝ} (hM : ∀ u ∈ s, |u - q u| ≤ M) {x y : ℝ} (hx : x ∈ s) (hy : y ∈ s) :
    |stageErr q x - stageErr q y| ≤ 2 * M * |x - y| := by
  refine (abs_stageErr_sub_le_slope hq x y).trans ?_
  have h1 := hM x hx
  have h2 := hM y hy
  nlinarith [abs_nonneg (x - y), abs_nonneg (x - q x), abs_nonneg (y - q y)]

/-- The same as `LipschitzOnWith`.  The constant is `(2M)⁺`, since `LipschitzOnWith` takes a
nonnegative constant and `M` is only assumed to bound the error, not to be nonnegative; on a
nonempty `s` the two agree. -/
theorem lipschitzOnWith_stageErr {C : Set ℝ} {q : ℝ → ℝ} (hq : IsNearest C q) {s : Set ℝ}
    {M : ℝ} (hM : ∀ u ∈ s, |u - q u| ≤ M) :
    LipschitzOnWith (Real.toNNReal (2 * M)) (stageErr q) s := by
  refine LipschitzOnWith.of_dist_le_mul fun x hx y hy => ?_
  rw [Real.dist_eq, Real.dist_eq]
  exact (abs_stageErr_sub_le_of_sup hq hM hx hy).trans
    (mul_le_mul_of_nonneg_right (Real.le_coe_toNNReal _) (abs_nonneg _))

/-! ### A sup to put in -/

/-- A nearest map errs by at most the distance to any single codeword. -/
theorem abs_sub_nearest_le_abs_sub_codeword {cbk : Codebook K} {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (t : ℝ) (j : Fin (K + 1)) :
    |t - q t| ≤ |t - cbk.c j| := by
  have h := (hq t).2 (cbk.c j) ⟨j, rfl⟩
  rwa [Real.norm_eq_abs, Real.norm_eq_abs] at h

/-- **Half the widest cell, inside the codebook range.**  If every cell has width at most `g`,
then on `[c_0, c_K]` the error is at most `g/2`.  This is where the midpoint placement of the
thresholds is spent a second time: a threshold elsewhere in the cell would give `g`, not `g/2`.
The bound is stated with `g` as a hypothesis and is not claimed to be attained. -/
theorem abs_sub_stepQuant_le_half_of_gap (cbk : Codebook K) {g : ℝ} (hg : 0 ≤ g)
    (hgap : ∀ k : Fin K, cbk.c k.succ - cbk.c k.castSucc ≤ g) {t : ℝ}
    (ht : t ∈ Set.Icc (cbk.c 0) (cbk.c (Fin.last K))) :
    |t - stepQuant cbk t| ≤ g / 2 := by
  rw [stepQuant, abs_le]
  constructor
  · rcases Nat.eq_zero_or_pos (cellIdx cbk t : ℕ) with h0 | hpos
    · have hz : cbk.c (cellIdx cbk t) = cbk.c 0 := by
        refine congrArg cbk.c (Fin.ext ?_)
        simpa using h0
      rw [hz]; linarith [ht.1]
    · have hK : (cellIdx cbk t : ℕ) - 1 < K := by
        have := (cellIdx cbk t).isLt
        omega
      set m : Fin K := ⟨(cellIdx cbk t : ℕ) - 1, hK⟩ with hm
      have hmlt : (m : ℕ) < (cellIdx cbk t : ℕ) := by simp only [hm]; omega
      have hthr : thr cbk m < t := (thr_lt_iff_lt_cellIdx cbk t m).mpr hmlt
      have hsucc : cbk.c m.succ = cbk.c (cellIdx cbk t) := by
        refine congrArg cbk.c (Fin.val_injective ?_)
        simp only [hm, Fin.val_succ]
        omega
      have hgm := hgap m
      rw [hsucc] at hgm
      unfold thr at hthr
      rw [hsucc] at hthr
      linarith
  · rcases eq_or_lt_of_le (Nat.lt_succ_iff.mp (cellIdx cbk t).isLt) with heq | hlt
    · have hz : cbk.c (cellIdx cbk t) = cbk.c (Fin.last K) := by
        refine congrArg cbk.c (Fin.ext ?_)
        simpa using heq
      rw [hz]; linarith [ht.2]
    · set m : Fin K := ⟨(cellIdx cbk t : ℕ), hlt⟩ with hm
      have hmge : ¬ ((m : ℕ) < (cellIdx cbk t : ℕ)) := by simp [hm]
      have hthr : t ≤ thr cbk m := not_lt.mp fun h =>
        hmge ((thr_lt_iff_lt_cellIdx cbk t m).mp h)
      have hcast : cbk.c m.castSucc = cbk.c (cellIdx cbk t) := by
        refine congrArg cbk.c (Fin.val_injective ?_)
        simp [hm]
      have hgm := hgap m
      rw [hcast] at hgm
      unfold thr at hthr
      rw [hcast] at hthr
      linarith

/-- **The sup of the error on a bounded interval.**  Inside the codebook range it is half the
widest cell; outside it is the overshoot past the outermost codeword, which is what the two other
branches of the `max` are.  For a clipped deployed codebook, `[a, b] = [−R, R]` with codewords in
`[−c, c]`, both overshoot branches are `R − c`, so the constant is `max (g/2) (R − c)`.  Every
branch is an upper bound, and the `max` is not claimed to be the sup. -/
theorem abs_sub_nearest_le_of_mem_Icc {cbk : Codebook K} {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) {g a b : ℝ} (hg : 0 ≤ g)
    (hgap : ∀ k : Fin K, cbk.c k.succ - cbk.c k.castSucc ≤ g) {t : ℝ} (ht : t ∈ Set.Icc a b) :
    |t - q t| ≤ max (g / 2) (max (cbk.c 0 - a) (b - cbk.c (Fin.last K))) := by
  by_cases hlo : t < cbk.c 0
  · refine (abs_sub_nearest_le_abs_sub_codeword hq t 0).trans ?_
    rw [abs_of_nonpos (by linarith)]
    exact le_max_of_le_right (le_max_of_le_left (by linarith [ht.1]))
  by_cases hhi : cbk.c (Fin.last K) < t
  · refine (abs_sub_nearest_le_abs_sub_codeword hq t (Fin.last K)).trans ?_
    rw [abs_of_nonneg (by linarith)]
    exact le_max_of_le_right (le_max_of_le_right (by linarith [ht.2]))
  · have hmem : t ∈ Set.Icc (cbk.c 0) (cbk.c (Fin.last K)) := ⟨not_lt.mp hlo, not_lt.mp hhi⟩
    have heq : |t - q t| = |t - stepQuant cbk t| := by
      rw [abs_sub_eq_infDist hq, abs_sub_eq_infDist (isNearest_stepQuant cbk)]
    rw [heq]
    exact (abs_sub_stepQuant_le_half_of_gap cbk hg hgap hmem).trans (le_max_left _ _)

/-- **The Lipschitz statement with an explicit constant.**  On `[a, b]`, the stage error of any
nearest map of an ordered codebook whose cells have width at most `g` is Lipschitz with constant
twice `max (g/2) (max (c_0 − a) (b − c_K))`.  Nothing here is asymptotic in `d` and no codebook
is fixed; the constant is whatever the codebook and the interval make it. -/
theorem lipschitzOnWith_stageErr_Icc {cbk : Codebook K} {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) {g a b : ℝ} (hg : 0 ≤ g)
    (hgap : ∀ k : Fin K, cbk.c k.succ - cbk.c k.castSucc ≤ g) :
    LipschitzOnWith
      (Real.toNNReal (2 * max (g / 2) (max (cbk.c 0 - a) (b - cbk.c (Fin.last K)))))
      (stageErr q) (Set.Icc a b) :=
  lipschitzOnWith_stageErr hq fun _ hu => abs_sub_nearest_le_of_mem_Icc hq hg hgap hu

/-! ### Off the thresholds the stage error is differentiable, with `ψ' = 2(t − q t)` -/

/-- Away from every threshold the cell index, hence the quantiser, is locally constant.  The
proof is the counting characterisation of `cellIdx`: the set of thresholds below a point does not
change under a small enough perturbation, since there are finitely many of them and none sits at
the point. -/
theorem eventuallyEq_stepQuant_of_not_thr (cbk : Codebook K) {t : ℝ}
    (ht : ∀ k : Fin K, thr cbk k ≠ t) :
    stepQuant cbk =ᶠ[nhds t] fun _ => stepQuant cbk t := by
  have hev : ∀ᶠ u in nhds t, ∀ k : Fin K, (thr cbk k < u ↔ thr cbk k < t) := by
    rw [Filter.eventually_all]
    intro k
    rcases lt_or_gt_of_ne (ht k) with h | h
    · filter_upwards [eventually_gt_nhds h] with u hu
      exact iff_of_true hu h
    · filter_upwards [eventually_lt_nhds h] with u hu
      exact iff_of_false (not_lt.mpr hu.le) (not_lt.mpr h.le)
  filter_upwards [hev] with u hu
  have hset : belowThr cbk u = belowThr cbk t := by
    ext k
    rw [mem_belowThr, mem_belowThr]
    exact hu k
  have hidx : cellIdx cbk u = cellIdx cbk t := by
    refine Fin.ext ?_
    rw [cellIdx_val, cellIdx_val, hset]
  show stepQuant cbk u = stepQuant cbk t
  rw [stepQuant, stepQuant, hidx]

/-- **The derivative of the stage error off the thresholds.**  There `ψ` agrees with a single
quadratic near the point, so `ψ' = 2(t − q t)`.  The hypothesis excludes the `K` thresholds and
nothing else, so this covers all but finitely many points. -/
theorem hasDerivAt_stageErr_of_not_thr (cbk : Codebook K) {t : ℝ}
    (ht : ∀ k : Fin K, thr cbk k ≠ t) :
    HasDerivAt (stageErr (stepQuant cbk)) (2 * (t - stepQuant cbk t)) t := by
  have hbase : HasDerivAt (fun u : ℝ => (u - stepQuant cbk t) ^ 2)
      (2 * (t - stepQuant cbk t)) t := by
    have hb0 : HasDerivAt (fun u : ℝ => u - stepQuant cbk t) 1 t := (hasDerivAt_id t).sub_const _
    have h : HasDerivAt (fun u : ℝ => (u - stepQuant cbk t) * (u - stepQuant cbk t))
        (1 * (t - stepQuant cbk t) + (t - stepQuant cbk t) * 1) t := hb0.mul hb0
    have hfun : (fun u : ℝ => (u - stepQuant cbk t) * (u - stepQuant cbk t))
        = fun u : ℝ => (u - stepQuant cbk t) ^ 2 := by
      funext u; ring
    rw [hfun] at h
    exact h.congr_deriv (by ring)
  refine hbase.congr_of_eventuallyEq ?_
  filter_upwards [eventuallyEq_stepQuant_of_not_thr cbk ht] with u hu
  rw [stageErr, hu]

/-- The same for an arbitrary nearest map, since `ψ` is the same function for all of them. -/
theorem hasDerivAt_stageErr_of_isNearest {cbk : Codebook K} {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) {t : ℝ} (ht : ∀ k : Fin K, thr cbk k ≠ t) :
    HasDerivAt (stageErr q) (2 * (t - stepQuant cbk t)) t := by
  rw [stageErr_congr_of_isNearest hq (isNearest_stepQuant cbk)]
  exact hasDerivAt_stageErr_of_not_thr cbk ht

/-- **`|ψ'| = 2|t − q t|`**, which is why `2·sup|t − q t|` is the Lipschitz constant and not a
larger number: it is the sup of `|ψ'|` over the set. -/
theorem abs_deriv_stageErr {cbk : Codebook K} {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (t : ℝ) :
    |2 * (t - stepQuant cbk t)| = 2 * |t - q t| := by
  rw [abs_mul, abs_sub_eq_infDist (isNearest_stepQuant cbk), abs_sub_eq_infDist hq]
  norm_num

/-! ### What the Lipschitz constant says about `ρ²`, and what it does not -/

/-- The mean of `ψ` is at most the square of a bound on the error.  This is the only direction
that is free. -/
theorem integral_stageErr_le {q : ℝ → ℝ} {ν : Measure ℝ} [IsProbabilityMeasure ν] {M : ℝ}
    (hint : Integrable (stageErr q) ν) (hM : ∀ᵐ t ∂ν, |t - q t| ≤ M) :
    ∫ t, stageErr q t ∂ν ≤ M ^ 2 := by
  have h : ∫ t, stageErr q t ∂ν ≤ ∫ _t : ℝ, M ^ 2 ∂ν := by
    refine integral_mono_ae hint (integrable_const _) ?_
    filter_upwards [hM] with t htM
    rw [stageErr, ← sq_abs (t - q t)]
    exact pow_le_pow_left₀ (abs_nonneg _) htM 2
  simpa using h

/-- **`ρ²` is `d` times a mean, and a mean is at most a sup.**  A bound `M` on the error gives
`ρ² ≤ d·M²`.  The converse is not available: see `le_rhoSq_of_mass` for what it would cost. -/
theorem rhoSq_le_of_abs_sub_le {d : ℕ} {q : ℝ → ℝ} {ν : Measure ℝ} [IsProbabilityMeasure ν]
    {M : ℝ} (hint : Integrable (stageErr q) ν) (hM : ∀ᵐ t ∂ν, |t - q t| ≤ M) :
    rhoSq d q ν ≤ (d : ℝ) * M ^ 2 := by
  rw [rhoSq_eq_mul_integral_stageErr]
  exact mul_le_mul_of_nonneg_left (integral_stageErr_le hint hM) (Nat.cast_nonneg d)

/-- **The same read against the Lipschitz constant.**  With `L = 2·sup|t − q t|` the constant of
`lipschitzOnWith_stageErr`, the relation to the stage distortion is `ρ² ≤ d·L²/4`, equivalently
`L ≥ 2·√(ρ²/d)`.  That is an inequality, not an identity: `ρ²` is `d` times the **mean** of `ψ`
and `L` is twice its **sup**, and this bounds the mean by the sup and nothing more. -/
theorem rhoSq_le_of_lipschitz_const {d : ℕ} {q : ℝ → ℝ} {ν : Measure ℝ} [IsProbabilityMeasure ν]
    {L : ℝ} (hint : Integrable (stageErr q) ν) (hL : ∀ᵐ t ∂ν, 2 * |t - q t| ≤ L) :
    rhoSq d q ν ≤ (d : ℝ) * L ^ 2 / 4 := by
  have hM : ∀ᵐ t ∂ν, |t - q t| ≤ L / 2 := by
    filter_upwards [hL] with t h
    linarith
  have hb := rhoSq_le_of_abs_sub_le (d := d) hint hM
  calc rhoSq d q ν ≤ (d : ℝ) * (L / 2) ^ 2 := hb
    _ = (d : ℝ) * L ^ 2 / 4 := by ring

/-- **What a converse would cost.**  A sup is not a mean, and no bound on `sup|t − q t|` by any
function of `ρ²` is proved here or anywhere in this development.  What does hold is this: a set
of mass `ν(s)` on which the error is at least `m` forces `ρ² ≥ d·ν(s)·m²`.  So an argument in the
other direction has to supply a lower bound on the mass the law puts near its sup, and none is
supplied here; for a law concentrating almost all of its mass where the error is small, `ρ²` is
small while the sup is not. -/
theorem le_rhoSq_of_mass {d : ℕ} {q : ℝ → ℝ} {ν : Measure ℝ} {s : Set ℝ} {m : ℝ}
    (hs : MeasurableSet s) (hfin : ν s ≠ ⊤) (hm : 0 ≤ m)
    (hint : Integrable (stageErr q) ν) (hsm : ∀ t ∈ s, m ≤ |t - q t|) :
    (d : ℝ) * (ν.real s * m ^ 2) ≤ rhoSq d q ν := by
  have hlow : ν.real s • m ^ 2 ≤ ∫ t in s, stageErr q t ∂ν := by
    refine setIntegral_ge_of_const_le hs hfin (fun t ht => ?_) hint.integrableOn
    rw [stageErr, ← sq_abs (t - q t)]
    exact pow_le_pow_left₀ hm (hsm t ht) 2
  have hup : ∫ t in s, stageErr q t ∂ν ≤ ∫ t, stageErr q t ∂ν :=
    setIntegral_le_integral hint (ae_of_all _ fun t => stageErr_nonneg q t)
  rw [smul_eq_mul] at hlow
  rw [rhoSq_eq_mul_integral_stageErr]
  exact mul_le_mul_of_nonneg_left (by linarith) (Nat.cast_nonneg d)

end Threshold
