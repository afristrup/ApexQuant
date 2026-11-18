
/-! ## §55  The threshold form of the scalar distortion gap

This is the headline of the threshold sections, and it is an **identity**, not a bound.  For an
ordered scalar codebook and any nearest-codeword map of it, and for two probability laws that
live on a common bounded interval and agree in their first two moments, the difference of the two
distortions is

  `D(q, μ) − D(q, ν) = −2 · Σ_k (c_{k+1} − c_k) · ∫_{(a, τ_k]} (cdf μ − cdf ν)`,

a finite sum over the `K` thresholds `τ_k`, weighted by the codeword increments, of the
once-integrated difference of the two cumulative distribution functions.  Nothing is dropped and
nothing is estimated: both sides are the same real number.

Where each hypothesis is spent.  Equal **second** moments cancel the `∫ t²` term, which is the
only place they are used and is already done in §54.  Equal **means** are used exactly once, at
the very end: together with the half-open support they force `cdfGapInt μ ν a b = 0`, and that is
what kills the last surviving boundary term `2 · c_K · cdfGapInt μ ν a b` left over by the
summation by parts of §50.  Without matched means that term is present, and the identity as
stated does not follow.  The midpoint placement of the thresholds is spent inside §50's
`sum_cell_gap_eq_boundary_sub_threshold`, where it is what makes the mass contributions cancel at
every interior threshold.  No tie convention is used anywhere: §52's `sq_sub_eq_of_isNearest`
makes the value of `q` at a threshold immaterial, so `q` is an arbitrary nearest map.

Two hypotheses are stronger than the bare shape of the statement suggests, and both are stated
honestly rather than hidden.  The support hypothesis is half-open on the left,
`ρ (Ioc a b)ᶜ = 0`, not `ρ (Icc a b)ᶜ = 0`: it is what makes `cdf ρ a = 0`, so that the boundary
term at the left endpoint is an honest zero rather than a convention.  It costs nothing for a
compactly supported law, since `compl_Ioc_eq_zero_of_Icc` moves the left endpoint strictly below
the support, and both laws the paper compares are compactly supported.  The hypothesis `hthr`
places every threshold inside `[a, b]`; it is what identifies the two outer cells with the two
outer intervals, and it is a genuine restriction on the codebook relative to the interval, not a
bookkeeping device.  No hypothesis `a ≤ b` appears: for a probability law supported in `(a, b]`
the interval cannot be empty, which is `le_of_supported_Ioc`.

`abs_distortion_sub_distortion_le` reads the identity as a bound, and is the only inequality
here.  A sup-norm bound `η` on `cdf μ − cdf ν` bounds each once-integrated term by `η · (b − a)`,
and the codeword increments telescope to the spread, giving
`|D(q, μ) − D(q, ν)| ≤ 2 · η · spread cbk · (b − a)`.  This is a conditional statement about a
pair of laws: it says what a Kolmogorov-Smirnov bound buys, and it supplies no such bound.

**What is not claimed.**  Neither theorem says the distortion gap is small.  The identity is an
exact rewriting whose right-hand side is a sum of signed real numbers with no sign attached to
any of them; the bound is vacuous until some `η` is supplied from elsewhere.  In particular
nothing here says that the deployed randomised Hadamard rotation attains the Haar rate `ρ²`, and
nothing here supplies the `η` for the pair of laws the paper compares: that the coordinate law of
`H D_ε v` is close in Kolmogorov-Smirnov distance to the spherical coordinate law is §44's
business, and no theorem in this section invokes it or is specialised to those two laws.
Neither theorem is specific to any codebook the paper ships, and `K` is not claimed to be
the number of
levels of any deployed quantiser. -/

namespace Threshold

variable {K : ℕ}

/-! ### Two hypotheses that discharge themselves -/

/-- A probability law supported in `(a, b]` forces the interval to be nonempty, hence `a ≤ b`.
This is why no separate ordering hypothesis appears below. -/
theorem le_of_supported_Ioc {ρ : Measure ℝ} [IsProbabilityMeasure ρ] {a b : ℝ}
    (hsupp : ρ (Set.Ioc a b)ᶜ = 0) : a ≤ b := by
  by_contra hcon
  have he : Set.Ioc a b = (∅ : Set ℝ) := Set.Ioc_eq_empty fun h => hcon h.le
  have huniv : ρ Set.univ = 0 := by
    rw [← Set.compl_empty, ← he]
    exact hsupp
  rw [measure_univ] at huniv
  exact one_ne_zero huniv

/-- **The half-open support hypothesis costs nothing.**  A law supported in a closed interval is
supported in any half-open interval that contains it with room to spare on the left.  This is how
the two compactly supported laws of §52 meet the hypothesis of the identity below, and it is also
what lets the interval carrying the codebook's thresholds be wider than the one carrying the
laws. -/
theorem compl_Ioc_eq_zero_of_Icc {ρ : Measure ℝ} {a a' b' b : ℝ} (haa : a < a') (hbb : b' ≤ b)
    (hsupp : ρ (Set.Icc a' b')ᶜ = 0) : ρ (Set.Ioc a b)ᶜ = 0 :=
  measure_mono_null
    (Set.compl_subset_compl.mpr fun _t ht => ⟨lt_of_lt_of_le haa ht.1, le_trans ht.2 hbb⟩) hsupp

/-! ### Matched means kill the last boundary term -/

/-- **Matched means make the once-integrated CDF difference vanish over the whole interval.**
For two probability laws supported in `(a, b]` the CDF difference is zero at both endpoints, so
`cell_first_moment_gap` reduces to `cdfGapInt μ ν a b = ∫ t ∂ν − ∫ t ∂μ`.  This is the only place
equal means are used, and it is exactly what removes the boundary term left by the summation by
parts. -/
theorem cdfGapInt_eq_zero_of_mean {μ ν : Measure ℝ} [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] {a b : ℝ} (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0)
    (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0) (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) :
    cdfGapInt μ ν a b = 0 := by
  have hab : a ≤ b := le_of_supported_Ioc hsupp_mu
  have htrimmu : ∫ t, t ∂μ = ∫ t in Set.Ioc a b, t ∂μ := by
    have h := setIntegral_eq_of_supported_Ioc hsupp_mu (fun t : ℝ => t) Set.univ
    rw [Set.univ_inter, setIntegral_univ] at h
    exact h
  have htrimnu : ∫ t, t ∂ν = ∫ t in Set.Ioc a b, t ∂ν := by
    have h := setIntegral_eq_of_supported_Ioc hsupp_nu (fun t : ℝ => t) Set.univ
    rw [Set.univ_inter, setIntegral_univ] at h
    exact h
  have h := cell_first_moment_gap μ ν hab
  rw [cdf_eq_zero_of_supported_Ioc hsupp_mu, cdf_eq_zero_of_supported_Ioc hsupp_nu,
    cdf_eq_one_of_supported_Ioc hsupp_mu, cdf_eq_one_of_supported_Ioc hsupp_nu,
    ← htrimmu, ← htrimnu, hmean] at h
  linarith

/-! ### The threshold identity -/

/-- **The threshold form of the distortion gap.**  Let `cbk` be an ordered scalar codebook whose
thresholds all lie in `[a, b]`, let `q` be *any* nearest-codeword map for it, and let `μ` and `ν`
be probability laws supported in `(a, b]` with equal means and equal second moments.  Then

  `D(q, μ) − D(q, ν) = −2 · Σ_k (c k.succ − c k.castSucc) · cdfGapInt μ ν a (thr cbk k)`.

This is an exact identity between two real numbers.  It is not a bound, it asserts no sign for
any summand, and it does not say the gap is small.  The right-hand side involves the two laws
only through the once-integrated CDF difference at the `K` thresholds. -/
theorem distortion_sub_distortion_eq_threshold (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a b : ℝ) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν) :
    distortion q μ - distortion q ν
      = -2 * ∑ k : Fin K,
          (cbk.c k.succ - cbk.c k.castSucc) * cdfGapInt μ ν a (thr cbk k) := by
  have hab : a ≤ b := le_of_supported_Ioc hsupp_mu
  have hzero : cdfGapInt μ ν a b = 0 :=
    cdfGapInt_eq_zero_of_mean hsupp_mu hsupp_nu hmean
  have hGlast : bdryGapInt cbk μ ν a b (Fin.last (K + 1)) = 0 := by
    rw [bdryGapInt, bdry_last, hzero]
  have hsum : ∑ k : Fin K,
        (cbk.c k.succ - cbk.c k.castSucc) * bdryGapInt cbk μ ν a b k.succ.castSucc
      = ∑ k : Fin K,
          (cbk.c k.succ - cbk.c k.castSucc) * cdfGapInt μ ν a (thr cbk k) :=
    Finset.sum_congr rfl fun k _ => by rw [bdryGapInt_succ_castSucc]
  have hcell := distortion_sub_distortion_eq_cellGap cbk hq μ ν hab hthr hsupp_mu hsupp_nu hsq
  have habel := sum_cell_gap_eq_boundary_sub_threshold (K := K) cbk.c (bdry cbk a b)
    (bdryCdfGap cbk μ ν a b) (bdryGapInt cbk μ ν a b) (c_add_c_eq_two_mul_bdry cbk a b)
  have hmain := hcell.trans habel
  rw [bdryCdfGap_zero cbk hsupp_mu hsupp_nu, bdryCdfGap_last cbk hsupp_mu hsupp_nu,
    bdryGapInt_zero cbk μ ν a b, hGlast, hsum] at hmain
  rw [hmain]
  ring

/-! ### Reading the identity as a bound -/

/-- **A sup-norm bound on the CDF difference bounds the distortion gap.**  Under the hypotheses of
the identity, a uniform bound `η` on `|cdf μ − cdf ν|` gives

  `|D(q, μ) − D(q, ν)| ≤ 2 · η · spread cbk · (b − a)`,

since each once-integrated term is bounded by `η · (b − a)` and the codeword increments telescope
to the spread.  This is conditional on `η`: no bound on the Kolmogorov-Smirnov distance of any
particular pair of laws is proved here or used here. -/
theorem abs_distortion_sub_distortion_le (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a b : ℝ) {η : ℝ} (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν)
    (hks : ∀ u, |(cdf μ) u - (cdf ν) u| ≤ η) :
    |distortion q μ - distortion q ν| ≤ 2 * η * spread cbk * (b - a) := by
  have hab : a ≤ b := le_of_supported_Ioc hsupp_mu
  have hG : ∀ k : Fin K, |cdfGapInt μ ν a (thr cbk k)| ≤ η * (b - a) := by
    intro k
    have h := abs_bdryGapInt_le cbk μ ν hab hthr hks k.succ.castSucc
    rwa [bdryGapInt_succ_castSucc] at h
  have hw := abs_sum_fin_weighted_le (c := cbk.c) cbk.mono.monotone
    (fun k : Fin K => cdfGapInt μ ν a (thr cbk k)) hG
  rw [distortion_sub_distortion_eq_threshold cbk hq μ ν a b hthr hsupp_mu hsupp_nu hmean hsq,
    abs_mul]
  have habs : |(-2 : ℝ)| = 2 := by norm_num
  rw [habs]
  simp only [spread]
  nlinarith [hw, abs_nonneg (∑ k : Fin K,
    (cbk.c k.succ - cbk.c k.castSucc) * cdfGapInt μ ν a (thr cbk k))]

/-- **A uniform bound on `cdfGapInt` bounds the distortion gap, with no `(b − a)` factor.**  Under
the hypotheses of the identity, a bound `C` on `|cdfGapInt μ ν a x|` uniform in `x` gives

  `|D(q, μ) − D(q, ν)| ≤ 2 · C · spread cbk`.

The proof is shorter than `abs_distortion_sub_distortion_le`'s: `abs_sum_fin_weighted_le` only
ever needed a uniform bound on the summands `cdfGapInt μ ν a (thr cbk k)`, and here that bound is
supplied directly, with no detour through a sup-norm `η` on `cdf μ − cdf ν` and no factor of
`(b − a)`.  This is conditional on `C`: no such bound is proved here for any particular pair of
laws. -/
theorem abs_distortion_sub_distortion_le_of_cdfGapInt (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (μ ν : Measure ℝ) [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] (a b : ℝ) {C : ℝ} (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hsupp_mu : μ (Set.Ioc a b)ᶜ = 0) (hsupp_nu : ν (Set.Ioc a b)ᶜ = 0)
    (hmean : ∫ t, t ∂μ = ∫ t, t ∂ν) (hsq : ∫ t, t ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν)
    (hC : ∀ x, |cdfGapInt μ ν a x| ≤ C) :
    |distortion q μ - distortion q ν| ≤ 2 * C * spread cbk := by
  have hw := abs_sum_fin_weighted_le (c := cbk.c) cbk.mono.monotone
    (fun k : Fin K => cdfGapInt μ ν a (thr cbk k)) (fun k => hC (thr cbk k))
  rw [distortion_sub_distortion_eq_threshold cbk hq μ ν a b hthr hsupp_mu hsupp_nu hmean hsq,
    abs_mul]
  have habs : |(-2 : ℝ)| = 2 := by norm_num
  rw [habs]
  simp only [spread]
  nlinarith [hw]

end Threshold
