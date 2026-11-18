
/-!
# §47b  Sphere-covering floor on the distortion of a finite codebook

For `U` uniform on the unit sphere of a `d`-dimensional space (`d ≥ 3`), a codebook `C` with at
most `N` points and `0 < s < 1`,

  `E min_{c ∈ C} ‖U − c‖² ≥ s² (1 − N σ(s))`,  `σ(s) = s^(d−1) / (√(2π(d−1)) √(1 − s²))`.

`SphereFloor.distortion_floor` takes the marginal law of `⟪e, U⟫` as a hypothesis;
`SphereFloor.distortion_floor_sphereUniform` discharges it with `sphereUniformAmbient_map_inner`,
so the floor holds for the uniform law of §21 with no marginal hypothesis.  This fragment is the
body of `floor/SphereFloor.lean` with `sphericalPdf` and `sphericalCoordLaw` taken from §4.
-/

namespace SphereFloor


open scoped InnerProductSpace

/-! ## Step A: geometry -/

/-- A point within distance `s < 1` of a unit vector `u` is nonzero and its direction lies in the
cap of angular radius `arcsin s` around `u`. -/
theorem cap_of_dist_lt {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {u c : E}
    {s : ℝ} (hu : ‖u‖ = 1) (hs0 : 0 < s) (hs1 : s < 1) (h : ‖u - c‖ ^ 2 < s ^ 2) :
    c ≠ 0 ∧ Real.sqrt (1 - s ^ 2) < ⟪u, (‖c‖⁻¹ : ℝ) • c⟫_ℝ := by
  have hs2 : s ^ 2 < 1 := by nlinarith
  have hc : c ≠ 0 := by
    rintro rfl
    rw [sub_zero, hu] at h
    linarith
  have hr : 0 < ‖c‖ := norm_pos_iff.mpr hc
  refine ⟨hc, ?_⟩
  set p : ℝ := ⟪u, (‖c‖⁻¹ : ℝ) • c⟫_ℝ with hp
  have hpc : ⟪u, c⟫_ℝ = ‖c‖ * p := by
    rw [hp, real_inner_smul_right]
    field_simp
  have hexp : ‖u - c‖ ^ 2 = 1 - 2 * (‖c‖ * p) + ‖c‖ ^ 2 := by
    rw [norm_sub_sq_real, hu, hpc]; ring
  rw [hexp] at h
  have hp0 : 0 < p := by
    by_contra hneg
    have : ‖c‖ * p ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hr.le (not_lt.mp hneg)
    nlinarith [sq_nonneg ‖c‖]
  rw [Real.sqrt_lt' hp0]
  nlinarith [sq_nonneg (‖c‖ - p)]

/-- Free-scale geometry: if a nonnegative multiple `a • c` is within distance `s < 1` of a unit
vector `u`, then `c ≠ 0`, `a > 0`, and the direction of `c` lies in the same cap around `u`. -/
theorem cap_of_dist_lt_scaled {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {u c : E} {s a : ℝ} (hu : ‖u‖ = 1) (hs0 : 0 < s) (hs1 : s < 1) (ha : 0 ≤ a)
    (h : ‖u - a • c‖ ^ 2 < s ^ 2) :
    c ≠ 0 ∧ 0 < a ∧ Real.sqrt (1 - s ^ 2) < ⟪u, (‖c‖⁻¹ : ℝ) • c⟫_ℝ := by
  obtain ⟨hac, hin⟩ := cap_of_dist_lt hu hs0 hs1 h
  have hc : c ≠ 0 := fun h0 => hac (by rw [h0, smul_zero])
  have ha0 : a ≠ 0 := fun h0 => hac (by rw [h0, zero_smul])
  have hapos : 0 < a := lt_of_le_of_ne ha (Ne.symm ha0)
  have hcn : ‖c‖ ≠ 0 := norm_ne_zero_iff.mpr hc
  have hdir : (‖a • c‖⁻¹ : ℝ) • (a • c) = (‖c‖⁻¹ : ℝ) • c := by
    rw [norm_smul, Real.norm_of_nonneg ha, smul_smul]
    congr 1
    field_simp
  rw [hdir] at hin
  exact ⟨hc, hapos, hin⟩

/-- The squared distance from `u` to the ray `{a • c : a ≥ 0}`, in closed form. -/
theorem iInf_scale_eq {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (u c : E) :
    (⨅ a : {a : ℝ // 0 ≤ a}, ‖u - a.1 • c‖ ^ 2)
      = ‖u‖ ^ 2 - (max 0 ⟪u, c⟫_ℝ) ^ 2 / ‖c‖ ^ 2 := by
  haveI : Nonempty {a : ℝ // 0 ≤ a} := ⟨⟨0, le_rfl⟩⟩
  have hexp : ∀ a : ℝ, ‖u - a • c‖ ^ 2 = ‖u‖ ^ 2 - 2 * a * ⟪u, c⟫_ℝ + a ^ 2 * ‖c‖ ^ 2 := by
    intro a
    rw [norm_sub_sq_real, real_inner_smul_right, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
    ring
  by_cases hc : c = 0
  · simp [hc]
  set p : ℝ := ⟪u, c⟫_ℝ with hp
  set m : ℝ := max 0 p with hm
  set r : ℝ := ‖c‖ ^ 2 with hr
  have hr0 : 0 < r := by rw [hr]; exact pow_pos (norm_pos_iff.mpr hc) 2
  have hm0 : 0 ≤ m := le_max_left _ _
  have hpm : p ≤ m := le_max_right _ _
  have hmp : m * p = m ^ 2 := by
    rcases le_total p 0 with h | h
    · rw [hm, max_eq_left h]; ring
    · rw [hm, max_eq_right h]; ring
  refine le_antisymm ?_ (le_ciInf fun a => ?_)
  · refine (ciInf_le ⟨0, ?_⟩ ⟨m / r, div_nonneg hm0 hr0.le⟩).trans (le_of_eq ?_)
    · rintro _ ⟨a, rfl⟩
      exact sq_nonneg _
    · rw [hexp]
      have h1 : 2 * (m / r) * p = 2 * (m ^ 2 / r) := by rw [← hmp]; ring
      have h2 : (m / r) ^ 2 * r = m ^ 2 / r := by field_simp
      rw [h1, h2]
      ring
  · rw [hexp]
    have ha : 0 ≤ a.1 := a.2
    have h1 : (2 * a.1 * p - a.1 ^ 2 * r) * r ≤ m ^ 2 := by
      nlinarith [sq_nonneg (m - a.1 * r), mul_nonneg (mul_nonneg ha hr0.le) (sub_nonneg.mpr hpm)]
    have h2 : m ^ 2 / r ≤ m ^ 2 / r := le_rfl
    have h3 : 2 * a.1 * p - a.1 ^ 2 * r ≤ m ^ 2 / r := by rwa [le_div_iff₀ hr0]
    linarith

/-! ## Step B: covering -/

/-- A finite minimum of measurable real functions is measurable. -/
theorem measurable_finset_inf' {ι Ω : Type*} [MeasurableSpace Ω] {C : Finset ι}
    (hC : C.Nonempty) {g : ι → Ω → ℝ} (hg : ∀ c, Measurable (g c)) :
    Measurable fun ω => C.inf' hC (fun c => g c ω) := by
  induction hC using Finset.Nonempty.cons_induction with
  | singleton a => simpa using hg a
  | cons a t ha ht ih =>
    simp only [Finset.inf'_cons ht]
    exact (hg a).min ih

section Covering

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [SecondCountableTopology E]

/-- **Covering bound, abstract form.**  If every cap `{⟪U, c⟫ > √(1 − s²)}` has probability at
most `σ`, and a nonnegative integrable `f` can be below `s²` only where `U` lies in the cap of
some nonzero codeword, then `E f ≥ s² (1 − N σ)`. -/
theorem covering_floor_of_cap {U : Ω → E} (hUm : Measurable U) (hU : ∀ᵐ ω ∂μ, ‖U ω‖ = 1)
    {C : Finset E} {N : ℕ} (hN : C.card ≤ N) {s σ : ℝ}
    (hcap : ∀ c : E, ‖c‖ = 1 → μ.real {ω | Real.sqrt (1 - s ^ 2) < ⟪U ω, c⟫_ℝ} ≤ σ)
    {f : Ω → ℝ} (hfi : Integrable f μ) (hfnn : ∀ ω, 0 ≤ f ω)
    (hf : ∀ᵐ ω ∂μ, f ω < s ^ 2 →
      ∃ c ∈ C, c ≠ 0 ∧ Real.sqrt (1 - s ^ 2) < ⟪U ω, (‖c‖⁻¹ : ℝ) • c⟫_ℝ) :
    s ^ 2 * (1 - N * σ) ≤ ∫ ω, f ω ∂μ := by
  classical
  set cap : E → Set Ω := fun c => {ω | Real.sqrt (1 - s ^ 2) < ⟪U ω, (‖c‖⁻¹ : ℝ) • c⟫_ℝ}
    with hcapdef
  set B : Set Ω := ⋃ c ∈ C.filter (fun c => c ≠ 0), cap c with hB
  have hcapm : ∀ c, MeasurableSet (cap c) := fun c =>
    measurableSet_lt measurable_const (hUm.inner measurable_const)
  have hBm : MeasurableSet B := by
    rw [hB]
    exact Finset.measurableSet_biUnion _ fun c _ => hcapm c
  have hσ : 0 ≤ σ := by
    obtain ⟨ω, hω⟩ := hU.exists
    exact measureReal_nonneg.trans (hcap (U ω) hω)
  have hBμ : μ.real B ≤ N * σ := by
    have h1 : μ.real B ≤ ∑ c ∈ C.filter (fun c => c ≠ 0), μ.real (cap c) :=
      measureReal_biUnion_finset_le _ _
    have h2 : ∑ c ∈ C.filter (fun c => c ≠ 0), μ.real (cap c)
        ≤ ∑ _c ∈ C.filter (fun c => c ≠ 0), σ := by
      refine Finset.sum_le_sum fun c hc => ?_
      have hc0 : c ≠ 0 := (Finset.mem_filter.mp hc).2
      exact hcap _ (by
        rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr hc0)])
    have h3 : ((C.filter (fun c => c ≠ 0)).card : ℝ) ≤ N := by
      exact_mod_cast (Finset.card_filter_le _ _).trans hN
    calc μ.real B ≤ ∑ _c ∈ C.filter (fun c => c ≠ 0), σ := h1.trans h2
      _ = (C.filter (fun c => c ≠ 0)).card * σ := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ N * σ := mul_le_mul_of_nonneg_right h3 hσ
  have hlow : ∀ᵐ ω ∂μ, Bᶜ.indicator (fun _ => s ^ 2) ω ≤ f ω := by
    filter_upwards [hf] with ω hω
    by_cases hωB : ω ∈ B
    · rw [Set.indicator_of_notMem (by simpa using hωB)]
      exact hfnn ω
    · rw [Set.indicator_of_mem (by simpa using hωB)]
      by_contra hlt
      obtain ⟨c, hc, hc0, hin⟩ := hω (not_le.mp hlt)
      refine hωB ?_
      rw [hB]
      simp only [Set.mem_iUnion]
      exact ⟨c, Finset.mem_filter.mpr ⟨hc, hc0⟩, hin⟩
  have hint : ∫ ω, Bᶜ.indicator (fun _ => s ^ 2) ω ∂μ = s ^ 2 * (1 - μ.real B) := by
    rw [integral_indicator hBm.compl, setIntegral_const, smul_eq_mul, measureReal_compl hBm,
      probReal_univ, mul_comm]
  calc s ^ 2 * (1 - N * σ) ≤ s ^ 2 * (1 - μ.real B) := by
        have : 0 ≤ s ^ 2 := sq_nonneg s
        nlinarith
    _ = ∫ ω, Bᶜ.indicator (fun _ => s ^ 2) ω ∂μ := hint.symm
    _ ≤ ∫ ω, f ω ∂μ :=
        integral_mono_ae ((integrable_const _).indicator hBm.compl) hfi hlow

/-- **Covering bound.**  If every cap `{⟪U, c⟫ > √(1 − s²)}` has probability at most `σ`, then a
codebook of at most `N` points has mean squared distance to `U` at least `s² (1 − N σ)`. -/
theorem covering_floor {U : Ω → E} (hUm : Measurable U) (hU : ∀ᵐ ω ∂μ, ‖U ω‖ = 1)
    {C : Finset E} (hC : C.Nonempty) {N : ℕ} (hN : C.card ≤ N) {s σ : ℝ} (hs0 : 0 < s)
    (hs1 : s < 1)
    (hcap : ∀ c : E, ‖c‖ = 1 → μ.real {ω | Real.sqrt (1 - s ^ 2) < ⟪U ω, c⟫_ℝ} ≤ σ) :
    s ^ 2 * (1 - N * σ) ≤ ∫ ω, C.inf' hC (fun c => ‖U ω - c‖ ^ 2) ∂μ := by
  set f : Ω → ℝ := fun ω => C.inf' hC (fun c => ‖U ω - c‖ ^ 2) with hf
  have hfm : Measurable f := by
    exact measurable_finset_inf' hC fun c => ((hUm.sub measurable_const).norm.pow_const 2)
  have hfnn : ∀ ω, 0 ≤ f ω := fun ω => by
    rw [hf]
    simp only [Finset.le_inf'_iff]
    exact fun c _ => sq_nonneg _
  have hfi : Integrable f μ := by
    have ⟨c0, hc0⟩ := hC
    refine Integrable.of_bound hfm.aestronglyMeasurable ((1 + ‖c0‖) ^ 2) ?_
    filter_upwards [hU] with ω hω
    rw [Real.norm_of_nonneg (hfnn ω)]
    refine (Finset.inf'_le _ hc0).trans ?_
    have h1 : ‖U ω - c0‖ ≤ 1 + ‖c0‖ := by
      rw [← hω]; exact norm_sub_le _ _
    exact pow_le_pow_left₀ (norm_nonneg _) h1 2
  refine covering_floor_of_cap hUm hU hN hcap hfi hfnn ?_
  filter_upwards [hU] with ω hω hlt
  obtain ⟨c, hc, hlt'⟩ := (Finset.inf'_lt_iff (H := hC)).mp hlt
  obtain ⟨hc0, hin⟩ := cap_of_dist_lt hω hs0 hs1 hlt'
  exact ⟨c, hc, hc0, hin⟩

/-- **Covering bound, free-scale form, for a chosen family of scales.**  The codeword `c` may be
rescaled by any nonnegative `a ω c` depending on the input. -/
theorem covering_floor_scales {U : Ω → E} (hUm : Measurable U) (hU : ∀ᵐ ω ∂μ, ‖U ω‖ = 1)
    {C : Finset E} (hC : C.Nonempty) {N : ℕ} (hN : C.card ≤ N) {s σ : ℝ} (hs0 : 0 < s)
    (hs1 : s < 1)
    (hcap : ∀ c : E, ‖c‖ = 1 → μ.real {ω | Real.sqrt (1 - s ^ 2) < ⟪U ω, c⟫_ℝ} ≤ σ)
    {a : Ω → E → ℝ} (ha : ∀ ω c, 0 ≤ a ω c)
    (hint : Integrable (fun ω => C.inf' hC (fun c => ‖U ω - a ω c • c‖ ^ 2)) μ) :
    s ^ 2 * (1 - N * σ) ≤ ∫ ω, C.inf' hC (fun c => ‖U ω - a ω c • c‖ ^ 2) ∂μ := by
  refine covering_floor_of_cap hUm hU hN hcap hint (fun ω => ?_) ?_
  · simp only [Finset.le_inf'_iff]
    exact fun c _ => sq_nonneg _
  · filter_upwards [hU] with ω hω hlt
    obtain ⟨c, hc, hlt'⟩ := (Finset.inf'_lt_iff (H := hC)).mp hlt
    obtain ⟨hc0, -, hin⟩ := cap_of_dist_lt_scaled hω hs0 hs1 (ha ω c) hlt'
    exact ⟨c, hc, hc0, hin⟩

/-- The free-scale integrand `min_c inf_{a ≥ 0} ‖U − a c‖²` is measurable, nonnegative and
integrable. -/
theorem scaled_integrand {U : Ω → E} (hUm : Measurable U) (hU : ∀ᵐ ω ∂μ, ‖U ω‖ = 1)
    {C : Finset E} (hC : C.Nonempty) :
    (∀ ω, 0 ≤ C.inf' hC (fun c => ⨅ a : {a : ℝ // 0 ≤ a}, ‖U ω - a.1 • c‖ ^ 2)) ∧
      Integrable (fun ω => C.inf' hC (fun c => ⨅ a : {a : ℝ // 0 ≤ a}, ‖U ω - a.1 • c‖ ^ 2)) μ := by
  haveI : Nonempty {a : ℝ // 0 ≤ a} := ⟨⟨0, le_rfl⟩⟩
  have hfnn : ∀ ω, 0 ≤ C.inf' hC (fun c => ⨅ a : {a : ℝ // 0 ≤ a}, ‖U ω - a.1 • c‖ ^ 2) := by
    intro ω
    simp only [Finset.le_inf'_iff]
    exact fun c _ => le_ciInf fun a => sq_nonneg _
  have hfm : Measurable
      fun ω => C.inf' hC (fun c => ⨅ a : {a : ℝ // 0 ≤ a}, ‖U ω - a.1 • c‖ ^ 2) := by
    simp only [iInf_scale_eq]
    exact measurable_finset_inf' hC fun c => (hUm.norm.pow_const 2).sub
      (((measurable_const.max (hUm.inner measurable_const)).pow_const 2).div_const _)
  refine ⟨hfnn, ?_⟩
  have ⟨c0, hc0⟩ := hC
  refine Integrable.of_bound hfm.aestronglyMeasurable 1 ?_
  filter_upwards [hU] with ω hω
  rw [Real.norm_of_nonneg (hfnn ω)]
  refine (Finset.inf'_le _ hc0).trans ((ciInf_le ⟨0, ?_⟩ ⟨0, le_rfl⟩).trans ?_)
  · rintro _ ⟨a, rfl⟩
    exact sq_nonneg _
  · simp [hω]

/-- **Covering bound, free-scale form.**  Each codeword may be rescaled by the best nonnegative
factor for each input; the bound is unchanged. -/
theorem covering_floor_scaled {U : Ω → E} (hUm : Measurable U) (hU : ∀ᵐ ω ∂μ, ‖U ω‖ = 1)
    {C : Finset E} (hC : C.Nonempty) {N : ℕ} (hN : C.card ≤ N) {s σ : ℝ} (hs0 : 0 < s)
    (hs1 : s < 1)
    (hcap : ∀ c : E, ‖c‖ = 1 → μ.real {ω | Real.sqrt (1 - s ^ 2) < ⟪U ω, c⟫_ℝ} ≤ σ) :
    s ^ 2 * (1 - N * σ)
      ≤ ∫ ω, C.inf' hC (fun c => ⨅ a : {a : ℝ // 0 ≤ a}, ‖U ω - a.1 • c‖ ^ 2) ∂μ := by
  haveI : Nonempty {a : ℝ // 0 ≤ a} := ⟨⟨0, le_rfl⟩⟩
  obtain ⟨hfnn, hfi⟩ := scaled_integrand (μ := μ) hUm hU hC
  refine covering_floor_of_cap hUm hU hN hcap hfi hfnn ?_
  filter_upwards [hU] with ω hω hlt
  obtain ⟨c, hc, hlt'⟩ := (Finset.inf'_lt_iff (H := hC)).mp hlt
  obtain ⟨a, ha⟩ := exists_lt_of_ciInf_lt hlt'
  obtain ⟨hc0, -, hin⟩ := cap_of_dist_lt_scaled hω hs0 hs1 a.2 ha
  exact ⟨c, hc, hc0, hin⟩

end Covering

/-! ## Step C: the one-dimensional cap integral -/

/-- `∫_a^1 (1 − t²)^((d−3)/2) dt ≤ (1 − a²)^((d−1)/2) / ((d − 1) a)` for `d ≥ 3`, `0 < a < 1`. -/
theorem cap_integral_le {d : ℕ} (hd : 3 ≤ d) {a : ℝ} (ha0 : 0 < a) (ha1 : a < 1) :
    ∫ t in a..1, (1 - t ^ 2) ^ (((d : ℝ) - 3) / 2)
      ≤ (1 - a ^ 2) ^ (((d : ℝ) - 1) / 2) / (((d : ℝ) - 1) * a) := by
  have hdR : (3 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
  have hq : (0 : ℝ) ≤ ((d : ℝ) - 3) / 2 := by linarith
  have hp : (1 : ℝ) ≤ ((d : ℝ) - 1) / 2 := by linarith
  have hd1 : (0 : ℝ) < (d : ℝ) - 1 := by linarith
  set q : ℝ := ((d : ℝ) - 3) / 2 with hqdef
  set p : ℝ := ((d : ℝ) - 1) / 2 with hpdef
  have hpq : p - 1 = q := by rw [hpdef, hqdef]; ring
  have hcf : Continuous fun t : ℝ => (1 - t ^ 2) ^ q :=
    (continuous_const.sub (continuous_pow 2)).rpow_const fun _ => Or.inr hq
  have hcg : Continuous fun t : ℝ => t / a * (1 - t ^ 2) ^ q :=
    (continuous_id.div_const a).mul hcf
  have hder : ∀ t ∈ Set.uIcc a 1,
      HasDerivAt (fun t : ℝ => -((1 - t ^ 2) ^ p / (((d : ℝ) - 1) * a)))
        (t / a * (1 - t ^ 2) ^ q) t := by
    intro t _
    have h1 : HasDerivAt (fun t : ℝ => 1 - t ^ 2) (-(2 * t)) t := by
      simpa using ((hasDerivAt_pow 2 t).const_sub 1)
    have h2 := (Real.hasDerivAt_rpow_const (x := 1 - t ^ 2) (p := p) (Or.inr hp)).comp t h1
    have h3 : HasDerivAt (fun t : ℝ => -((1 - t ^ 2) ^ p / (((d : ℝ) - 1) * a)))
        (-(p * (1 - t ^ 2) ^ (p - 1) * -(2 * t) / (((d : ℝ) - 1) * a))) t :=
      (h2.div_const (((d : ℝ) - 1) * a)).neg
    rw [hpq] at h3
    convert h3 using 1
    rw [hpdef]
    field_simp
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt hder (hcg.intervalIntegrable a 1)
  have hmono : ∫ t in a..1, (1 - t ^ 2) ^ q ≤ ∫ t in a..1, t / a * (1 - t ^ 2) ^ q := by
    refine intervalIntegral.integral_mono_on ha1.le (hcf.intervalIntegrable a 1)
      (hcg.intervalIntegrable a 1) fun t ht => ?_
    have hnn : 0 ≤ (1 - t ^ 2) ^ q :=
      Real.rpow_nonneg (by nlinarith [ht.1, ht.2]) q
    have h1 : 1 ≤ t / a := by rw [le_div_iff₀ ha0]; linarith [ht.1]
    nlinarith
  rw [hFTC] at hmono
  have h0 : ((1 : ℝ) - 1 ^ 2) ^ p = 0 := by
    rw [one_pow, sub_self, Real.zero_rpow (by linarith)]
  rw [h0] at hmono
  simpa using hmono

/-! ## Step D: Wendel's inequality -/

/-- **Wendel's inequality**: `Γ(x + 1/2) ≤ √x Γ(x)` for `x > 0`. -/
theorem gamma_add_half_le {x : ℝ} (hx : 0 < x) :
    Real.Gamma (x + 1 / 2) ≤ Real.sqrt x * Real.Gamma x := by
  have h := Real.Gamma_mul_add_mul_le_rpow_Gamma_mul_rpow_Gamma (s := x) (t := x + 1)
    (a := 1 / 2) (b := 1 / 2) hx (by linarith) (by norm_num) (by norm_num) (by norm_num)
  have hG : 0 < Real.Gamma x := Real.Gamma_pos_of_pos hx
  rw [show (1 / 2 : ℝ) * x + 1 / 2 * (x + 1) = x + 1 / 2 by ring, Real.Gamma_add_one hx.ne',
    ← Real.mul_rpow hG.le (mul_nonneg hx.le hG.le), ← Real.sqrt_eq_rpow,
    show Real.Gamma x * (x * Real.Gamma x) = x * Real.Gamma x ^ 2 by ring,
    Real.sqrt_mul hx.le, Real.sqrt_sq hG.le] at h
  exact h

/-- The normalising constant of `sphericalPdf d` is at most `√((d − 1)/(2π))`. -/
theorem sphericalPdf_const_le {d : ℕ} (hd : 2 ≤ d) :
    Real.Gamma ((d : ℝ) / 2) / (Real.sqrt π * Real.Gamma (((d : ℝ) - 1) / 2))
      ≤ Real.sqrt (((d : ℝ) - 1) / (2 * π)) := by
  have hdR : (2 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
  have hx : (0 : ℝ) < ((d : ℝ) - 1) / 2 := by linarith
  have hG : 0 < Real.Gamma (((d : ℝ) - 1) / 2) := Real.Gamma_pos_of_pos hx
  have hπ : 0 < Real.sqrt π := Real.sqrt_pos.mpr Real.pi_pos
  have hW := gamma_add_half_le hx
  rw [show ((d : ℝ) - 1) / 2 + 1 / 2 = (d : ℝ) / 2 by ring] at hW
  rw [div_le_iff₀ (mul_pos hπ hG)]
  have hsq : Real.sqrt (((d : ℝ) - 1) / (2 * π)) * Real.sqrt π
      = Real.sqrt (((d : ℝ) - 1) / 2) := by
    rw [← Real.sqrt_mul (div_nonneg (by linarith) (by positivity))]
    congr 1
    field_simp
  calc Real.Gamma ((d : ℝ) / 2) ≤ Real.sqrt (((d : ℝ) - 1) / 2) * Real.Gamma (((d : ℝ) - 1) / 2) := hW
    _ = Real.sqrt (((d : ℝ) - 1) / (2 * π)) * (Real.sqrt π * Real.Gamma (((d : ℝ) - 1) / 2)) := by
        rw [← hsq]; ring

/-! ## Step E: the cap measure bound -/

/-- The paper's `σ(s)` bound: `s^(d−1) / (√(2π(d−1)) √(1 − s²))`. -/
noncomputable def capBound (d : ℕ) (s : ℝ) : ℝ :=
  s ^ (d - 1) / (Real.sqrt (2 * π * ((d : ℝ) - 1)) * Real.sqrt (1 - s ^ 2))

/-- The cap `{t > √(1 − s²)}` has `f_d`-mass at most `capBound d s`. -/
theorem integral_sphericalPdf_cap_le {d : ℕ} (hd : 3 ≤ d) {s : ℝ} (hs0 : 0 < s) (hs1 : s < 1) :
    ∫ t in Real.sqrt (1 - s ^ 2)..1, sphericalPdf d t ≤ capBound d s := by
  have hdR : (3 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
  have hd1 : (0 : ℝ) < (d : ℝ) - 1 := by linarith
  have hs2 : 0 < 1 - s ^ 2 := by nlinarith
  set a : ℝ := Real.sqrt (1 - s ^ 2) with ha
  have ha0 : 0 < a := Real.sqrt_pos.mpr hs2
  have ha1 : a < 1 := by
    rw [ha, Real.sqrt_lt' one_pos]; nlinarith
  have ha2 : 1 - a ^ 2 = s ^ 2 := by rw [ha, Real.sq_sqrt hs2.le]; ring
  have hpow : (s ^ 2 : ℝ) ^ (((d : ℝ) - 1) / 2) = s ^ (d - 1) := by
    rw [← Real.rpow_natCast s 2, ← Real.rpow_mul hs0.le, ← Real.rpow_natCast s (d - 1)]
    congr 1
    rw [Nat.cast_sub (by omega : 1 ≤ d)]
    push_cast
    ring
  have hC := cap_integral_le hd ha0 ha1
  rw [ha2, hpow] at hC
  set K : ℝ := Real.Gamma ((d : ℝ) / 2) / (Real.sqrt π * Real.Gamma (((d : ℝ) - 1) / 2)) with hK
  have hK0 : 0 ≤ K := by
    rw [hK]
    exact div_nonneg (Real.Gamma_pos_of_pos (by linarith)).le
      (mul_nonneg (Real.sqrt_nonneg _) (Real.Gamma_pos_of_pos (by linarith)).le)
  have hKle : K ≤ Real.sqrt (((d : ℝ) - 1) / (2 * π)) := sphericalPdf_const_le (by omega)
  have hsn : 0 ≤ s ^ (d - 1) / (((d : ℝ) - 1) * a) :=
    div_nonneg (by positivity) (mul_pos hd1 ha0).le
  have hint : ∫ t in a..1, sphericalPdf d t = K * ∫ t in a..1, (1 - t ^ 2) ^ (((d : ℝ) - 3) / 2) := by
    simp only [sphericalPdf, ← hK]
    exact intervalIntegral.integral_const_mul K _
  rw [hint]
  have hroot : Real.sqrt (((d : ℝ) - 1) / (2 * π)) * Real.sqrt (2 * π * ((d : ℝ) - 1))
      = (d : ℝ) - 1 := by
    rw [← Real.sqrt_mul (div_nonneg hd1.le (by positivity)),
      show ((d : ℝ) - 1) / (2 * π) * (2 * π * ((d : ℝ) - 1)) = ((d : ℝ) - 1) ^ 2 by
        field_simp,
      Real.sqrt_sq hd1.le]
  have hrt : 0 < Real.sqrt (2 * π * ((d : ℝ) - 1)) :=
    Real.sqrt_pos.mpr (mul_pos (by positivity) hd1)
  calc K * ∫ t in a..1, (1 - t ^ 2) ^ (((d : ℝ) - 3) / 2)
      ≤ Real.sqrt (((d : ℝ) - 1) / (2 * π)) * (s ^ (d - 1) / (((d : ℝ) - 1) * a)) :=
        (mul_le_mul_of_nonneg_left hC hK0).trans (mul_le_mul_of_nonneg_right hKle hsn)
    _ = capBound d s := by
        rw [capBound, ← ha]
        generalize Real.sqrt (((d : ℝ) - 1) / (2 * π)) = S at hroot ⊢
        generalize Real.sqrt (2 * π * ((d : ℝ) - 1)) = A at hroot hrt ⊢
        have hS : S ≠ 0 := by
          rintro rfl
          rw [zero_mul] at hroot
          linarith
        rw [← hroot]
        field_simp

/-! ## Step F: assembly -/

/-- The mass that `sphericalCoordLaw d` gives to `(a, ∞)` is the integral of `f_d` over `(a, 1]`. -/
theorem sphericalCoordLaw_real_Ioi {d : ℕ} (hd : 3 ≤ d) {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    (sphericalCoordLaw d).real (Set.Ioi a) = ∫ t in a..1, sphericalPdf d t := by
  have hdR : (3 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
  have hcont : Continuous (sphericalPdf d) := by
    unfold sphericalPdf
    exact continuous_const.mul
      ((continuous_const.sub (continuous_pow 2)).rpow_const fun _ => Or.inr (by linarith))
  have hset : Set.Ioi a ∩ Set.Icc (-1 : ℝ) 1 = Set.Ioc a 1 := by
    ext t
    simp only [Set.mem_inter_iff, Set.mem_Ioi, Set.mem_Icc, Set.mem_Ioc]
    constructor
    · rintro ⟨h1, _, h3⟩; exact ⟨h1, h3⟩
    · rintro ⟨h1, h2⟩; exact ⟨h1, by linarith, h2⟩
  have hnn : 0 ≤ᵐ[volume.restrict (Set.Ioc a 1)] sphericalPdf d := by
    refine (ae_restrict_iff' measurableSet_Ioc).mpr (Filter.Eventually.of_forall fun t ht => ?_)
    unfold sphericalPdf
    refine mul_nonneg (div_nonneg (Real.Gamma_pos_of_pos (by linarith)).le
      (mul_nonneg (Real.sqrt_nonneg _) (Real.Gamma_pos_of_pos (by linarith)).le))
      (Real.rpow_nonneg ?_ _)
    nlinarith [ht.1, ht.2]
  have hi : Integrable (sphericalPdf d) (volume.restrict (Set.Ioc a 1)) :=
    hcont.integrableOn_Ioc
  rw [measureReal_def, sphericalCoordLaw, withDensity_apply _ measurableSet_Ioi,
    Measure.restrict_restrict measurableSet_Ioi, hset,
    ← ofReal_integral_eq_lintegral_ofReal hi hnn,
    ENNReal.toReal_ofReal (integral_nonneg_of_ae hnn), intervalIntegral.integral_of_le ha1]

section Assembly

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [SecondCountableTopology E]

omit [IsProbabilityMeasure μ] in
/-- Under the marginal hypothesis every cap has probability at most the paper's `σ(s)`. -/
theorem cap_measure_le {d : ℕ} (hd : 3 ≤ d) {U : Ω → E} (hUm : Measurable U)
    (hmarg : ∀ e : E, ‖e‖ = 1 → μ.map (fun ω => ⟪e, U ω⟫_ℝ) = sphericalCoordLaw d)
    {s : ℝ} (hs0 : 0 < s) (hs1 : s < 1) (c : E) (hc : ‖c‖ = 1) :
    μ.real {ω | Real.sqrt (1 - s ^ 2) < ⟪U ω, c⟫_ℝ}
      ≤ s ^ (d - 1) / (Real.sqrt (2 * π * ((d : ℝ) - 1)) * Real.sqrt (1 - s ^ 2)) := by
  have hm : Measurable fun ω => ⟪c, U ω⟫_ℝ := measurable_const.inner hUm
  have hs2 : 0 < 1 - s ^ 2 := by nlinarith
  have ha1 : Real.sqrt (1 - s ^ 2) ≤ 1 :=
    (Real.sqrt_le_sqrt (by nlinarith [sq_nonneg s] : 1 - s ^ 2 ≤ 1)).trans_eq Real.sqrt_one
  have hset : {ω | Real.sqrt (1 - s ^ 2) < ⟪U ω, c⟫_ℝ}
      = (fun ω => ⟪c, U ω⟫_ℝ) ⁻¹' Set.Ioi (Real.sqrt (1 - s ^ 2)) := by
    ext ω; simp [real_inner_comm]
  rw [hset, ← map_measureReal_apply hm measurableSet_Ioi, hmarg c hc,
    sphericalCoordLaw_real_Ioi hd (Real.sqrt_nonneg _) ha1]
  exact integral_sphericalPdf_cap_le hd hs0 hs1

/-- **Distortion floor.**  If `U` is almost surely a unit vector whose inner product with every
unit vector has the law `sphericalCoordLaw d` (as for the uniform law on the sphere, by
`sphereUniformAmbient_map_inner`), then a codebook of at most `N` points has
`E min_c ‖U − c‖² ≥ s² (1 − N s^(d−1) / (√(2π(d−1)) √(1 − s²)))` for every `0 < s < 1`. -/
theorem distortion_floor {d : ℕ} (hd : 3 ≤ d) {U : Ω → E} (hUm : Measurable U)
    (hU : ∀ᵐ ω ∂μ, ‖U ω‖ = 1)
    (hmarg : ∀ e : E, ‖e‖ = 1 → μ.map (fun ω => ⟪e, U ω⟫_ℝ) = sphericalCoordLaw d)
    {C : Finset E} (hC : C.Nonempty) {N : ℕ} (hN : C.card ≤ N) {s : ℝ} (hs0 : 0 < s)
    (hs1 : s < 1) :
    s ^ 2 * (1 - N * (s ^ (d - 1) / (Real.sqrt (2 * π * ((d : ℝ) - 1)) * Real.sqrt (1 - s ^ 2))))
      ≤ ∫ ω, C.inf' hC (fun c => ‖U ω - c‖ ^ 2) ∂μ :=
  covering_floor hUm hU hC hN hs0 hs1 (cap_measure_le hd hUm hmarg hs0 hs1)

/-- **Distortion floor, free-scale form.**  The same floor when each codeword may be rescaled by
the best nonnegative factor for each input. -/
theorem distortion_floor_scaled {d : ℕ} (hd : 3 ≤ d) {U : Ω → E} (hUm : Measurable U)
    (hU : ∀ᵐ ω ∂μ, ‖U ω‖ = 1)
    (hmarg : ∀ e : E, ‖e‖ = 1 → μ.map (fun ω => ⟪e, U ω⟫_ℝ) = sphericalCoordLaw d)
    {C : Finset E} (hC : C.Nonempty) {N : ℕ} (hN : C.card ≤ N) {s : ℝ} (hs0 : 0 < s)
    (hs1 : s < 1) :
    s ^ 2 * (1 - N * (s ^ (d - 1) / (Real.sqrt (2 * π * ((d : ℝ) - 1)) * Real.sqrt (1 - s ^ 2))))
      ≤ ∫ ω, C.inf' hC (fun c => ⨅ a : {a : ℝ // 0 ≤ a}, ‖U ω - a.1 • c‖ ^ 2) ∂μ :=
  covering_floor_scaled hUm hU hC hN hs0 hs1 (cap_measure_le hd hUm hmarg hs0 hs1)

/-- **Distortion floor, free-scale form, for a chosen family of scales.** -/
theorem distortion_floor_scales {d : ℕ} (hd : 3 ≤ d) {U : Ω → E} (hUm : Measurable U)
    (hU : ∀ᵐ ω ∂μ, ‖U ω‖ = 1)
    (hmarg : ∀ e : E, ‖e‖ = 1 → μ.map (fun ω => ⟪e, U ω⟫_ℝ) = sphericalCoordLaw d)
    {C : Finset E} (hC : C.Nonempty) {N : ℕ} (hN : C.card ≤ N) {s : ℝ} (hs0 : 0 < s)
    (hs1 : s < 1) {a : Ω → E → ℝ} (ha : ∀ ω c, 0 ≤ a ω c)
    (hint : Integrable (fun ω => C.inf' hC (fun c => ‖U ω - a ω c • c‖ ^ 2)) μ) :
    s ^ 2 * (1 - N * (s ^ (d - 1) / (Real.sqrt (2 * π * ((d : ℝ) - 1)) * Real.sqrt (1 - s ^ 2))))
      ≤ ∫ ω, C.inf' hC (fun c => ‖U ω - a ω c • c‖ ^ 2) ∂μ :=
  covering_floor_scales hUm hU hC hN hs0 hs1 (cap_measure_le hd hUm hmarg hs0 hs1) ha hint

/-- The right-hand side of the rate form of the floor:
`4^(−R) t² (1 − 2^R t^(d−1) / √(2π(d−1)(1 − 4^(−R))))`. -/
noncomputable def Dfloor (R : ℝ) (d : ℕ) (t : ℝ) : ℝ :=
  (4 : ℝ) ^ (-R) * t ^ 2
    * (1 - (2 : ℝ) ^ R * t ^ (d - 1) / Real.sqrt (2 * π * ((d : ℝ) - 1) * (1 - (4 : ℝ) ^ (-R))))

/-- Algebra of the rate form: with `K ≤ 2^(dR)` and `s = 2^(−R) t`, `t ∈ (0,1]`, one has
`0 < s < 1` and `Dfloor R d t ≤ s² (1 − K s^(d−1) / (√(2π(d−1)) √(1 − s²)))`. -/
theorem Dfloor_le_floor {d : ℕ} (hd : 3 ≤ d) {K : ℕ} {R : ℝ} (hR : 0 < R)
    (hN : (K : ℝ) ≤ (2 : ℝ) ^ ((d : ℝ) * R)) {t : ℝ} (ht0 : 0 < t) (ht1 : t ≤ 1) :
    0 < (2 : ℝ) ^ (-R) * t ∧ (2 : ℝ) ^ (-R) * t < 1 ∧
      Dfloor R d t ≤ ((2 : ℝ) ^ (-R) * t) ^ 2 * (1 - K * (((2 : ℝ) ^ (-R) * t) ^ (d - 1)
        / (Real.sqrt (2 * π * ((d : ℝ) - 1)) * Real.sqrt (1 - ((2 : ℝ) ^ (-R) * t) ^ 2)))) := by
  have hdR : (3 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
  have hd1 : (0 : ℝ) < (d : ℝ) - 1 := by linarith
  set r : ℝ := (2 : ℝ) ^ (-R) with hr
  have hr0 : 0 < r := Real.rpow_pos_of_pos two_pos _
  have hr1 : r < 1 := Real.rpow_lt_one_of_one_lt_of_neg one_lt_two (by linarith)
  have hr4 : (4 : ℝ) ^ (-R) = r ^ 2 := by
    rw [hr, ← Real.rpow_natCast, ← Real.rpow_mul two_pos.le,
      show (4 : ℝ) = 2 ^ (2 : ℝ) by norm_num, ← Real.rpow_mul two_pos.le]
    congr 1; push_cast; ring
  have h2R : (2 : ℝ) ^ R = r⁻¹ := by rw [hr, Real.rpow_neg two_pos.le, inv_inv]
  have hNr : (K : ℝ) ≤ (r⁻¹) ^ d := by
    rw [← h2R, ← Real.rpow_natCast, ← Real.rpow_mul two_pos.le, mul_comm]; exact hN
  set s : ℝ := r * t with hs
  have hs0 : 0 < s := mul_pos hr0 ht0
  have hs1 : s < 1 := by nlinarith
  have hsr : s ≤ r := by nlinarith
  have h1r : 0 < 1 - r ^ 2 := by nlinarith
  have h1s : 1 - r ^ 2 ≤ 1 - s ^ 2 := by nlinarith
  have hA : 0 < Real.sqrt (2 * π * ((d : ℝ) - 1)) :=
    Real.sqrt_pos.mpr (mul_pos (by positivity) hd1)
  have hBr : 0 < Real.sqrt (1 - r ^ 2) := Real.sqrt_pos.mpr h1r
  have hBs : Real.sqrt (1 - r ^ 2) ≤ Real.sqrt (1 - s ^ 2) := Real.sqrt_le_sqrt h1s
  have hsplit : Real.sqrt (2 * π * ((d : ℝ) - 1) * (1 - r ^ 2))
      = Real.sqrt (2 * π * ((d : ℝ) - 1)) * Real.sqrt (1 - r ^ 2) :=
    Real.sqrt_mul (mul_pos (by positivity) hd1).le _
  have hspow : s ^ (d - 1) = r ^ (d - 1) * t ^ (d - 1) := by rw [hs, mul_pow]
  have hrd : (r⁻¹) ^ d * r ^ (d - 1) = r⁻¹ := by
    obtain ⟨k, rfl⟩ : ∃ k, d = k + 1 := ⟨d - 1, by omega⟩
    rw [Nat.add_sub_cancel, pow_succ, inv_pow]
    field_simp
  have hkey : (K : ℝ) * (s ^ (d - 1)
        / (Real.sqrt (2 * π * ((d : ℝ) - 1)) * Real.sqrt (1 - s ^ 2)))
      ≤ r⁻¹ * t ^ (d - 1) / Real.sqrt (2 * π * ((d : ℝ) - 1) * (1 - r ^ 2)) := by
    rw [hsplit, hspow]
    have htp : 0 ≤ t ^ (d - 1) := by positivity
    have hrp : 0 ≤ r ^ (d - 1) := by positivity
    calc (K : ℝ) * (r ^ (d - 1) * t ^ (d - 1)
          / (Real.sqrt (2 * π * ((d : ℝ) - 1)) * Real.sqrt (1 - s ^ 2)))
        ≤ (r⁻¹) ^ d * (r ^ (d - 1) * t ^ (d - 1)
          / (Real.sqrt (2 * π * ((d : ℝ) - 1)) * Real.sqrt (1 - r ^ 2))) := by
          refine mul_le_mul hNr ?_ (by positivity) (by positivity)
          exact div_le_div_of_nonneg_left (by positivity) (by positivity)
            (mul_le_mul_of_nonneg_left hBs hA.le)
      _ = r⁻¹ * t ^ (d - 1) / (Real.sqrt (2 * π * ((d : ℝ) - 1)) * Real.sqrt (1 - r ^ 2)) := by
          rw [← mul_div_assoc, ← mul_assoc, hrd]
  refine ⟨hs0, hs1, ?_⟩
  rw [Dfloor, hr4, h2R, show r ^ 2 * t ^ 2 = s ^ 2 by rw [hs]; ring]
  exact mul_le_mul_of_nonneg_left (by linarith) (sq_nonneg s)

/-- **Distortion floor, rate form.**  With `N ≤ 2^(dR)` codewords and `s = 2^(−R) t`,
`t ∈ (0,1]`: `E min_c ‖U − c‖² ≥ 4^(−R) t² (1 − 2^R t^(d−1) / √(2π(d−1)(1 − 4^(−R))))`. -/
theorem distortion_floor_rate {d : ℕ} (hd : 3 ≤ d) {U : Ω → E} (hUm : Measurable U)
    (hU : ∀ᵐ ω ∂μ, ‖U ω‖ = 1)
    (hmarg : ∀ e : E, ‖e‖ = 1 → μ.map (fun ω => ⟪e, U ω⟫_ℝ) = sphericalCoordLaw d)
    {C : Finset E} (hC : C.Nonempty) {R : ℝ} (hR : 0 < R)
    (hN : (C.card : ℝ) ≤ (2 : ℝ) ^ ((d : ℝ) * R)) {t : ℝ} (ht0 : 0 < t) (ht1 : t ≤ 1) :
    Dfloor R d t ≤ ∫ ω, C.inf' hC (fun c => ‖U ω - c‖ ^ 2) ∂μ := by
  obtain ⟨hs0, hs1, hD⟩ := Dfloor_le_floor hd hR hN ht0 ht1
  exact hD.trans (distortion_floor hd hUm hU hmarg hC le_rfl hs0 hs1)

/-- **Distortion floor, rate form, free scale.**  The same floor when each codeword may be
rescaled by the best nonnegative factor for each input:
`E min_c inf_{a ≥ 0} ‖U − a c‖² ≥ Dfloor R d t`. -/
theorem distortion_floor_rate_scaled {d : ℕ} (hd : 3 ≤ d) {U : Ω → E} (hUm : Measurable U)
    (hU : ∀ᵐ ω ∂μ, ‖U ω‖ = 1)
    (hmarg : ∀ e : E, ‖e‖ = 1 → μ.map (fun ω => ⟪e, U ω⟫_ℝ) = sphericalCoordLaw d)
    {C : Finset E} (hC : C.Nonempty) {R : ℝ} (hR : 0 < R)
    (hN : (C.card : ℝ) ≤ (2 : ℝ) ^ ((d : ℝ) * R)) {t : ℝ} (ht0 : 0 < t) (ht1 : t ≤ 1) :
    Dfloor R d t
      ≤ ∫ ω, C.inf' hC (fun c => ⨅ a : {a : ℝ // 0 ≤ a}, ‖U ω - a.1 • c‖ ^ 2) ∂μ := by
  obtain ⟨hs0, hs1, hD⟩ := Dfloor_le_floor hd hR hN ht0 ht1
  exact hD.trans (distortion_floor_scaled hd hUm hU hmarg hC le_rfl hs0 hs1)

/-- **Distortion floor, rate form, for a chosen family of scales.**  For every choice of
nonnegative scales `a ω c` with integrable resulting distortion, the floor holds. -/
theorem distortion_floor_rate_scales {d : ℕ} (hd : 3 ≤ d) {U : Ω → E} (hUm : Measurable U)
    (hU : ∀ᵐ ω ∂μ, ‖U ω‖ = 1)
    (hmarg : ∀ e : E, ‖e‖ = 1 → μ.map (fun ω => ⟪e, U ω⟫_ℝ) = sphericalCoordLaw d)
    {C : Finset E} (hC : C.Nonempty) {R : ℝ} (hR : 0 < R)
    (hN : (C.card : ℝ) ≤ (2 : ℝ) ^ ((d : ℝ) * R)) {t : ℝ} (ht0 : 0 < t) (ht1 : t ≤ 1)
    {a : Ω → E → ℝ} (ha : ∀ ω c, 0 ≤ a ω c)
    (hint : Integrable (fun ω => C.inf' hC (fun c => ‖U ω - a ω c • c‖ ^ 2)) μ) :
    Dfloor R d t ≤ ∫ ω, C.inf' hC (fun c => ‖U ω - a ω c • c‖ ^ 2) ∂μ := by
  obtain ⟨hs0, hs1, hD⟩ := Dfloor_le_floor hd hR hN ht0 ht1
  exact hD.trans (distortion_floor_scales hd hUm hU hmarg hC le_rfl hs0 hs1 ha hint)

end Assembly

/-! ## Step H: numeric instances -/

/-- At rate `R = 2`, `d = 512`, `t = 0.995` the floor is at least `0.0615`. -/
theorem Dfloor_two_512 : (0.0615 : ℝ) ≤ Dfloor 2 512 0.995 := by
  have hπ := Real.pi_gt_d2
  have h4 : (4 : ℝ) ^ (-2 : ℝ) = 1 / 16 := by
    rw [Real.rpow_neg (by norm_num), Real.rpow_two]; norm_num
  have h2 : (2 : ℝ) ^ (2 : ℝ) = 4 := by rw [Real.rpow_two]; norm_num
  have hpow : (0.995 : ℝ) ^ (512 - 1) ≤ 0.078 := by
    have h1 : (0.995 : ℝ) ^ 73 ≤ 0.6936 := by norm_num
    calc (0.995 : ℝ) ^ (512 - 1) = ((0.995 : ℝ) ^ 73) ^ 7 := by rw [← pow_mul]
      _ ≤ (0.6936 : ℝ) ^ 7 := pow_le_pow_left₀ (by positivity) h1 7
      _ ≤ 0.078 := by norm_num
  have hpow0 : (0 : ℝ) ≤ (0.995 : ℝ) ^ (512 - 1) := by positivity
  have hsq : (54.8 : ℝ) ≤ Real.sqrt (2 * π * (((512 : ℕ) : ℝ) - 1) * (1 - 1 / 16)) := by
    have h : (54.8 : ℝ) ^ 2 ≤ 2 * π * (512 - 1) * (1 - 1 / 16) := by nlinarith
    exact Real.le_sqrt_of_sq_le (by push_cast; exact h)
  have hfrac : (4 : ℝ) * (0.995 : ℝ) ^ (512 - 1)
      / Real.sqrt (2 * π * (((512 : ℕ) : ℝ) - 1) * (1 - 1 / 16)) ≤ 4 * 0.078 / 54.8 :=
    div_le_div₀ (by norm_num) (mul_le_mul_of_nonneg_left hpow (by norm_num)) (by norm_num) hsq
  have hnum : (0.0615 : ℝ) ≤ 1 / 16 * 0.995 ^ 2 * (1 - 4 * 0.078 / 54.8) := by norm_num
  rw [Dfloor, h4, h2]
  exact hnum.trans (mul_le_mul_of_nonneg_left (sub_le_sub_left hfrac 1) (by norm_num))

/-- At rate `R = 2`, `d = 256`, `t = 0.99` the floor is at least `0.0607`. -/
theorem Dfloor_two_256 : (0.0607 : ℝ) ≤ Dfloor 2 256 0.99 := by
  have hπ := Real.pi_gt_d2
  have h4 : (4 : ℝ) ^ (-2 : ℝ) = 1 / 16 := by
    rw [Real.rpow_neg (by norm_num), Real.rpow_two]; norm_num
  have h2 : (2 : ℝ) ^ (2 : ℝ) = 4 := by rw [Real.rpow_two]; norm_num
  have hpow : (0.99 : ℝ) ^ (256 - 1) ≤ 0.078 := by
    have h1 : (0.99 : ℝ) ^ 51 ≤ 0.599 := by norm_num
    calc (0.99 : ℝ) ^ (256 - 1) = ((0.99 : ℝ) ^ 51) ^ 5 := by rw [← pow_mul]
      _ ≤ (0.599 : ℝ) ^ 5 := pow_le_pow_left₀ (by positivity) h1 5
      _ ≤ 0.078 := by norm_num
  have hpow0 : (0 : ℝ) ≤ (0.99 : ℝ) ^ (256 - 1) := by positivity
  have hsq : (38.7 : ℝ) ≤ Real.sqrt (2 * π * (((256 : ℕ) : ℝ) - 1) * (1 - 1 / 16)) := by
    have h : (38.7 : ℝ) ^ 2 ≤ 2 * π * (256 - 1) * (1 - 1 / 16) := by nlinarith
    exact Real.le_sqrt_of_sq_le (by push_cast; exact h)
  have hfrac : (4 : ℝ) * (0.99 : ℝ) ^ (256 - 1)
      / Real.sqrt (2 * π * (((256 : ℕ) : ℝ) - 1) * (1 - 1 / 16)) ≤ 4 * 0.078 / 38.7 :=
    div_le_div₀ (by norm_num) (mul_le_mul_of_nonneg_left hpow (by norm_num)) (by norm_num) hsq
  have hnum : (0.0607 : ℝ) ≤ 1 / 16 * 0.99 ^ 2 * (1 - 4 * 0.078 / 38.7) := by norm_num
  rw [Dfloor, h4, h2]
  exact hnum.trans (mul_le_mul_of_nonneg_left (sub_le_sub_left hfrac 1) (by norm_num))

/-- `4^(−4) = 1/256` and `2^4 = 16` as real powers. -/
theorem rpow_four_consts : (4 : ℝ) ^ (-4 : ℝ) = 1 / 256 ∧ (2 : ℝ) ^ (4 : ℝ) = 16 := by
  have h4 : (4 : ℝ) ^ (4 : ℝ) = 256 := by
    rw [show (4 : ℝ) = ((4 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]; norm_num
  have h2 : (2 : ℝ) ^ (4 : ℝ) = 16 := by
    rw [show (4 : ℝ) = ((4 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]; norm_num
  refine ⟨?_, h2⟩
  rw [Real.rpow_neg (by norm_num), h4]; norm_num

/-- At rate `R = 4`, `d = 512`, `t = 0.995` the floor is at least `0.00378`. -/
theorem Dfloor_four_512 : (0.00378 : ℝ) ≤ Dfloor 4 512 0.995 := by
  have hπ := Real.pi_gt_d2
  obtain ⟨h4, h2⟩ := rpow_four_consts
  have hpow : (0.995 : ℝ) ^ (512 - 1) ≤ 0.078 := by
    have h1 : (0.995 : ℝ) ^ 73 ≤ 0.6936 := by norm_num
    calc (0.995 : ℝ) ^ (512 - 1) = ((0.995 : ℝ) ^ 73) ^ 7 := by rw [← pow_mul]
      _ ≤ (0.6936 : ℝ) ^ 7 := pow_le_pow_left₀ (by positivity) h1 7
      _ ≤ 0.078 := by norm_num
  have hsq : (56.5 : ℝ) ≤ Real.sqrt (2 * π * (((512 : ℕ) : ℝ) - 1) * (1 - 1 / 256)) := by
    have h : (56.5 : ℝ) ^ 2 ≤ 2 * π * (512 - 1) * (1 - 1 / 256) := by nlinarith
    exact Real.le_sqrt_of_sq_le (by push_cast; exact h)
  have hfrac : (16 : ℝ) * (0.995 : ℝ) ^ (512 - 1)
      / Real.sqrt (2 * π * (((512 : ℕ) : ℝ) - 1) * (1 - 1 / 256)) ≤ 16 * 0.078 / 56.5 :=
    div_le_div₀ (by norm_num) (mul_le_mul_of_nonneg_left hpow (by norm_num)) (by norm_num) hsq
  have hnum : (0.00378 : ℝ) ≤ 1 / 256 * 0.995 ^ 2 * (1 - 16 * 0.078 / 56.5) := by norm_num
  rw [Dfloor, h4, h2]
  exact hnum.trans (mul_le_mul_of_nonneg_left (sub_le_sub_left hfrac 1) (by norm_num))

/-- At rate `R = 4`, `d = 512`, the better choice `t = 0.992` gives at least `0.00382`. -/
theorem Dfloor_four_512_sharp : (0.00382 : ℝ) ≤ Dfloor 4 512 0.992 := by
  have hπ := Real.pi_gt_d2
  obtain ⟨h4, h2⟩ := rpow_four_consts
  have hpow : (0.992 : ℝ) ^ (512 - 1) ≤ 0.0166 := by
    have h1 : (0.992 : ℝ) ^ 73 ≤ 0.5564 := by norm_num
    calc (0.992 : ℝ) ^ (512 - 1) = ((0.992 : ℝ) ^ 73) ^ 7 := by rw [← pow_mul]
      _ ≤ (0.5564 : ℝ) ^ 7 := pow_le_pow_left₀ (by positivity) h1 7
      _ ≤ 0.0166 := by norm_num
  have hsq : (56.5 : ℝ) ≤ Real.sqrt (2 * π * (((512 : ℕ) : ℝ) - 1) * (1 - 1 / 256)) := by
    have h : (56.5 : ℝ) ^ 2 ≤ 2 * π * (512 - 1) * (1 - 1 / 256) := by nlinarith
    exact Real.le_sqrt_of_sq_le (by push_cast; exact h)
  have hfrac : (16 : ℝ) * (0.992 : ℝ) ^ (512 - 1)
      / Real.sqrt (2 * π * (((512 : ℕ) : ℝ) - 1) * (1 - 1 / 256)) ≤ 16 * 0.0166 / 56.5 :=
    div_le_div₀ (by norm_num) (mul_le_mul_of_nonneg_left hpow (by norm_num)) (by norm_num) hsq
  have hnum : (0.00382 : ℝ) ≤ 1 / 256 * 0.992 ^ 2 * (1 - 16 * 0.0166 / 56.5) := by norm_num
  rw [Dfloor, h4, h2]
  exact hnum.trans (mul_le_mul_of_nonneg_left (sub_le_sub_left hfrac 1) (by norm_num))


/-! ## Step G: the floor for the uniform law on the sphere -/

section Linked

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E]

/-- The ambient uniform law is carried by the unit sphere. -/
theorem ae_norm_eq_one_sphereUniformAmbient :
    ∀ᵐ x ∂(sphereUniformAmbient E), ‖x‖ = 1 := by
  have hset : MeasurableSet {x : E | ‖x‖ = 1} := measurableSet_eq_fun measurable_norm measurable_const
  rw [sphereUniformAmbient, ae_map_iff measurable_subtype_coe.aemeasurable hset]
  exact Filter.Eventually.of_forall fun u => norm_eq_of_mem_sphere u

/-- **Distortion floor for the uniform law on the sphere**, no marginal hypothesis. -/
theorem distortion_floor_sphereUniform (hd : 3 ≤ Module.finrank ℝ E) {C : Finset E}
    (hC : C.Nonempty) {N : ℕ} (hN : C.card ≤ N) {s : ℝ} (hs0 : 0 < s) (hs1 : s < 1) :
    s ^ 2 * (1 - N * (s ^ (Module.finrank ℝ E - 1)
        / (Real.sqrt (2 * π * ((Module.finrank ℝ E : ℝ) - 1)) * Real.sqrt (1 - s ^ 2))))
      ≤ ∫ x, C.inf' hC (fun c => ‖x - c‖ ^ 2) ∂(sphereUniformAmbient E) := by
  haveI : Nontrivial E := Module.nontrivial_of_finrank_pos (R := ℝ) (by omega)
  exact distortion_floor (μ := sphereUniformAmbient E) (U := fun x => x) hd measurable_id
    ae_norm_eq_one_sphereUniformAmbient
    (fun e he => sphereUniformAmbient_map_inner (by omega) he) hC hN hs0 hs1

/-- **Distortion floor for the uniform law on the sphere, rate form**, no marginal hypothesis. -/
theorem distortion_floor_rate_sphereUniform (hd : 3 ≤ Module.finrank ℝ E) {C : Finset E}
    (hC : C.Nonempty) {R : ℝ} (hR : 0 < R)
    (hN : (C.card : ℝ) ≤ (2 : ℝ) ^ ((Module.finrank ℝ E : ℝ) * R)) {t : ℝ} (ht0 : 0 < t)
    (ht1 : t ≤ 1) :
    Dfloor R (Module.finrank ℝ E) t
      ≤ ∫ x, C.inf' hC (fun c => ‖x - c‖ ^ 2) ∂(sphereUniformAmbient E) := by
  haveI : Nontrivial E := Module.nontrivial_of_finrank_pos (R := ℝ) (by omega)
  exact distortion_floor_rate (μ := sphereUniformAmbient E) (U := fun x => x) hd measurable_id
    ae_norm_eq_one_sphereUniformAmbient
    (fun e he => sphereUniformAmbient_map_inner (by omega) he) hC hR hN ht0 ht1

/-- **Distortion floor for the uniform law on the sphere, rate form, free scale**: the same floor
when each codeword may be rescaled by the best nonnegative factor for each input. -/
theorem distortion_floor_rate_sphereUniform_scaled (hd : 3 ≤ Module.finrank ℝ E) {C : Finset E}
    (hC : C.Nonempty) {R : ℝ} (hR : 0 < R)
    (hN : (C.card : ℝ) ≤ (2 : ℝ) ^ ((Module.finrank ℝ E : ℝ) * R)) {t : ℝ} (ht0 : 0 < t)
    (ht1 : t ≤ 1) :
    Dfloor R (Module.finrank ℝ E) t
      ≤ ∫ x, C.inf' hC (fun c => ⨅ a : {a : ℝ // 0 ≤ a}, ‖x - a.1 • c‖ ^ 2)
          ∂(sphereUniformAmbient E) := by
  haveI : Nontrivial E := Module.nontrivial_of_finrank_pos (R := ℝ) (by omega)
  exact distortion_floor_rate_scaled (μ := sphereUniformAmbient E) (U := fun x => x) hd
    measurable_id ae_norm_eq_one_sphereUniformAmbient
    (fun e he => sphereUniformAmbient_map_inner (by omega) he) hC hR hN ht0 ht1

end Linked

end SphereFloor

