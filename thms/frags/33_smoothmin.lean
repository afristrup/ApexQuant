/-! ## §33  A smooth cutoff and the smooth minimum

Two approximations that reduce `f_C = dist(·/‖·‖, C)²` to the setting of §32.  The cutoff
`χ_δ(z) = τ(‖z‖²/δ² − 1)`, with `τ` Mathlib's smooth transition, vanishes on the ball of
radius `δ`, equals `1` outside radius `√2 δ`, and has derivative at most `C/δ`
(`exists_bound_fderiv_cutoff`).  The smooth minimum `f_ε = −ε log ∑_c exp(−h_c/ε)` is within
`ε log |C|` of `f_C` (`abs_smoothMin_sub_fC_le`), is `C¹` off the origin with derivative the
softmax combination of the branch derivatives (`hasFDerivAt_smoothMin`), and satisfies the
sharp gradient bound `‖f_ε'(z)‖² ≤ 4 (f_C(z) + |C| ε)/‖z‖²` (`norm_fderiv_smoothMin_sq_le`),
by Cauchy–Schwarz for the convex combination and the estimate `w_c (h_c − f_C) ≤ ε` on the
softmax weights. -/



section Cutoff

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A smooth cutoff vanishing on the ball of radius `δ` and equal to `1` outside radius `√2 δ`. -/
noncomputable def cutoff (δ : ℝ) (z : E) : ℝ := Real.smoothTransition (‖z‖ ^ 2 / δ ^ 2 - 1)

theorem cutoff_nonneg (δ : ℝ) (z : E) : 0 ≤ cutoff δ z := Real.smoothTransition.nonneg _

theorem cutoff_le_one (δ : ℝ) (z : E) : cutoff δ z ≤ 1 := Real.smoothTransition.le_one _

theorem cutoff_eq_zero {δ : ℝ} (hδ : 0 < δ) {z : E} (hz : ‖z‖ ≤ δ) : cutoff δ z = 0 := by
  unfold cutoff
  apply Real.smoothTransition.zero_of_nonpos
  rw [sub_nonpos, div_le_one (by positivity)]
  exact pow_le_pow_left₀ (norm_nonneg _) hz 2

theorem cutoff_eq_one {δ : ℝ} (hδ : 0 < δ) {z : E} (hz : 2 * δ ^ 2 ≤ ‖z‖ ^ 2) : cutoff δ z = 1 := by
  unfold cutoff
  apply Real.smoothTransition.one_of_one_le
  rw [le_sub_iff_add_le, le_div_iff₀ (by positivity)]
  linarith

theorem contDiff_cutoff {δ : ℝ} (hδ : 0 < δ) : ContDiff ℝ 1 (cutoff (E := E) δ) := by
  unfold cutoff
  exact Real.smoothTransition.contDiff.comp ((contDiff_norm_sq ℝ).div_const _ |>.sub contDiff_const)

/-- The derivative of the transition function is bounded. -/
theorem exists_bound_deriv_smoothTransition :
    ∃ C₀ : ℝ, 0 ≤ C₀ ∧ ∀ s, |deriv Real.smoothTransition s| ≤ C₀ := by
  have hcont : Continuous (deriv Real.smoothTransition) :=
    (Real.smoothTransition.contDiff (n := 1)).continuous_deriv le_rfl
  obtain ⟨C₀, hC₀⟩ := (isCompact_Icc (a := (0 : ℝ)) (b := 1)).exists_bound_of_continuousOn
    hcont.continuousOn
  refine ⟨max C₀ 0, le_max_right _ _, fun s => ?_⟩
  rcases le_or_gt s 0 with hs | hs
  · rcases eq_or_lt_of_le hs with rfl | hs'
    · exact (Real.norm_eq_abs _ ▸ hC₀ 0 ⟨le_rfl, zero_le_one⟩).trans (le_max_left _ _)
    · have : deriv Real.smoothTransition s = 0 := by
        have hev : Real.smoothTransition =ᶠ[𝓝 s] fun _ => 0 := by
          filter_upwards [Iio_mem_nhds hs'] with t ht
          exact Real.smoothTransition.zero_of_nonpos (le_of_lt ht)
        rw [hev.deriv_eq, deriv_const]
      rw [this, abs_zero]; exact le_max_right _ _
  · rcases le_or_gt s 1 with hs1 | hs1
    · exact (Real.norm_eq_abs _ ▸ hC₀ s ⟨hs.le, hs1⟩).trans (le_max_left _ _)
    · have : deriv Real.smoothTransition s = 0 := by
        have hev : Real.smoothTransition =ᶠ[𝓝 s] fun _ => 1 := by
          filter_upwards [Ioi_mem_nhds hs1] with t ht
          exact Real.smoothTransition.one_of_one_le (le_of_lt ht)
        rw [hev.deriv_eq, deriv_const]
      rw [this, abs_zero]; exact le_max_right _ _

theorem hasFDerivAt_cutoff {δ : ℝ} (hδ : 0 < δ) (z : E) :
    HasFDerivAt (cutoff δ) (deriv Real.smoothTransition (‖z‖ ^ 2 / δ ^ 2 - 1) •
      (δ ^ 2)⁻¹ • (2 : ℕ) • innerSL ℝ z) z := by
  have h1' := ((hasStrictFDerivAt_norm_sq z).hasFDerivAt.const_smul (δ ^ 2)⁻¹).sub_const 1
  have h1 : HasFDerivAt (fun z : E => ‖z‖ ^ 2 / δ ^ 2 - 1) ((δ ^ 2)⁻¹ • (2 : ℕ) • innerSL ℝ z) z :=
    h1'.congr_of_eventuallyEq (Filter.Eventually.of_forall fun z => by
      simp only [Pi.smul_apply, smul_eq_mul, div_eq_inv_mul])
  have h2 : HasDerivAt Real.smoothTransition (deriv Real.smoothTransition (‖z‖ ^ 2 / δ ^ 2 - 1))
      (‖z‖ ^ 2 / δ ^ 2 - 1) :=
    ((Real.smoothTransition.contDiff (n := 1)).differentiable one_ne_zero _).hasDerivAt
  exact h2.comp_hasFDerivAt z h1

theorem norm_fderiv_cutoff_le {δ : ℝ} (hδ : 0 < δ) (z : E) :
    ‖fderiv ℝ (cutoff δ) z‖
      ≤ |deriv Real.smoothTransition (‖z‖ ^ 2 / δ ^ 2 - 1)| * (2 * ‖z‖ / δ ^ 2) := by
  rw [(hasFDerivAt_cutoff hδ z).fderiv, norm_smul, Real.norm_eq_abs]
  refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity : (0 : ℝ) < (δ ^ 2)⁻¹)]
  calc (δ ^ 2)⁻¹ * ‖(2 : ℕ) • innerSL ℝ z‖ ≤ (δ ^ 2)⁻¹ * ((2 : ℕ) * ‖innerSL ℝ z‖) :=
        mul_le_mul_of_nonneg_left norm_nsmul_le (by positivity)
    _ = 2 * ‖z‖ / δ ^ 2 := by rw [innerSL_apply_norm]; push_cast; ring

theorem fderiv_cutoff_eq_zero {δ : ℝ} (hδ : 0 < δ) {z : E} (hz : 2 * δ ^ 2 < ‖z‖ ^ 2) :
    fderiv ℝ (cutoff δ) z = 0 := by
  have hs : 1 < ‖z‖ ^ 2 / δ ^ 2 - 1 := by
    rw [lt_sub_iff_add_lt, lt_div_iff₀ (by positivity)]; linarith
  have hev : Real.smoothTransition =ᶠ[𝓝 (‖z‖ ^ 2 / δ ^ 2 - 1)] fun _ => 1 := by
    filter_upwards [Ioi_mem_nhds hs] with t ht
    exact Real.smoothTransition.one_of_one_le (le_of_lt ht)
  rw [(hasFDerivAt_cutoff hδ z).fderiv, hev.deriv_eq, deriv_const, zero_smul]

/-- `‖(cutoff δ)'(z)‖ ≤ C/δ` uniformly, and it vanishes outside the ball of radius `√2 δ`. -/
theorem exists_bound_fderiv_cutoff :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ δ : ℝ, 0 < δ → ∀ z : E, ‖fderiv ℝ (cutoff δ) z‖ ≤ C / δ := by
  obtain ⟨C₀, hC₀0, hC₀⟩ := exists_bound_deriv_smoothTransition
  refine ⟨2 * Real.sqrt 2 * C₀, by positivity, fun δ hδ z => ?_⟩
  rcases le_or_gt (‖z‖ ^ 2) (2 * δ ^ 2) with hz | hz
  · have hnz : ‖z‖ ≤ Real.sqrt 2 * δ := by
      rw [← Real.sqrt_sq (norm_nonneg z), ← Real.sqrt_sq hδ.le, ← Real.sqrt_mul zero_le_two]
      exact Real.sqrt_le_sqrt hz
    calc ‖fderiv ℝ (cutoff δ) z‖
        ≤ |deriv Real.smoothTransition (‖z‖ ^ 2 / δ ^ 2 - 1)| * (2 * ‖z‖ / δ ^ 2) :=
          norm_fderiv_cutoff_le hδ z
      _ ≤ C₀ * (2 * (Real.sqrt 2 * δ) / δ ^ 2) := by gcongr; exact hC₀ _
      _ = 2 * Real.sqrt 2 * C₀ / δ := by first | (field_simp; ring) | field_simp
  · rw [fderiv_cutoff_eq_zero hδ hz, norm_zero]
    positivity

end Cutoff

section SmoothMin

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- The smooth minimum `f_ε(z) = −ε log ∑_c exp(−h_c(z)/ε)`. -/
noncomputable def smoothMin (C : Finset E) (ε : ℝ) (z : E) : ℝ :=
  -ε * Real.log (∑ c ∈ C, Real.exp (-hc c z / ε))

/-- The softmax weights. -/
noncomputable def softWeight (C : Finset E) (ε : ℝ) (c z : E) : ℝ :=
  Real.exp (-hc c z / ε) / ∑ c' ∈ C, Real.exp (-hc c' z / ε)

theorem sum_exp_pos (C : Finset E) (hC : C.Nonempty) (ε : ℝ) (z : E) :
    0 < ∑ c ∈ C, Real.exp (-hc c z / ε) :=
  Finset.sum_pos (fun c _ => Real.exp_pos _) hC

theorem softWeight_nonneg (C : Finset E) (ε : ℝ) (c z : E) : 0 ≤ softWeight C ε c z :=
  div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun c _ => (Real.exp_pos _).le)

theorem sum_softWeight (C : Finset E) (hC : C.Nonempty) (ε : ℝ) (z : E) :
    ∑ c ∈ C, softWeight C ε c z = 1 := by
  unfold softWeight
  rw [← Finset.sum_div, div_self (sum_exp_pos C hC ε z).ne']

/-- `f_C − ε log |C| ≤ f_ε ≤ f_C`. -/
theorem smoothMin_le_fC (C : Finset E) (hC : C.Nonempty) {ε : ℝ} (hε : 0 < ε) (z : E) :
    smoothMin C ε z ≤ fC C hC z := by
  unfold smoothMin
  obtain ⟨c₀, hc₀, hmin⟩ := Finset.exists_mem_eq_inf' hC (fun c => hc c z)
  have hfC : fC C hC z = hc c₀ z := hmin
  rw [hfC]
  have h1 : Real.exp (-hc c₀ z / ε) ≤ ∑ c ∈ C, Real.exp (-hc c z / ε) :=
    Finset.single_le_sum (f := fun c => Real.exp (-hc c z / ε)) (fun c _ => (Real.exp_pos _).le) hc₀
  have h2 : -hc c₀ z / ε ≤ Real.log (∑ c ∈ C, Real.exp (-hc c z / ε)) := by
    rw [← Real.log_exp (-hc c₀ z / ε)]
    exact Real.log_le_log (Real.exp_pos _) h1
  have h3 : -ε * Real.log (∑ c ∈ C, Real.exp (-hc c z / ε)) ≤ -ε * (-hc c₀ z / ε) :=
    mul_le_mul_of_nonpos_left h2 (by linarith)
  have h4 : -ε * (-hc c₀ z / ε) = hc c₀ z := by field_simp
  linarith

theorem fC_le_smoothMin_add (C : Finset E) (hC : C.Nonempty) {ε : ℝ} (hε : 0 < ε) (z : E) :
    fC C hC z ≤ smoothMin C ε z + ε * Real.log C.card := by
  unfold smoothMin
  have hcard : (0 : ℝ) < C.card := by exact_mod_cast hC.card_pos
  have h1 : ∑ c ∈ C, Real.exp (-hc c z / ε) ≤ C.card * Real.exp (-fC C hC z / ε) := by
    calc ∑ c ∈ C, Real.exp (-hc c z / ε) ≤ ∑ _c ∈ C, Real.exp (-fC C hC z / ε) := by
          refine Finset.sum_le_sum fun c hc' => Real.exp_le_exp.mpr ?_
          have : fC C hC z ≤ hc c z := Finset.inf'_le _ hc'
          exact div_le_div_of_nonneg_right (by linarith) hε.le
      _ = C.card * Real.exp (-fC C hC z / ε) := by rw [Finset.sum_const, nsmul_eq_mul]
  have h2 : Real.log (∑ c ∈ C, Real.exp (-hc c z / ε)) ≤ Real.log C.card + -fC C hC z / ε := by
    rw [← Real.log_exp (-fC C hC z / ε), ← Real.log_mul hcard.ne' (Real.exp_pos _).ne']
    exact Real.log_le_log (sum_exp_pos C hC ε z) h1
  have h3 : -ε * (Real.log C.card + -fC C hC z / ε)
      ≤ -ε * Real.log (∑ c ∈ C, Real.exp (-hc c z / ε)) :=
    mul_le_mul_of_nonpos_left h2 (by linarith)
  have h4 : -ε * (Real.log C.card + -fC C hC z / ε) = -ε * Real.log C.card + fC C hC z := by
    field_simp; ring
  linarith

theorem abs_smoothMin_sub_fC_le (C : Finset E) (hC : C.Nonempty) {ε : ℝ} (hε : 0 < ε) (z : E) :
    |smoothMin C ε z - fC C hC z| ≤ ε * Real.log C.card := by
  have hlog : 0 ≤ Real.log C.card := Real.log_nonneg (by exact_mod_cast hC.card_pos)
  rw [abs_le]
  constructor
  · linarith [fC_le_smoothMin_add C hC hε z]
  · linarith [smoothMin_le_fC C hC hε z, mul_nonneg hε.le hlog]

/-- The derivative of the smooth minimum off the origin: the softmax combination of the
branch derivatives. -/
theorem hasFDerivAt_smoothMin (C : Finset E) (hC : C.Nonempty) {ε : ℝ} (hε : 0 < ε) {z : E}
    (hz : z ≠ 0) :
    HasFDerivAt (smoothMin C ε) (∑ c ∈ C, softWeight C ε c z • fderiv ℝ (hc c) z) z := by
  have hsum : HasFDerivAt (fun z => ∑ c ∈ C, Real.exp (-hc c z / ε))
      (∑ c ∈ C, (Real.exp (-hc c z / ε) * (-1 / ε)) • fderiv ℝ (hc c) z) z := by
    have hsum0 := HasFDerivAt.sum (x := z) (u := C) (A := fun c z => Real.exp (-hc c z / ε))
      (A' := fun c => (Real.exp (-hc c z / ε) * (-1 / ε)) • fderiv ℝ (hc c) z) ?_
    · exact hsum0.congr_of_eventuallyEq (Filter.Eventually.of_forall fun z =>
        (Finset.sum_apply z C (fun c z => Real.exp (-hc c z / ε))).symm)
    intro c _
    have h0 : HasFDerivAt (fun z => -hc c z / ε) ((-1 / ε) • fderiv ℝ (hc c) z) z := by
      have h := ((differentiableAt_hc c hz).hasFDerivAt.neg).const_smul ε⁻¹
      have h' : HasFDerivAt (fun z => -hc c z / ε) (ε⁻¹ • -fderiv ℝ (hc c) z) z :=
        h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun z => by
          simp only [Pi.smul_apply, Pi.neg_apply, smul_eq_mul, div_eq_inv_mul])
      exact h'.congr_fderiv (by rw [smul_neg, neg_div, one_div, neg_smul])
    exact h0.exp.congr_fderiv (smul_smul _ _ _)
  have hlog := hsum.log (sum_exp_pos C hC ε z).ne'
  have h := hlog.const_smul (-ε)
  have e : (-ε) • (∑ c ∈ C, Real.exp (-hc c z / ε))⁻¹ •
      ∑ c ∈ C, (Real.exp (-hc c z / ε) * (-1 / ε)) • fderiv ℝ (hc c) z
      = ∑ c ∈ C, softWeight C ε c z • fderiv ℝ (hc c) z := by
    rw [Finset.smul_sum, Finset.smul_sum]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [smul_smul, smul_smul]
    unfold softWeight
    congr 1
    field_simp
  rw [e] at h
  exact h

theorem norm_fderiv_smoothMin_le (C : Finset E) (hC : C.Nonempty) {ε : ℝ} (hε : 0 < ε) {z : E}
    (hz : z ≠ 0) :
    ‖fderiv ℝ (smoothMin C ε) z‖ ≤ ∑ c ∈ C, softWeight C ε c z * (2 * ‖dir z - c‖ / ‖z‖) := by
  rw [(hasFDerivAt_smoothMin C hC hε hz).fderiv]
  refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun c _ => ?_)
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (softWeight_nonneg C ε c z)]
  exact mul_le_mul_of_nonneg_left (norm_fderiv_hc_le_sharp c hz) (softWeight_nonneg C ε c z)

/-- A softmax weight times the excess of its branch over the minimum is at most `ε`. -/
theorem softWeight_mul_sub_le (C : Finset E) (hC : C.Nonempty) {ε : ℝ} (hε : 0 < ε) (c z : E)
    (hc' : c ∈ C) : softWeight C ε c z * (hc c z - fC C hC z) ≤ ε := by
  set t := (hc c z - fC C hC z) / ε with ht
  have ht0 : 0 ≤ t := div_nonneg (sub_nonneg.mpr (Finset.inf'_le _ hc')) hε.le
  have hw : softWeight C ε c z ≤ Real.exp (-t) := by
    unfold softWeight
    obtain ⟨c₀, hc₀, hmin⟩ := Finset.exists_mem_eq_inf' hC (fun c => hc c z)
    have hfC : fC C hC z = hc c₀ z := hmin
    have h1 : Real.exp (-hc c₀ z / ε) ≤ ∑ c ∈ C, Real.exp (-hc c z / ε) :=
      Finset.single_le_sum (f := fun c => Real.exp (-hc c z / ε)) (fun c _ => (Real.exp_pos _).le)
        hc₀
    rw [div_le_iff₀ (sum_exp_pos C hC ε z)]
    calc Real.exp (-hc c z / ε) = Real.exp (-t) * Real.exp (-hc c₀ z / ε) := by
          rw [← Real.exp_add, ht, hfC]; congr 1; field_simp; ring
      _ ≤ Real.exp (-t) * ∑ c ∈ C, Real.exp (-hc c z / ε) :=
          mul_le_mul_of_nonneg_left h1 (Real.exp_pos _).le
  have hte : t * Real.exp (-t) ≤ 1 := by
    have := Real.add_one_le_exp t
    rw [Real.exp_neg]
    rw [mul_inv_le_iff₀ (Real.exp_pos t), one_mul]
    linarith
  have hsub : hc c z - fC C hC z = t * ε := by rw [ht]; field_simp
  rw [hsub]
  calc softWeight C ε c z * (t * ε) ≤ Real.exp (-t) * (t * ε) :=
        mul_le_mul_of_nonneg_right hw (by positivity)
    _ = (t * Real.exp (-t)) * ε := by ring
    _ ≤ 1 * ε := mul_le_mul_of_nonneg_right hte hε.le
    _ = ε := one_mul ε

/-- **The gradient bound for the smooth minimum**:
`‖f_ε'(z)‖² ≤ 4 (f_C(z) + |C| ε)/‖z‖²`. -/
theorem norm_fderiv_smoothMin_sq_le (C : Finset E) (hC : C.Nonempty) {ε : ℝ} (hε : 0 < ε) {z : E}
    (hz : z ≠ 0) :
    ‖fderiv ℝ (smoothMin C ε) z‖ ^ 2 ≤ 4 * (fC C hC z + C.card * ε) / ‖z‖ ^ 2 := by
  have hz' : 0 < ‖z‖ := norm_pos_iff.mpr hz
  have h1 := norm_fderiv_smoothMin_le C hC hε hz
  have hw0 := softWeight_nonneg C ε
  -- Cauchy–Schwarz for the convex combination
  have h2 : (∑ c ∈ C, softWeight C ε c z * (2 * ‖dir z - c‖ / ‖z‖)) ^ 2
      ≤ ∑ c ∈ C, softWeight C ε c z * (2 * ‖dir z - c‖ / ‖z‖) ^ 2 := by
    have hcs := Finset.sum_mul_sq_le_sq_mul_sq C (fun c => Real.sqrt (softWeight C ε c z))
      (fun c => Real.sqrt (softWeight C ε c z) * (2 * ‖dir z - c‖ / ‖z‖))
    have e1 : ∀ c ∈ C, Real.sqrt (softWeight C ε c z) *
        (Real.sqrt (softWeight C ε c z) * (2 * ‖dir z - c‖ / ‖z‖))
        = softWeight C ε c z * (2 * ‖dir z - c‖ / ‖z‖) := fun c _ => by
      rw [← mul_assoc, Real.mul_self_sqrt (hw0 c z)]
    have e2 : ∀ c ∈ C, Real.sqrt (softWeight C ε c z) ^ 2 = softWeight C ε c z := fun c _ =>
      Real.sq_sqrt (hw0 c z)
    have e3 : ∀ c ∈ C, (Real.sqrt (softWeight C ε c z) * (2 * ‖dir z - c‖ / ‖z‖)) ^ 2
        = softWeight C ε c z * (2 * ‖dir z - c‖ / ‖z‖) ^ 2 := fun c _ => by
      rw [mul_pow, Real.sq_sqrt (hw0 c z)]
    rw [Finset.sum_congr rfl e1, Finset.sum_congr rfl e2, Finset.sum_congr rfl e3,
      sum_softWeight C hC, one_mul] at hcs
    exact hcs
  -- each squared slope is `4 h_c(z)/‖z‖²`
  have h3 : ∑ c ∈ C, softWeight C ε c z * (2 * ‖dir z - c‖ / ‖z‖) ^ 2
      = (4 / ‖z‖ ^ 2) * ∑ c ∈ C, softWeight C ε c z * hc c z := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun c _ => ?_
    unfold hc
    field_simp
    ring
  have h4 : ∑ c ∈ C, softWeight C ε c z * hc c z ≤ fC C hC z + C.card * ε := by
    have : ∀ c ∈ C, softWeight C ε c z * hc c z ≤ softWeight C ε c z * fC C hC z + ε := fun c hc' =>
      by linarith [softWeight_mul_sub_le C hC hε c z hc']
    calc ∑ c ∈ C, softWeight C ε c z * hc c z
        ≤ ∑ c ∈ C, (softWeight C ε c z * fC C hC z + ε) := Finset.sum_le_sum this
      _ = fC C hC z + C.card * ε := by
          rw [Finset.sum_add_distrib, ← Finset.sum_mul, sum_softWeight C hC, one_mul,
            Finset.sum_const, nsmul_eq_mul]
  calc ‖fderiv ℝ (smoothMin C ε) z‖ ^ 2
      ≤ (∑ c ∈ C, softWeight C ε c z * (2 * ‖dir z - c‖ / ‖z‖)) ^ 2 :=
        pow_le_pow_left₀ (norm_nonneg _) h1 2
    _ ≤ (4 / ‖z‖ ^ 2) * ∑ c ∈ C, softWeight C ε c z * hc c z := h2.trans (le_of_eq h3)
    _ ≤ (4 / ‖z‖ ^ 2) * (fC C hC z + C.card * ε) :=
        mul_le_mul_of_nonneg_left h4 (by positivity)
    _ = 4 * (fC C hC z + C.card * ε) / ‖z‖ ^ 2 := by ring

end SmoothMin
