/-! ## §30  The Poincaré inequality for a finite codebook, proved

Integrating the path inequality of §29 over `γ ⊗ γ`, exchanging the order of integration, and
using that the rotation preserves `γ ⊗ γ` for every `θ` (§27) gives
`E(f_C(X) − f_C(Y))² ≤ (π/2)² E S_C(X, Y)`.  Almost surely one codeword is active, and its
derivative is bounded by `4 · dist(Z/‖Z‖, C)/‖Z‖` (`norm_fderiv_hc_le`), so
`E S_C ≤ 16 E[f_C(Z)/‖Z‖²] = 16 E_σ f_C/(d − 2)` by §26.  Hence
`Var_γ f_C ≤ 2π²/(d − 2) · E f_C` (`variance_fC_le`), and, transporting along the direction
map, `Var_σ ‖u − q u‖² ≤ 2π²/(d − 2) · ρ²` for any nearest-codeword stage with a finite
codebook (`variance_stageErr_le_finite`).  This replaces the hypothesis `GaussPoincare` of §26:
`meas_not_decayWithin_le_finite` is the concentration remark for scalar, `E8`, and trellis
codebooks, at rate `2π² P/((d − 2) ε² ρ²)`, with **no unproved hypothesis**. -/

section Assembly

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- The path inequality in `lintegral` form. -/
theorem ofReal_sq_sub_fC_le (C : Finset E) (hC : C.Nonempty) {x y : E} (hg : Good x y) :
    ENNReal.ofReal ((fC C hC y - fC C hC x) ^ 2)
      ≤ ENNReal.ofReal (Real.pi / 2) * ∫⁻ θ in Set.Ioc (0 : ℝ) (Real.pi / 2),
          ENNReal.ofReal (SC C hC (pathX x y θ) (pathY x y θ)) := by
  obtain ⟨hint, h⟩ := sq_sub_fC_le C hC hg
  have hpi : (0 : ℝ) ≤ Real.pi / 2 := by positivity
  calc ENNReal.ofReal ((fC C hC y - fC C hC x) ^ 2)
      ≤ ENNReal.ofReal ((Real.pi / 2) *
          ∫ θ in (0 : ℝ)..(Real.pi / 2), SC C hC (pathX x y θ) (pathY x y θ)) :=
        ENNReal.ofReal_le_ofReal h
    _ = ENNReal.ofReal (Real.pi / 2) * ENNReal.ofReal
          (∫ θ in (0 : ℝ)..(Real.pi / 2), SC C hC (pathX x y θ) (pathY x y θ)) :=
        ENNReal.ofReal_mul hpi
    _ = _ := by
        rw [intervalIntegral.integral_of_le hpi, ofReal_integral_eq_lintegral_ofReal
          ((intervalIntegrable_iff_integrableOn_Ioc_of_le hpi).mp hint)
          (Filter.Eventually.of_forall fun θ => SC_nonneg _ _ _ _)]

theorem dir_dir (x : E) : dir (dir x) = dir x := by
  rcases eq_or_ne x 0 with rfl | hx
  · simp [dir]
  have h : ‖dir x‖ = 1 := by
    rw [dir, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr hx)]
  show ‖dir x‖⁻¹ • dir x = dir x
  rw [h, inv_one, one_smul]

theorem fC_dir (C : Finset E) (hC : C.Nonempty) (x : E) : fC C hC (dir x) = fC C hC x := by
  unfold fC hc; rw [dir_dir]

/-- Almost surely at most one codeword is active. -/
theorem ae_unique_active (hd : 2 ≤ Module.finrank ℝ E) (C : Finset E) (hC : C.Nonempty) :
    ∀ᵐ z ∂(gaussianE E), ∀ c ∈ (C : Set E), ∀ c' ∈ (C : Set E),
      hc c z = fC C hC z → hc c' z = fC C hC z → c = c' := by
  rw [ae_ball_iff C.countable_toSet]
  intro c _
  rw [ae_ball_iff C.countable_toSet]
  intro c' _
  by_cases h : c = c'
  · exact Filter.Eventually.of_forall fun _ _ _ => h
  · have hnull := gaussianE_tie_null hd h
    rw [ae_iff]
    refine measure_mono_null (fun z hz => ?_) hnull
    simp only [Set.mem_setOf_eq, _root_.not_imp] at hz
    simp only [Set.mem_setOf_eq]
    rw [hz.1, hz.2.1]

/-- The sum of the active indicators is at most one, almost surely. -/
theorem ae_sum_active_le_one (hd : 2 ≤ Module.finrank ℝ E) (C : Finset E) (hC : C.Nonempty) :
    ∀ᵐ z ∂(gaussianE E),
      (∑ c ∈ C, ({x : E | hc c x = fC C hC x}).indicator (fun _ => (1 : ℝ≥0∞)) z) ≤ 1 := by
  filter_upwards [ae_unique_active hd C hC] with z hz
  classical
  have : ∀ c ∈ C, ({x : E | hc c x = fC C hC x}).indicator (fun _ => (1 : ℝ≥0∞)) z
      = if hc c z = fC C hC z then 1 else 0 := fun c _ => by
    by_cases h : hc c z = fC C hC z <;> simp [Set.indicator_apply, h]
  rw [Finset.sum_congr rfl this, Finset.sum_boole]
  have hcard : (C.filter fun c => hc c z = fC C hC z).card ≤ 1 := by
    refine Finset.card_le_one.mpr fun a ha b hb => ?_
    rw [Finset.mem_filter] at ha hb
    exact hz a ha.1 b hb.1 ha.2 hb.2
  exact_mod_cast hcard

/-- **The Gaussian integral of the active-derivative sum** is bounded by
`16 E[f(Z)/‖Z‖²]`. -/
theorem lintegral_SC_le (hd : 2 ≤ Module.finrank ℝ E) (C : Finset E) (hC : C.Nonempty) :
    ∫⁻ p, ENNReal.ofReal (SC C hC p.1 p.2) ∂((gaussianE E).prod (gaussianE E))
      ≤ 16 * ∫⁻ x, ENNReal.ofReal (fC C hC x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹) ∂(gaussianE E) := by
  set A : E → Set E := fun c => {x : E | hc c x = fC C hC x} with hA
  have hAmeas : ∀ c, MeasurableSet (A c) := fun c =>
    measurableSet_eq_fun (measurable_hc c) (measurable_fC C hC)
  -- pointwise: `ofReal (SC p) = ∑_c 1_{A c}(p.1) · ofReal ((h_c'(p.1) p.2)²)`
  have hpt : ∀ p : E × E, ENNReal.ofReal (SC C hC p.1 p.2)
      = ∑ c ∈ C, (A c).indicator (fun _ => (1 : ℝ≥0∞)) p.1 *
          ENNReal.ofReal (fderiv ℝ (hc c) p.1 p.2 ^ 2) := by
    intro p
    unfold SC
    rw [ENNReal.ofReal_sum_of_nonneg fun c _ => Set.indicator_nonneg (fun _ _ => sq_nonneg _) _]
    refine Finset.sum_congr rfl fun c _ => ?_
    by_cases h : p.1 ∈ A c
    · have h' : (p.1, p.2) ∈ active C hC c := h
      rw [Set.indicator_of_mem h, Set.indicator_of_mem h', one_mul]
    · have h' : (p.1, p.2) ∉ active C hC c := h
      rw [Set.indicator_of_notMem h, Set.indicator_of_notMem h', zero_mul, ENNReal.ofReal_zero]
  have hmeasc : ∀ c, Measurable fun p : E × E =>
      (A c).indicator (fun _ => (1 : ℝ≥0∞)) p.1 * ENNReal.ofReal (fderiv ℝ (hc c) p.1 p.2 ^ 2) :=
    fun c => ((measurable_const.indicator (hAmeas c)).comp measurable_fst).mul
      (ENNReal.measurable_ofReal.comp (measurable_fderiv_hc_apply c))
  simp_rw [hpt]
  rw [lintegral_finsetSum _ fun c _ => hmeasc c]
  -- each term: integrate out `y`
  have hterm : ∀ c ∈ C, ∫⁻ p, (A c).indicator (fun _ => (1 : ℝ≥0∞)) p.1 *
        ENNReal.ofReal (fderiv ℝ (hc c) p.1 p.2 ^ 2) ∂((gaussianE E).prod (gaussianE E))
      = ∫⁻ x, (A c).indicator (fun _ => (1 : ℝ≥0∞)) x *
          ENNReal.ofReal (‖fderiv ℝ (hc c) x‖ ^ 2) ∂(gaussianE E) := by
    intro c _
    rw [lintegral_prod _ (hmeasc c).aemeasurable]
    refine lintegral_congr fun x => ?_
    show ∫⁻ y, (A c).indicator (fun _ => (1 : ℝ≥0∞)) x *
        ENNReal.ofReal (fderiv ℝ (hc c) x y ^ 2) ∂(gaussianE E) = _
    rw [lintegral_const_mul _ (((fderiv ℝ (hc c) x).continuous.measurable.pow_const 2).ennreal_ofReal),
      lintegral_clm_sq_gaussianE hd]
  rw [Finset.sum_congr rfl hterm]
  -- bound each term on the active set, using `x ≠ 0` a.e.
  have hbound : ∀ c ∈ C, ∫⁻ x, (A c).indicator (fun _ => (1 : ℝ≥0∞)) x *
        ENNReal.ofReal (‖fderiv ℝ (hc c) x‖ ^ 2) ∂(gaussianE E)
      ≤ ∫⁻ x, (A c).indicator (fun _ => (1 : ℝ≥0∞)) x *
          (16 * (ENNReal.ofReal (fC C hC x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹))) ∂(gaussianE E) := by
    intro c _
    refine lintegral_mono_ae ?_
    filter_upwards [gaussianE_ae_ne_zero (E := E)] with x hx
    by_cases hxA : x ∈ A c
    · rw [Set.indicator_of_mem hxA, one_mul, one_mul]
      have h1 := norm_fderiv_hc_le c hx
      have hxA' : hc c x = fC C hC x := hxA
      have hn : 0 < ‖x‖ := norm_pos_iff.mpr hx
      have h2 : ‖fderiv ℝ (hc c) x‖ ^ 2 ≤ 16 * (fC C hC x * (‖x‖ ^ 2)⁻¹) := by
        have h3 : ‖fderiv ℝ (hc c) x‖ ^ 2 ≤ (2 * (2 * ‖dir x - c‖ / ‖x‖)) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) h1 2
        refine h3.trans (le_of_eq ?_)
        rw [← hxA', hc]
        field_simp
        ring
      calc ENNReal.ofReal (‖fderiv ℝ (hc c) x‖ ^ 2)
          ≤ ENNReal.ofReal (16 * (fC C hC x * (‖x‖ ^ 2)⁻¹)) := ENNReal.ofReal_le_ofReal h2
        _ = 16 * (ENNReal.ofReal (fC C hC x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹)) := by
            rw [ENNReal.ofReal_mul (by norm_num), ENNReal.ofReal_mul (fC_nonneg C hC x)]
            norm_num
    · rw [Set.indicator_of_notMem hxA, zero_mul, zero_mul]
  refine (Finset.sum_le_sum hbound).trans ?_
  have hm16 : Measurable fun x : E =>
      16 * (ENNReal.ofReal (fC C hC x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹)) :=
    measurable_const.mul ((measurable_fC C hC).ennreal_ofReal.mul
      ((measurable_norm.pow_const 2).inv.ennreal_ofReal))
  have hsum : ∫⁻ x, ∑ c ∈ C, (A c).indicator (fun _ => (1 : ℝ≥0∞)) x *
        (16 * (ENNReal.ofReal (fC C hC x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹))) ∂(gaussianE E)
      = ∑ c ∈ C, ∫⁻ x, (A c).indicator (fun _ => (1 : ℝ≥0∞)) x *
          (16 * (ENNReal.ofReal (fC C hC x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹))) ∂(gaussianE E) :=
    lintegral_finsetSum _ fun c _ => (measurable_const.indicator (hAmeas c)).mul hm16
  have hconst : ∫⁻ x, 16 * (ENNReal.ofReal (fC C hC x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹))
        ∂(gaussianE E)
      = 16 * ∫⁻ x, ENNReal.ofReal (fC C hC x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹) ∂(gaussianE E) :=
    lintegral_const_mul _ ((measurable_fC C hC).ennreal_ofReal.mul
      ((measurable_norm.pow_const 2).inv.ennreal_ofReal))
  rw [← hsum, ← hconst]
  refine lintegral_mono_ae ?_
  filter_upwards [ae_sum_active_le_one hd C hC] with x hx
  rw [← Finset.sum_mul]
  calc (∑ c ∈ C, (A c).indicator (fun _ => (1 : ℝ≥0∞)) x) *
        (16 * (ENNReal.ofReal (fC C hC x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹)))
      ≤ 1 * (16 * (ENNReal.ofReal (fC C hC x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹))) :=
        mul_le_mul' hx le_rfl
    _ = _ := by rw [one_mul]

theorem measurable_pathMap :
    Measurable fun q : (E × E) × ℝ => (pathX q.1.1 q.1.2 q.2, pathY q.1.1 q.1.2 q.2) := by
  unfold pathX pathY; fun_prop

theorem measurable_SC_pathMap (C : Finset E) (hC : C.Nonempty) :
    Measurable fun q : (E × E) × ℝ =>
      ENNReal.ofReal (SC C hC (pathX q.1.1 q.1.2 q.2) (pathY q.1.1 q.1.2 q.2)) := by
  have h := Measurable.comp (g := fun p : E × E => SC C hC p.1 p.2)
    (f := fun q : (E × E) × ℝ => (pathX q.1.1 q.1.2 q.2, pathY q.1.1 q.1.2 q.2))
    (measurable_SC C hC) measurable_pathMap
  exact h.ennreal_ofReal

/-- **The Gaussian path bound**: `E (f(X) − f(Y))² ≤ (π/2)² · 16 · E[f(Z)] / (d − 2)` for
`f = fC`, by the Maurey–Pisier rotation, Tonelli, and the `1/‖Z‖²` moment. -/
theorem lintegral_sq_sub_fC_le (hd : 3 ≤ Module.finrank ℝ E) (C : Finset E) (hC : C.Nonempty) :
    ∫⁻ p, ENNReal.ofReal ((fC C hC p.1 - fC C hC p.2) ^ 2) ∂((gaussianE E).prod (gaussianE E))
      ≤ ENNReal.ofReal (Real.pi / 2) * ENNReal.ofReal (Real.pi / 2) *
          (16 * ((∫⁻ x, ENNReal.ofReal (fC C hC x) ∂(gaussianE E)) *
            ENNReal.ofReal (1 / ((Module.finrank ℝ E : ℝ) - 2)))) := by
  have hd2 : 2 ≤ Module.finrank ℝ E := by omega
  have h1 : ∫⁻ p, ENNReal.ofReal ((fC C hC p.1 - fC C hC p.2) ^ 2)
        ∂((gaussianE E).prod (gaussianE E))
      ≤ ENNReal.ofReal (Real.pi / 2) * ∫⁻ p, (∫⁻ θ in Set.Ioc (0 : ℝ) (Real.pi / 2),
          ENNReal.ofReal (SC C hC (pathX p.1 p.2 θ) (pathY p.1 p.2 θ)))
          ∂((gaussianE E).prod (gaussianE E)) := by
    rw [← lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
    refine lintegral_mono_ae ?_
    filter_upwards [ae_good hd2] with p hp
    have := ofReal_sq_sub_fC_le C hC hp
    rwa [show (fC C hC p.2 - fC C hC p.1) ^ 2 = (fC C hC p.1 - fC C hC p.2) ^ 2 by ring] at this
  have h2 : ∫⁻ p, (∫⁻ θ in Set.Ioc (0 : ℝ) (Real.pi / 2),
        ENNReal.ofReal (SC C hC (pathX p.1 p.2 θ) (pathY p.1 p.2 θ)))
        ∂((gaussianE E).prod (gaussianE E))
      = ∫⁻ θ in Set.Ioc (0 : ℝ) (Real.pi / 2), (∫⁻ p,
          ENNReal.ofReal (SC C hC (pathX p.1 p.2 θ) (pathY p.1 p.2 θ))
          ∂((gaussianE E).prod (gaussianE E))) :=
    lintegral_lintegral_swap (measurable_SC_pathMap C hC).aemeasurable
  have h3 : ∀ θ : ℝ, ∫⁻ p, ENNReal.ofReal (SC C hC (pathX p.1 p.2 θ) (pathY p.1 p.2 θ))
        ∂((gaussianE E).prod (gaussianE E))
      = ∫⁻ p, ENNReal.ofReal (SC C hC p.1 p.2) ∂((gaussianE E).prod (gaussianE E)) := fun θ =>
    (measurePreserving_rotProd θ).lintegral_comp
      (f := fun p : E × E => ENNReal.ofReal (SC C hC p.1 p.2)) (measurable_SC C hC).ennreal_ofReal
  have h4 : ∫⁻ _ in Set.Ioc (0 : ℝ) (Real.pi / 2),
        (∫⁻ p, ENNReal.ofReal (SC C hC p.1 p.2) ∂((gaussianE E).prod (gaussianE E)))
      = ENNReal.ofReal (Real.pi / 2) *
          ∫⁻ p, ENNReal.ofReal (SC C hC p.1 p.2) ∂((gaussianE E).prod (gaussianE E)) := by
    rw [lintegral_const, Measure.restrict_apply_univ, Real.volume_Ioc, sub_zero, mul_comm]
  have h6 : ∫⁻ x, ENNReal.ofReal (fC C hC x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹) ∂(gaussianE E)
      = (∫⁻ x, ENNReal.ofReal (fC C hC x) ∂(gaussianE E)) *
          ENNReal.ofReal (1 / ((Module.finrank ℝ E : ℝ) - 2)) := by
    have := lintegral_gaussianE_dir_div_normSq hd (ψ := fun u => ENNReal.ofReal (fC C hC u))
      (measurable_fC C hC).ennreal_ofReal
    simp only [fC_dir] at this
    rw [this, ← measurePreserving_dir.lintegral_comp (f := fun u => ENNReal.ofReal (fC C hC u))
      (measurable_fC C hC).ennreal_ofReal]
    simp only [fC_dir]
  calc _ ≤ _ := h1
    _ = ENNReal.ofReal (Real.pi / 2) * (ENNReal.ofReal (Real.pi / 2) *
          ∫⁻ p, ENNReal.ofReal (SC C hC p.1 p.2) ∂((gaussianE E).prod (gaussianE E))) := by
        rw [h2]; simp_rw [h3]; rw [h4]
    _ ≤ ENNReal.ofReal (Real.pi / 2) * (ENNReal.ofReal (Real.pi / 2) *
          (16 * ((∫⁻ x, ENNReal.ofReal (fC C hC x) ∂(gaussianE E)) *
            ENNReal.ofReal (1 / ((Module.finrank ℝ E : ℝ) - 2))))) := by
        refine mul_le_mul' le_rfl (mul_le_mul' le_rfl ?_)
        rw [← h6]; exact lintegral_SC_le hd2 C hC
    _ = _ := by ring

theorem norm_dir_le_one (z : E) : ‖dir z‖ ≤ 1 := by
  rcases eq_or_ne z 0 with rfl | hz
  · simp [dir]
  · rw [dir, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr hz)]

theorem fC_le (C : Finset E) (hC : C.Nonempty) (z : E) :
    fC C hC z ≤ (1 + C.sup' hC (fun c => ‖c‖)) ^ 2 := by
  have hmem := hC.choose_spec
  calc fC C hC z ≤ hc hC.choose z := Finset.inf'_le _ hmem
    _ = ‖dir z - hC.choose‖ ^ 2 := rfl
    _ ≤ (‖dir z‖ + ‖hC.choose‖) ^ 2 := pow_le_pow_left₀ (norm_nonneg _) (norm_sub_le _ _) 2
    _ ≤ (1 + C.sup' hC (fun c => ‖c‖)) ^ 2 := by
        gcongr
        · exact norm_dir_le_one z
        · exact Finset.le_sup' (fun c => ‖c‖) hmem

theorem memLp_fC (C : Finset E) (hC : C.Nonempty) : MemLp (fC C hC) 2 (gaussianE E) :=
  MemLp.of_bound (measurable_fC C hC).aestronglyMeasurable _
    (Filter.Eventually.of_forall fun z => by
      rw [Real.norm_eq_abs, abs_of_nonneg (fC_nonneg C hC z)]; exact fC_le C hC z)

theorem lintegral_fC_ne_top (C : Finset E) (hC : C.Nonempty) :
    ∫⁻ x, ENNReal.ofReal (fC C hC x) ∂(gaussianE E) ≠ ⊤ := by
  refine ne_top_of_le_ne_top (b := ENNReal.ofReal ((1 + C.sup' hC (fun c => ‖c‖)) ^ 2) *
    gaussianE E Set.univ) (ENNReal.mul_ne_top ENNReal.ofReal_ne_top (measure_ne_top _ _)) ?_
  rw [← lintegral_const]
  exact lintegral_mono fun z => ENNReal.ofReal_le_ofReal (fC_le C hC z)

/-- **The Gaussian Poincaré inequality for the squared distance to a finite codebook**, with
the explicit constant `2π²/(d − 2)`: `Var_γ fC ≤ 2π²/(d − 2) · E_γ fC`. -/
theorem variance_fC_le (hd : 3 ≤ Module.finrank ℝ E) (C : Finset E) (hC : C.Nonempty) :
    variance (fC C hC) (gaussianE E)
      ≤ 2 * Real.pi ^ 2 / ((Module.finrank ℝ E : ℝ) - 2) * ∫ x, fC C hC x ∂(gaussianE E) := by
  have hd' : (0 : ℝ) < (Module.finrank ℝ E : ℝ) - 2 := by
    have : (3 : ℝ) ≤ Module.finrank ℝ E := by exact_mod_cast hd
    linarith
  have hdne := hd'.ne'
  rw [variance_eq_half_integral_prod (memLp_fC C hC)]
  have hnn : 0 ≤ᵐ[(gaussianE E).prod (gaussianE E)]
      fun p : E × E => (fC C hC p.1 - fC C hC p.2) ^ 2 :=
    Filter.Eventually.of_forall fun p => sq_nonneg _
  have hmeas : AEStronglyMeasurable (fun p : E × E => (fC C hC p.1 - fC C hC p.2) ^ 2)
      ((gaussianE E).prod (gaussianE E)) :=
    ((((measurable_fC C hC).comp measurable_fst).sub
      ((measurable_fC C hC).comp measurable_snd)).pow_const 2).aestronglyMeasurable
  rw [integral_eq_lintegral_of_nonneg_ae hnn hmeas]
  have hI : ∫ x, fC C hC x ∂(gaussianE E)
      = (∫⁻ x, ENNReal.ofReal (fC C hC x) ∂(gaussianE E)).toReal :=
    integral_eq_lintegral_of_nonneg_ae (Filter.Eventually.of_forall (fC_nonneg C hC))
      (measurable_fC C hC).aestronglyMeasurable
  have hRne : ENNReal.ofReal (Real.pi / 2) * ENNReal.ofReal (Real.pi / 2) *
      (16 * ((∫⁻ x, ENNReal.ofReal (fC C hC x) ∂(gaussianE E)) *
        ENNReal.ofReal (1 / ((Module.finrank ℝ E : ℝ) - 2)))) ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top ENNReal.ofReal_ne_top ENNReal.ofReal_ne_top)
      (ENNReal.mul_ne_top (by simp)
        (ENNReal.mul_ne_top (lintegral_fC_ne_top C hC) ENNReal.ofReal_ne_top))
  have hle := ENNReal.toReal_mono hRne (lintegral_sq_sub_fC_le hd C hC)
  simp only [ENNReal.toReal_mul, ENNReal.toReal_ofNat] at hle
  rw [ENNReal.toReal_ofReal (by positivity), ENNReal.toReal_ofReal (one_div_nonneg.mpr hd'.le),
    ← hI] at hle
  calc (1 / 2 : ℝ) * _ ≤ (1 / 2) * (Real.pi / 2 * (Real.pi / 2) *
        (16 * ((∫ x, fC C hC x ∂(gaussianE E)) * (1 / ((Module.finrank ℝ E : ℝ) - 2))))) :=
        mul_le_mul_of_nonneg_left hle (by norm_num)
    _ = _ := by field_simp; ring

theorem fC_eq_of_isNearest {q : E → E} (C : Finset E) (hC : C.Nonempty)
    (hq : IsNearest (C : Set E) q) (z : E) : fC C hC z = ‖dir z - q (dir z)‖ ^ 2 := by
  obtain ⟨hmem, hmin⟩ := hq (dir z)
  refine le_antisymm (Finset.inf'_le _ (Finset.mem_coe.mp hmem))
    (Finset.le_inf' hC _ fun c hc' => ?_)
  exact pow_le_pow_left₀ (norm_nonneg _) (hmin c (Finset.mem_coe.mpr hc')) 2

/-- **The variance bound for a finite codebook on the sphere**, with the explicit constant
`2π²/(d − 2)`: `Var_σ ‖u − q u‖² ≤ 2π²/(d − 2) · ρ²`, `ρ² = stageDistortion q σ`. -/
theorem variance_stageErr_le_finite (hd : 3 ≤ Module.finrank ℝ E) {q : E → E} (C : Finset E)
    (hC : C.Nonempty) (hq : IsNearest (C : Set E) q) (hmeas : Measurable q) :
    variance (fun u => ‖u - q u‖ ^ 2) (sphereUniformAmbient E)
      ≤ 2 * Real.pi ^ 2 / ((Module.finrank ℝ E : ℝ) - 2) *
          stageDistortion q (sphereUniformAmbient E) := by
  have hF : Measurable fun x : E => ‖x - q x‖ ^ 2 := (measurable_id.sub hmeas).norm.pow_const 2
  have hcomp : (fun x : E => ‖x - q x‖ ^ 2) ∘ dir = fC C hC := by
    funext z; simp only [Function.comp]; rw [fC_eq_of_isNearest C hC hq]
  have hint : ∫ x, ‖dir x - q (dir x)‖ ^ 2 ∂(gaussianE E) = ∫ x, fC C hC x ∂(gaussianE E) :=
    integral_congr_ae (Filter.Eventually.of_forall fun z => (fC_eq_of_isNearest C hC hq z).symm)
  rw [← gaussianE_map_dir, variance_map (by rw [gaussianE_map_dir]; exact hF.aemeasurable)
    measurable_dir.aemeasurable, hcomp, stageDistortion, integral_map measurable_dir.aemeasurable
    (by rw [gaussianE_map_dir]; exact hF.aestronglyMeasurable), hint]
  exact variance_fC_le hd C hC

end Assembly

section StreamFinite

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] [FiniteDimensional ℝ E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G] [MeasurableMul G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {q : E → E} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G}
  [κ.IsMulRightInvariant] {x₀ : E}

/-- **The per-pass variance bound on the stream for a finite codebook**, with constant
`2π²/(d − 2)` and no Poincaré hypothesis. -/
theorem variance_stageErrSq_le_finite (hd : 3 ≤ Module.finrank ℝ E)
    (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1) (C : Finset E) (hC : C.Nonempty)
    (hq : IsNearest (C : Set E) q) (p : ℕ) :
    variance (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p) μ
      ≤ 2 * Real.pi ^ 2 / ((Module.finrank ℝ E : ℝ) - 2) *
          stageDistortion q (sphereUniformAmbient E) := by
  haveI : Nontrivial E := ⟨⟨x₀, 0, by rintro rfl; simp at hx₀⟩⟩
  have hmp := measurePreserving_unitVec_stream h hx₀ p
  have hF : Measurable fun x : E => ‖x - q x‖ ^ 2 := (measurable_id.sub h.meas_q).norm.pow_const 2
  have heq : (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p)
      = (fun x : E => ‖x - q x‖ ^ 2) ∘ (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p) := rfl
  rw [heq, ← variance_map (by rw [hmp.map_eq]; exact hF.aemeasurable) hmp.measurable.aemeasurable,
    hmp.map_eq]
  exact variance_stageErr_le_finite hd C hC hq h.meas_q

/-- **Concentration of the decay law for a finite codebook, with no unproved hypothesis.**
For a nearest-codeword stage with a finite codebook in dimension `d ≥ 3` and `P` passes of a
rotation stream, the `P`-pass squared error lies within the compounded band
`(1 ± ε)^P ρ^{2P}‖w‖²` except on an event of probability at most `P · 2π² / ((d − 2) ε² ρ²)`.
This is the paper's Remark (Concentration) with its rate `P/(d ε² ρ²)` made explicit, for
scalar, `E8`, and trellis codebooks alike, and it is proved outright. -/
theorem meas_not_decayWithin_le_finite (hd : 3 ≤ Module.finrank ℝ E)
    (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1) (C : Finset E) (hC : C.Nonempty)
    (hq : IsNearest (C : Set E) q) {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε ≤ 1)
    (hρ : 0 < stageDistortion q (sphereUniformAmbient E)) (P : ℕ) :
    μ {ω | ¬ DecayWithin (rotScheme ι q (R ω)) (W ω)
        (Real.sqrt (stageDistortion q (sphereUniformAmbient E))) ε P}
      ≤ P * ENNReal.ofReal (2 * Real.pi ^ 2 / (((Module.finrank ℝ E : ℝ) - 2) * ε ^ 2 *
          stageDistortion q (sphereUniformAmbient E))) := by
  haveI : Nontrivial E := ⟨⟨x₀, 0, by rintro rfl; simp at hx₀⟩⟩
  have hCb : Bornology.IsBounded (C : Set E) := C.finite_toSet.isBounded
  set D := stageDistortion q (sphereUniformAmbient E) with hD
  have hρ2 : Real.sqrt D ^ 2 = D := Real.sq_sqrt hρ.le
  have hmem : ∀ p < P, MemLp (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p) 2 μ := fun p _ =>
    (memLp_stageErr_of_bounded hq hCb h.meas_q).comp_measurePreserving
      (measurePreserving_unitVec_stream h hx₀ p)
  have hmean : ∀ p < P, ∫ ω, stageErrSq (rotScheme ι q (R ω)) (W ω) p ∂μ = Real.sqrt D ^ 2 := by
    intro p _
    rw [hρ2, hD, ← h.orbitLaw_eq hx₀]
    exact integral_stageErr_eq_stageDistortion h p
  have hvar : ∀ p < P, variance (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p) μ
      ≤ 2 * Real.pi ^ 2 / ((Module.finrank ℝ E : ℝ) - 2) * D := fun p _ =>
    variance_stageErrSq_le_finite hd h hx₀ C hC hq p
  have hρne : Real.sqrt D ≠ 0 := (Real.sqrt_pos.mpr hρ).ne'
  have key := meas_not_decayWithin_le_of_variance hε0 hε1 hρne hmem hmean hvar
  have hD0 : D ≠ 0 := hρ.ne'
  have hε : ε ≠ 0 := hε0.ne'
  have hd' : ((Module.finrank ℝ E : ℝ) - 2) ≠ 0 := by
    have : (3 : ℝ) ≤ Module.finrank ℝ E := by exact_mod_cast hd
    linarith
  have hconst : 2 * Real.pi ^ 2 / ((Module.finrank ℝ E : ℝ) - 2) * D / (ε * Real.sqrt D ^ 2) ^ 2
      = 2 * Real.pi ^ 2 / (((Module.finrank ℝ E : ℝ) - 2) * ε ^ 2 * D) := by
    rw [hρ2]; field_simp
  rw [← hconst]; exact key

end StreamFinite
