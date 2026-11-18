/-! ## §34  The sharp constant `4/(d − 2)` for finite codebooks

Applying §32 to `g = χ_δ f_ε` and letting `δ = ε → 0` gives the Gaussian Poincaré inequality
for `f_C` with the sharp constant, `Var_γ f_C ≤ 4/(d − 2) · E_γ f_C` (`variance_fC_le_sharp`),
replacing the `2π²/(d − 2)` of §30.  The error terms are the cutoff's contribution to the
derivative, `O(γ(B(0, 2δ))/δ²) = O(δ^{d−2})` since the Gaussian density is at most `1/Z`
(`gaussianE_ball_le`), and the `L²` distance from `f_C` to `χ_δ f_ε`, `O(ε² + δ^d)`; the
comparison `Var f ≤ (1 + η) Var g + (1 + 1/η) E(f − g)²` (`variance_le_of_close`) and two
limits close the argument.  Transported to the sphere, `Var_σ ‖u − q u‖² ≤ 4ρ²/(d − 2)` for
every nearest-codeword stage with a finite codebook (`variance_stageErr_le_finite_sharp`), and
`meas_not_decayWithin_le_finite_sharp` is the concentration remark at rate
`4P/((d − 2) ε² ρ²)`, the paper's `P/(d ε² ρ²)` with the constant made explicit, for scalar,
`E8` and trellis codebooks, with no unproved hypothesis. -/

section BallBound

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

theorem gaussDensity_le_one (x : E) : gaussDensity x ≤ 1 := by
  unfold gaussDensity
  rw [← ENNReal.ofReal_one]
  exact ENNReal.ofReal_le_ofReal (Real.exp_le_one_iff.mpr (by
    have := sq_nonneg ‖x‖; linarith))

/-- The Gaussian measure of a set is at most `Z⁻¹` times its volume. -/
theorem gaussianE_apply_le {s : Set E} (hs : MeasurableSet s) :
    gaussianE E s ≤ (gaussRaw E Set.univ)⁻¹ * volume s := by
  unfold gaussianE
  rw [Measure.smul_apply, smul_eq_mul]
  refine mul_le_mul' le_rfl ?_
  unfold gaussRaw
  rw [withDensity_apply _ hs]
  calc ∫⁻ x in s, gaussDensity x ∂volume ≤ ∫⁻ _ in s, (1 : ℝ≥0∞) ∂volume :=
        lintegral_mono fun x => gaussDensity_le_one x
    _ = volume s := setLIntegral_one s

/-- The normalising constant of the ball bound, as a real number. -/
noncomputable def ballConst (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [MeasurableSpace E] [BorelSpace E] [FiniteDimensional ℝ E] : ℝ :=
  ((gaussRaw E Set.univ)⁻¹ * volume (Metric.ball (0 : E) 1)).toReal

theorem ballConst_nonneg : 0 ≤ ballConst E := ENNReal.toReal_nonneg

/-- `γ(B(0, r)) ≤ ballConst · r^d`. -/
theorem gaussianE_ball_le {r : ℝ} (hr : 0 ≤ r) :
    (gaussianE E (Metric.ball 0 r)).toReal ≤ ballConst E * r ^ Module.finrank ℝ E := by
  have h1 := gaussianE_apply_le (E := E) (measurableSet_ball (x := (0 : E)) (ε := r))
  rw [Measure.addHaar_ball (volume : Measure E) (0 : E) hr] at h1
  have hfin : (gaussRaw E Set.univ)⁻¹ * volume (Metric.ball (0 : E) 1) ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.inv_ne_top.mpr gaussRaw_univ_ne_zero) measure_ball_lt_top.ne
  calc (gaussianE E (Metric.ball 0 r)).toReal
      ≤ ((gaussRaw E Set.univ)⁻¹ * (ENNReal.ofReal (r ^ Module.finrank ℝ E) *
          volume (Metric.ball (0 : E) 1))).toReal :=
        ENNReal.toReal_mono (ENNReal.mul_ne_top (ENNReal.inv_ne_top.mpr gaussRaw_univ_ne_zero)
          (ENNReal.mul_ne_top ENNReal.ofReal_ne_top measure_ball_lt_top.ne)) h1
    _ = ballConst E * r ^ Module.finrank ℝ E := by
        unfold ballConst
        rw [mul_left_comm, ENNReal.toReal_mul, ENNReal.toReal_ofReal (by positivity), mul_comm]

end BallBound

section VarianceCompare

variable {X : Type*} [MeasurableSpace X] {μ : Measure X} [IsProbabilityMeasure μ]

/-- `Var f ≤ (1 + η) Var g + (1 + 1/η) E(f − g)²` for bounded `f, g`. -/
theorem variance_le_of_close {f g : X → ℝ} (hf : Measurable f) (hg : Measurable g) {Mf Mg : ℝ}
    (hfb : ∀ x, |f x| ≤ Mf) (hgb : ∀ x, |g x| ≤ Mg) {η : ℝ} (hη : 0 < η) :
    variance f μ ≤ (1 + η) * variance g μ + (1 + 1 / η) * ∫ x, (f x - g x) ^ 2 ∂μ := by
  have hfmem : MemLp f 2 μ := MemLp.of_bound hf.aestronglyMeasurable Mf
    (Filter.Eventually.of_forall fun x => by rw [Real.norm_eq_abs]; exact hfb x)
  have hgmem : MemLp g 2 μ := MemLp.of_bound hg.aestronglyMeasurable Mg
    (Filter.Eventually.of_forall fun x => by rw [Real.norm_eq_abs]; exact hgb x)
  have hbound : ∀ x, (f x - ∫ y, g y ∂μ) ^ 2
      ≤ (1 + η) * (g x - ∫ y, g y ∂μ) ^ 2 + (1 + 1 / η) * (f x - g x) ^ 2 := by
    intro x
    have key : ∀ a b : ℝ, (a + b) ^ 2 ≤ (1 + η) * a ^ 2 + (1 + 1 / η) * b ^ 2 := by
      intro a b
      have h0 : 0 ≤ (η * a - b) ^ 2 / η := div_nonneg (sq_nonneg _) hη.le
      have e : (η * a - b) ^ 2 / η = η * a ^ 2 - 2 * a * b + b ^ 2 / η := by
        field_simp; ring
      rw [e] at h0
      have e2 : (1 + 1 / η) * b ^ 2 = b ^ 2 + b ^ 2 / η := by ring
      rw [e2]
      nlinarith [h0]
    have := key (g x - ∫ y, g y ∂μ) (f x - g x)
    rwa [show g x - ∫ y, g y ∂μ + (f x - g x) = f x - ∫ y, g y ∂μ by ring] at this
  -- `Var f ≤ E(f − a)²` for every `a`
  have hvar : variance f μ ≤ ∫ x, (f x - ∫ y, g y ∂μ) ^ 2 ∂μ := by
    set a := ∫ y, g y ∂μ
    have hfi : Integrable f μ := hfmem.integrable one_le_two
    have hf2 : Integrable (fun x => f x ^ 2) μ := hfmem.integrable_sq
    rw [variance_eq_sub hfmem]
    have h1 : ∫ x, (f x - a) ^ 2 ∂μ = ∫ x, f x ^ 2 ∂μ - 2 * a * ∫ x, f x ∂μ + a ^ 2 := by
      have : (fun x => (f x - a) ^ 2) = fun x => f x ^ 2 - 2 * a * f x + a ^ 2 := by
        funext x; ring
      have hB : Integrable (fun x => 2 * a * f x) μ := hfi.const_mul _
      have hA : Integrable (fun x => f x ^ 2 - 2 * a * f x) μ := hf2.sub hB
      rw [this, integral_add hA (integrable_const _), integral_sub hf2 hB, integral_const_mul,
        integral_const, measureReal_def, measure_univ, ENNReal.toReal_one, one_smul]
    rw [h1]
    have : ∫ x, (f ^ 2) x ∂μ = ∫ x, f x ^ 2 ∂μ := rfl
    rw [this]
    nlinarith [sq_nonneg ((∫ x, f x ∂μ) - a)]
  refine hvar.trans ?_
  have hgi : Integrable (fun x => (g x - ∫ y, g y ∂μ) ^ 2) μ :=
    (hgmem.sub (memLp_const _)).integrable_sq
  have hfgi : Integrable (fun x => (f x - g x) ^ 2) μ := (hfmem.sub hgmem).integrable_sq
  calc ∫ x, (f x - ∫ y, g y ∂μ) ^ 2 ∂μ
      ≤ ∫ x, ((1 + η) * (g x - ∫ y, g y ∂μ) ^ 2 + (1 + 1 / η) * (f x - g x) ^ 2) ∂μ :=
        integral_mono ((hfmem.sub (memLp_const _)).integrable_sq)
          ((hgi.const_mul _).add (hfgi.const_mul _)) hbound
    _ = (1 + η) * variance g μ + (1 + 1 / η) * ∫ x, (f x - g x) ^ 2 ∂μ := by
        rw [integral_add (hgi.const_mul _) (hfgi.const_mul _), integral_const_mul,
          integral_const_mul, variance_eq_integral hg.aemeasurable]

end VarianceCompare

section WeightedIntegrals

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- `E[1/‖Z‖²] = 1/(d − 2)` as a real integral. -/
theorem integral_inv_normSq_gaussianE (hd : 3 ≤ Module.finrank ℝ E) :
    ∫ x, (‖x‖ ^ 2)⁻¹ ∂(gaussianE E) = 1 / ((Module.finrank ℝ E : ℝ) - 2) := by
  have h := lintegral_gaussianE_dir_div_normSq hd (ψ := fun _ => (1 : ℝ≥0∞)) measurable_const
  simp only [one_mul, lintegral_const, measure_univ] at h
  have hm : Measurable fun x : E => (‖x‖ ^ 2)⁻¹ := (measurable_norm.pow_const 2).inv
  rw [integral_eq_lintegral_of_nonneg_ae (Filter.Eventually.of_forall fun x => by positivity)
    hm.aestronglyMeasurable, h, ENNReal.toReal_ofReal]
  have : (3 : ℝ) ≤ Module.finrank ℝ E := by exact_mod_cast hd
  exact div_nonneg zero_le_one (by linarith)

theorem integrable_inv_normSq_gaussianE (hd : 3 ≤ Module.finrank ℝ E) :
    Integrable (fun x : E => (‖x‖ ^ 2)⁻¹) (gaussianE E) := by
  have hm : Measurable fun x : E => (‖x‖ ^ 2)⁻¹ := (measurable_norm.pow_const 2).inv
  refine ⟨hm.aestronglyMeasurable, ?_⟩
  rw [hasFiniteIntegral_iff_ofReal (Filter.Eventually.of_forall fun x => by positivity)]
  have h := lintegral_gaussianE_dir_div_normSq hd (ψ := fun _ => (1 : ℝ≥0∞)) measurable_const
  simp only [one_mul, lintegral_const, measure_univ] at h
  rw [h]; exact ENNReal.ofReal_lt_top

/-- `E[f_C(Z)/‖Z‖²] = E[f_C]/(d − 2)`. -/
theorem integral_fC_div_normSq (hd : 3 ≤ Module.finrank ℝ E) (C : Finset E) (hC : C.Nonempty) :
    ∫ x, fC C hC x * (‖x‖ ^ 2)⁻¹ ∂(gaussianE E)
      = (∫ x, fC C hC x ∂(gaussianE E)) / ((Module.finrank ℝ E : ℝ) - 2) := by
  have hd' : (0 : ℝ) < (Module.finrank ℝ E : ℝ) - 2 := by
    have : (3 : ℝ) ≤ Module.finrank ℝ E := by exact_mod_cast hd
    linarith
  have h := lintegral_gaussianE_dir_div_normSq hd (ψ := fun u => ENNReal.ofReal (fC C hC u))
    (measurable_fC C hC).ennreal_ofReal
  simp only [fC_dir] at h
  rw [← measurePreserving_dir.lintegral_comp (f := fun u => ENNReal.ofReal (fC C hC u))
    (measurable_fC C hC).ennreal_ofReal] at h
  simp only [fC_dir] at h
  have hm2 : Measurable fun x : E => fC C hC x * (‖x‖ ^ 2)⁻¹ :=
    (measurable_fC C hC).mul (measurable_norm.pow_const 2).inv
  rw [integral_eq_lintegral_of_nonneg_ae (Filter.Eventually.of_forall fun x =>
      mul_nonneg (fC_nonneg C hC x) (by positivity)) hm2.aestronglyMeasurable]
  simp_rw [ENNReal.ofReal_mul (fC_nonneg C hC _)]
  rw [h, ENNReal.toReal_mul, ENNReal.toReal_ofReal (by positivity),
    integral_eq_lintegral_of_nonneg_ae (Filter.Eventually.of_forall (fC_nonneg C hC))
      (measurable_fC C hC).aestronglyMeasurable]
  ring

theorem integrable_fC_div_normSq (hd : 3 ≤ Module.finrank ℝ E) (C : Finset E) (hC : C.Nonempty) :
    Integrable (fun x : E => fC C hC x * (‖x‖ ^ 2)⁻¹) (gaussianE E) := by
  have hm2 : Measurable fun x : E => fC C hC x * (‖x‖ ^ 2)⁻¹ :=
    (measurable_fC C hC).mul (measurable_norm.pow_const 2).inv
  refine ⟨hm2.aestronglyMeasurable, ?_⟩
  rw [hasFiniteIntegral_iff_ofReal (Filter.Eventually.of_forall fun x =>
    mul_nonneg (fC_nonneg C hC x) (by positivity))]
  have h := lintegral_gaussianE_dir_div_normSq hd (ψ := fun u => ENNReal.ofReal (fC C hC u))
    (measurable_fC C hC).ennreal_ofReal
  simp only [fC_dir] at h
  simp_rw [ENNReal.ofReal_mul (fC_nonneg C hC _)]
  rw [h]
  exact ENNReal.mul_lt_top (lt_of_le_of_ne le_top (by
    rw [← measurePreserving_dir.lintegral_comp (f := fun u => ENNReal.ofReal (fC C hC u))
      (measurable_fC C hC).ennreal_ofReal]
    simp only [fC_dir]
    exact lintegral_fC_ne_top C hC)) ENNReal.ofReal_lt_top

end WeightedIntegrals

section SharpAssembly

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E] [SecondCountableTopology E]

/-- The cut-off smooth minimum `χ_δ · f_ε`. -/
noncomputable def gApprox (C : Finset E) (ε δ : ℝ) (z : E) : ℝ := cutoff δ z * smoothMin C ε z

/-- The bound on the codebook: `M₀ = (1 + max ‖c‖)²`. -/
noncomputable def codeBound (C : Finset E) (hC : C.Nonempty) : ℝ := (1 + C.sup' hC (fun c => ‖c‖)) ^ 2

theorem codeBound_nonneg (C : Finset E) (hC : C.Nonempty) : 0 ≤ codeBound C hC := sq_nonneg _

theorem sup_norm_nonneg (C : Finset E) (hC : C.Nonempty) : 0 ≤ C.sup' hC (fun c => ‖c‖) := by
  obtain ⟨c, hc⟩ := hC
  exact (norm_nonneg c).trans (Finset.le_sup' (fun c => ‖c‖) hc)

theorem abs_smoothMin_le (C : Finset E) (hC : C.Nonempty) {ε : ℝ} (hε : 0 < ε) (hε1 : ε ≤ 1)
    (z : E) : |smoothMin C ε z| ≤ codeBound C hC + Real.log C.card := by
  have hlog : 0 ≤ Real.log C.card := Real.log_nonneg (by exact_mod_cast hC.card_pos)
  have h1 := abs_smoothMin_sub_fC_le C hC hε z
  have h2 := fC_le C hC z
  have h3 := fC_nonneg C hC z
  have h4 : ε * Real.log C.card ≤ Real.log C.card := by nlinarith
  unfold codeBound
  rw [abs_le] at h1 ⊢
  constructor <;> linarith

theorem abs_gApprox_le (C : Finset E) (hC : C.Nonempty) {ε : ℝ} (hε : 0 < ε) (hε1 : ε ≤ 1)
    (δ : ℝ) (z : E) : |gApprox C ε δ z| ≤ codeBound C hC + Real.log C.card := by
  unfold gApprox
  rw [abs_mul, abs_of_nonneg (cutoff_nonneg δ z)]
  calc cutoff δ z * |smoothMin C ε z| ≤ 1 * |smoothMin C ε z| :=
        mul_le_mul_of_nonneg_right (cutoff_le_one δ z) (abs_nonneg _)
    _ ≤ _ := by rw [one_mul]; exact abs_smoothMin_le C hC hε hε1 z

theorem hasFDerivAt_gApprox_of_lt (C : Finset E) {ε δ : ℝ} (hδ : 0 < δ) {z : E} (hz : ‖z‖ < δ) :
    HasFDerivAt (gApprox C ε δ) (0 : E →L[ℝ] ℝ) z := by
  refine (hasFDerivAt_const (0 : ℝ) z).congr_of_eventuallyEq ?_
  have hopen : {w : E | ‖w‖ < δ} ∈ 𝓝 z := (isOpen_lt continuous_norm continuous_const).mem_nhds hz
  filter_upwards [hopen] with w hw
  unfold gApprox
  rw [cutoff_eq_zero hδ (le_of_lt hw), zero_mul]

theorem hasFDerivAt_gApprox_of_ne (C : Finset E) (hC : C.Nonempty) {ε δ : ℝ} (hε : 0 < ε)
    (hδ : 0 < δ) {z : E} (hz : z ≠ 0) :
    HasFDerivAt (gApprox C ε δ)
      (cutoff δ z • fderiv ℝ (smoothMin C ε) z + smoothMin C ε z • fderiv ℝ (cutoff δ) z) z := by
  have h1 := hasFDerivAt_cutoff hδ z
  have h2 := hasFDerivAt_smoothMin C hC hε hz
  have h := h1.mul h2
  rw [h1.fderiv, h2.fderiv]
  exact h

theorem differentiable_gApprox (C : Finset E) (hC : C.Nonempty) {ε δ : ℝ} (hε : 0 < ε)
    (hδ : 0 < δ) : Differentiable ℝ (gApprox C ε δ) := by
  intro z
  rcases lt_or_ge ‖z‖ δ with hz | hz
  · exact (hasFDerivAt_gApprox_of_lt C hδ hz).differentiableAt
  · have hz0 : z ≠ 0 := by
      intro h; rw [h, norm_zero] at hz; linarith
    exact (hasFDerivAt_gApprox_of_ne C hC hε hδ hz0).differentiableAt

theorem norm_fderiv_smoothMin_le' (C : Finset E) (hC : C.Nonempty) {ε : ℝ} (hε : 0 < ε) {z : E}
    (hz : z ≠ 0) : ‖fderiv ℝ (smoothMin C ε) z‖ ≤ 2 * (1 + C.sup' hC (fun c => ‖c‖)) / ‖z‖ := by
  have hz' : 0 < ‖z‖ := norm_pos_iff.mpr hz
  refine (norm_fderiv_smoothMin_le C hC hε hz).trans ?_
  have hb : ∀ c ∈ C, 2 * ‖dir z - c‖ / ‖z‖ ≤ 2 * (1 + C.sup' hC (fun c => ‖c‖)) / ‖z‖ := by
    intro c hc'
    refine div_le_div_of_nonneg_right ?_ hz'.le
    have := norm_dir_le_one z
    have := Finset.le_sup' (fun c => ‖c‖) hc'
    have := norm_sub_le (dir z) c
    linarith
  calc ∑ c ∈ C, softWeight C ε c z * (2 * ‖dir z - c‖ / ‖z‖)
      ≤ ∑ c ∈ C, softWeight C ε c z * (2 * (1 + C.sup' hC (fun c => ‖c‖)) / ‖z‖) :=
        Finset.sum_le_sum fun c hc' =>
          mul_le_mul_of_nonneg_left (hb c hc') (softWeight_nonneg C ε c z)
    _ = 2 * (1 + C.sup' hC (fun c => ‖c‖)) / ‖z‖ := by
        rw [← Finset.sum_mul, sum_softWeight C hC, one_mul]

/-- The derivative of the cut-off smooth minimum is bounded, by `K/δ`. -/
theorem norm_fderiv_gApprox_le (C : Finset E) (hC : C.Nonempty) {Cc : ℝ}
    (hCc : ∀ δ : ℝ, 0 < δ → ∀ z : E, ‖fderiv ℝ (cutoff δ) z‖ ≤ Cc / δ) {ε δ : ℝ} (hε : 0 < ε)
    (hε1 : ε ≤ 1) (hδ : 0 < δ) (z : E) :
    ‖fderiv ℝ (gApprox C ε δ) z‖ ≤ (2 * (1 + C.sup' hC (fun c => ‖c‖)) +
      (codeBound C hC + Real.log C.card) * Cc) / δ := by
  have hs := sup_norm_nonneg C hC
  have hM := codeBound_nonneg C hC
  have hlog : 0 ≤ Real.log C.card := Real.log_nonneg (by exact_mod_cast hC.card_pos)
  have hCc0 : 0 ≤ Cc := by
    have := (norm_nonneg _).trans (hCc 1 one_pos 0); simpa using this
  rcases lt_or_ge ‖z‖ δ with hz | hz
  · rw [(hasFDerivAt_gApprox_of_lt C hδ hz).fderiv, norm_zero]; positivity
  · have hz0 : z ≠ 0 := by intro h; rw [h, norm_zero] at hz; linarith
    rw [(hasFDerivAt_gApprox_of_ne C hC hε hδ hz0).fderiv]
    refine (norm_add_le _ _).trans ?_
    rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (cutoff_nonneg δ z)]
    have h1 : cutoff δ z * ‖fderiv ℝ (smoothMin C ε) z‖ ≤ 2 * (1 + C.sup' hC (fun c => ‖c‖)) / δ := by
      calc cutoff δ z * ‖fderiv ℝ (smoothMin C ε) z‖ ≤ 1 * (2 * (1 + C.sup' hC (fun c => ‖c‖)) / ‖z‖) :=
            mul_le_mul (cutoff_le_one δ z) (norm_fderiv_smoothMin_le' C hC hε hz0) (norm_nonneg _)
              zero_le_one
        _ ≤ 2 * (1 + C.sup' hC (fun c => ‖c‖)) / δ := by
            rw [one_mul]; exact div_le_div_of_nonneg_left (by positivity) hδ hz
    have h2 : |smoothMin C ε z| * ‖fderiv ℝ (cutoff δ) z‖
        ≤ (codeBound C hC + Real.log C.card) * (Cc / δ) :=
      mul_le_mul (abs_smoothMin_le C hC hε hε1 z) (hCc δ hδ z) (norm_nonneg _) (by positivity)
    rw [add_div]
    calc _ ≤ 2 * (1 + C.sup' hC (fun c => ‖c‖)) / δ + (codeBound C hC + Real.log C.card) * (Cc / δ) :=
          add_le_add h1 h2
      _ = _ := by ring

/-- The pointwise bound on the squared derivative:
`‖g'(z)‖² ≤ 4 (f_C(z) + |C| ε)/‖z‖² + (A/δ²) 1_{B(0, 2δ)}(z)`. -/
theorem norm_fderiv_gApprox_sq_le (C : Finset E) (hC : C.Nonempty) {Cc : ℝ}
    (hCc : ∀ δ : ℝ, 0 < δ → ∀ z : E, ‖fderiv ℝ (cutoff δ) z‖ ≤ Cc / δ) {ε δ : ℝ} (hε : 0 < ε)
    (hε1 : ε ≤ 1) (hδ : 0 < δ) (z : E) :
    ‖fderiv ℝ (gApprox C ε δ) z‖ ^ 2
      ≤ 4 * (fC C hC z + C.card * ε) * (‖z‖ ^ 2)⁻¹
        + (Metric.ball (0 : E) (2 * δ)).indicator (fun _ =>
            (4 * (1 + C.sup' hC (fun c => ‖c‖)) * ((codeBound C hC + Real.log C.card) * Cc)
              + ((codeBound C hC + Real.log C.card) * Cc) ^ 2) / δ ^ 2) z := by
  have hs := sup_norm_nonneg C hC
  have hM := codeBound_nonneg C hC
  have hlog : 0 ≤ Real.log C.card := Real.log_nonneg (by exact_mod_cast hC.card_pos)
  have hCc0 : 0 ≤ Cc := by
    have := (norm_nonneg _).trans (hCc 1 one_pos 0); simpa using this
  have hf := fC_nonneg C hC z
  have hcard : (0 : ℝ) ≤ C.card := by positivity
  have hind : 0 ≤ (Metric.ball (0 : E) (2 * δ)).indicator (fun _ =>
      (4 * (1 + C.sup' hC (fun c => ‖c‖)) * ((codeBound C hC + Real.log C.card) * Cc)
        + ((codeBound C hC + Real.log C.card) * Cc) ^ 2) / δ ^ 2) z :=
    Set.indicator_nonneg (fun _ _ => by positivity) _
  have hmain : 0 ≤ 4 * (fC C hC z + C.card * ε) * (‖z‖ ^ 2)⁻¹ := by positivity
  rcases lt_or_ge ‖z‖ δ with hz | hz
  · rw [(hasFDerivAt_gApprox_of_lt C hδ hz).fderiv, norm_zero]; simp only [zero_pow two_ne_zero]
    linarith
  · have hz0 : z ≠ 0 := by intro h; rw [h, norm_zero] at hz; linarith
    have hz' : 0 < ‖z‖ := norm_pos_iff.mpr hz0
    rw [(hasFDerivAt_gApprox_of_ne C hC hε hδ hz0).fderiv]
    set a := ‖fderiv ℝ (smoothMin C ε) z‖ with ha
    set b := |smoothMin C ε z| * ‖fderiv ℝ (cutoff δ) z‖ with hb
    have hab : ‖cutoff δ z • fderiv ℝ (smoothMin C ε) z + smoothMin C ε z • fderiv ℝ (cutoff δ) z‖
        ≤ a + b := by
      refine (norm_add_le _ _).trans ?_
      rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
        abs_of_nonneg (cutoff_nonneg δ z)]
      exact add_le_add (mul_le_of_le_one_left (norm_nonneg _) (cutoff_le_one δ z)) le_rfl
    have ha2 : a ^ 2 ≤ 4 * (fC C hC z + C.card * ε) * (‖z‖ ^ 2)⁻¹ := by
      rw [ha, ← div_eq_mul_inv]; exact norm_fderiv_smoothMin_sq_le C hC hε hz0
    have hb0 : 0 ≤ b := by positivity
    have ha0 : 0 ≤ a := norm_nonneg _
    rcases lt_or_ge (2 * δ ^ 2) (‖z‖ ^ 2) with hz2 | hz2
    · -- outside the support of the cutoff derivative
      have : b = 0 := by rw [hb, fderiv_cutoff_eq_zero hδ hz2, norm_zero, mul_zero]
      rw [this, add_zero] at hab
      calc _ ≤ a ^ 2 := pow_le_pow_left₀ (norm_nonneg _) hab 2
        _ ≤ _ := by linarith
    · have hball : z ∈ Metric.ball (0 : E) (2 * δ) := by
        rw [Metric.mem_ball, dist_zero_right]
        nlinarith [sq_nonneg (‖z‖ - 2 * δ), hz']
      rw [Set.indicator_of_mem hball]
      have ha' : a ≤ 2 * (1 + C.sup' hC (fun c => ‖c‖)) / δ :=
        (norm_fderiv_smoothMin_le' C hC hε hz0).trans
          (div_le_div_of_nonneg_left (by positivity) hδ hz)
      have hb' : b ≤ (codeBound C hC + Real.log C.card) * Cc / δ := by
        rw [hb, mul_div_assoc]
        exact mul_le_mul (abs_smoothMin_le C hC hε hε1 z) (hCc δ hδ z) (norm_nonneg _)
          (by positivity)
      calc _ ≤ (a + b) ^ 2 := pow_le_pow_left₀ (norm_nonneg _) hab 2
        _ = a ^ 2 + (2 * a * b + b ^ 2) := by ring
        _ ≤ 4 * (fC C hC z + C.card * ε) * (‖z‖ ^ 2)⁻¹ +
            (4 * (1 + C.sup' hC (fun c => ‖c‖)) * ((codeBound C hC + Real.log C.card) * Cc)
              + ((codeBound C hC + Real.log C.card) * Cc) ^ 2) / δ ^ 2 := by
            refine add_le_add ha2 ?_
            have e : (4 * (1 + C.sup' hC (fun c => ‖c‖)) * ((codeBound C hC + Real.log C.card) * Cc)
                + ((codeBound C hC + Real.log C.card) * Cc) ^ 2) / δ ^ 2
                = 2 * (2 * (1 + C.sup' hC (fun c => ‖c‖)) / δ) *
                    ((codeBound C hC + Real.log C.card) * Cc / δ)
                  + ((codeBound C hC + Real.log C.card) * Cc / δ) ^ 2 := by
              field_simp; ring
            rw [e]
            have := mul_le_mul ha' hb' hb0 (by positivity)
            nlinarith [pow_le_pow_left₀ hb0 hb' 2]


/-- The constant in the error term of the derivative bound. -/
noncomputable def errA (C : Finset E) (hC : C.Nonempty) (Cc : ℝ) : ℝ :=
  4 * (1 + C.sup' hC (fun c => ‖c‖)) * ((codeBound C hC + Real.log C.card) * Cc)
    + ((codeBound C hC + Real.log C.card) * Cc) ^ 2

theorem errA_nonneg (C : Finset E) (hC : C.Nonempty) {Cc : ℝ} (hCc : 0 ≤ Cc) : 0 ≤ errA C hC Cc := by
  have := sup_norm_nonneg C hC
  have := codeBound_nonneg C hC
  have : 0 ≤ Real.log C.card := Real.log_nonneg (by exact_mod_cast hC.card_pos)
  unfold errA; positivity

/-- **The variance of the cut-off smooth minimum**:
`Var g ≤ 4 (E f_C + |C| ε)/(d − 2) + (A/δ²) γ(B(0, 2δ))`. -/
theorem variance_gApprox_le (hd : 3 ≤ Module.finrank ℝ E) (C : Finset E) (hC : C.Nonempty)
    {Cc : ℝ} (hCc : ∀ δ : ℝ, 0 < δ → ∀ z : E, ‖fderiv ℝ (cutoff δ) z‖ ≤ Cc / δ) {ε δ : ℝ}
    (hε : 0 < ε) (hε1 : ε ≤ 1) (hδ : 0 < δ) :
    variance (gApprox C ε δ) (gaussianE E)
      ≤ 4 * ((∫ x, fC C hC x ∂(gaussianE E)) + C.card * ε) / ((Module.finrank ℝ E : ℝ) - 2)
        + errA C hC Cc / δ ^ 2 * (gaussianE E (Metric.ball (0 : E) (2 * δ))).toReal := by
  have hd2 : 2 ≤ Module.finrank ℝ E := by omega
  have hd' : (0 : ℝ) < (Module.finrank ℝ E : ℝ) - 2 := by
    have : (3 : ℝ) ≤ Module.finrank ℝ E := by exact_mod_cast hd
    linarith
  have hg := differentiable_gApprox C hC hε hδ
  have hcard : (0 : ℝ) ≤ C.card := by positivity
  refine (variance_le_integral_norm_fderiv_sq hd2 hg (abs_gApprox_le C hC hε hε1 δ)
    (norm_fderiv_gApprox_le C hC hCc hε hε1 hδ)).trans ?_
  have hint1 : Integrable (fun z : E => 4 * (fC C hC z + C.card * ε) * (‖z‖ ^ 2)⁻¹) (gaussianE E) := by
    have : (fun z : E => 4 * (fC C hC z + C.card * ε) * (‖z‖ ^ 2)⁻¹)
        = fun z => 4 * (fC C hC z * (‖z‖ ^ 2)⁻¹) + (4 * (C.card * ε)) * (‖z‖ ^ 2)⁻¹ := by
      funext z; ring
    rw [this]
    exact ((integrable_fC_div_normSq hd C hC).const_mul _).add
      ((integrable_inv_normSq_gaussianE hd).const_mul _)
  have hint2 : Integrable ((Metric.ball (0 : E) (2 * δ)).indicator fun _ => errA C hC Cc / δ ^ 2)
      (gaussianE E) := (integrable_const _).indicator measurableSet_ball
  calc ∫ z, ‖fderiv ℝ (gApprox C ε δ) z‖ ^ 2 ∂(gaussianE E)
      ≤ ∫ z, (4 * (fC C hC z + C.card * ε) * (‖z‖ ^ 2)⁻¹
          + (Metric.ball (0 : E) (2 * δ)).indicator (fun _ => errA C hC Cc / δ ^ 2) z) ∂(gaussianE E) := by
        refine integral_mono (integrable_norm_fderiv_sq (norm_fderiv_gApprox_le C hC hCc hε hε1 hδ)
          measurable_id) (hint1.add hint2) fun z => ?_
        exact norm_fderiv_gApprox_sq_le C hC hCc hε hε1 hδ z
    _ = _ := by
        rw [integral_add hint1 hint2, integral_indicator measurableSet_ball, setIntegral_const,
          smul_eq_mul, measureReal_def, mul_comm (errA C hC Cc / δ ^ 2)]
        congr 1
        have : (fun z : E => 4 * (fC C hC z + C.card * ε) * (‖z‖ ^ 2)⁻¹)
            = fun z => 4 * (fC C hC z * (‖z‖ ^ 2)⁻¹) + (4 * (C.card * ε)) * (‖z‖ ^ 2)⁻¹ := by
          funext z; ring
        rw [this, integral_add ((integrable_fC_div_normSq hd C hC).const_mul _)
          ((integrable_inv_normSq_gaussianE hd).const_mul _), integral_const_mul, integral_const_mul,
          integral_fC_div_normSq hd C hC, integral_inv_normSq_gaussianE hd]
        first | (field_simp; ring) | field_simp

/-- The `L²` distance from `f_C` to the cut-off smooth minimum. -/
theorem integral_sq_sub_gApprox_le (C : Finset E) (hC : C.Nonempty) {ε δ : ℝ} (hε : 0 < ε)
    (hε1 : ε ≤ 1) (hδ : 0 < δ) :
    ∫ z, (fC C hC z - gApprox C ε δ z) ^ 2 ∂(gaussianE E)
      ≤ 2 * (ε * Real.log C.card) ^ 2
        + 2 * (codeBound C hC + Real.log C.card) ^ 2 *
            (gaussianE E (Metric.ball (0 : E) (2 * δ))).toReal := by
  have hlog : 0 ≤ Real.log C.card := Real.log_nonneg (by exact_mod_cast hC.card_pos)
  have hM := codeBound_nonneg C hC
  set M₁ := codeBound C hC + Real.log C.card with hM₁
  have hpt : ∀ z, (fC C hC z - gApprox C ε δ z) ^ 2
      ≤ 2 * (ε * Real.log C.card) ^ 2
        + (Metric.ball (0 : E) (2 * δ)).indicator (fun _ => 2 * M₁ ^ 2) z := by
    intro z
    have h1 := abs_smoothMin_sub_fC_le C hC hε z
    have h2 := abs_smoothMin_le C hC hε hε1 z
    have hsplit : fC C hC z - gApprox C ε δ z
        = (fC C hC z - smoothMin C ε z) + (1 - cutoff δ z) * smoothMin C ε z := by
      unfold gApprox; ring
    rw [hsplit]
    have hc0 := cutoff_nonneg δ z
    have hc1 := cutoff_le_one δ z
    have key : ∀ a b : ℝ, (a + b) ^ 2 ≤ 2 * a ^ 2 + 2 * b ^ 2 := fun a b => by
      nlinarith [sq_nonneg (a - b)]
    refine (key _ _).trans (add_le_add ?_ ?_)
    · have : (fC C hC z - smoothMin C ε z) ^ 2 ≤ (ε * Real.log C.card) ^ 2 := by
        rw [← sq_abs, abs_sub_comm]
        exact pow_le_pow_left₀ (abs_nonneg _) h1 2
      linarith
    · rcases lt_or_ge ‖z‖ (2 * δ) with hz | hz
      · rw [Set.indicator_of_mem (by rw [Metric.mem_ball, dist_zero_right]; exact hz)]
        have : ((1 - cutoff δ z) * smoothMin C ε z) ^ 2 ≤ M₁ ^ 2 := by
          rw [← sq_abs, abs_mul, abs_of_nonneg (by linarith)]
          refine pow_le_pow_left₀ (by positivity) ?_ 2
          calc (1 - cutoff δ z) * |smoothMin C ε z| ≤ 1 * M₁ :=
                mul_le_mul (by linarith) h2 (abs_nonneg _) zero_le_one
            _ = M₁ := one_mul _
        nlinarith [sq_nonneg M₁]
      · rw [Set.indicator_of_notMem (by rw [Metric.mem_ball, dist_zero_right]; exact not_lt.mpr hz)]
        have hone : cutoff δ z = 1 := cutoff_eq_one hδ (by nlinarith)
        rw [hone, sub_self, zero_mul]; simp
  have hint : Integrable (fun z : E => 2 * (ε * Real.log C.card) ^ 2
      + (Metric.ball (0 : E) (2 * δ)).indicator (fun _ => 2 * M₁ ^ 2) z) (gaussianE E) :=
    (integrable_const _).add ((integrable_const _).indicator measurableSet_ball)
  have hmeas : Measurable fun z : E => (fC C hC z - gApprox C ε δ z) ^ 2 :=
    ((measurable_fC C hC).sub (differentiable_gApprox C hC hε hδ).continuous.measurable).pow_const 2
  calc ∫ z, (fC C hC z - gApprox C ε δ z) ^ 2 ∂(gaussianE E)
      ≤ ∫ z, (2 * (ε * Real.log C.card) ^ 2
          + (Metric.ball (0 : E) (2 * δ)).indicator (fun _ => 2 * M₁ ^ 2) z) ∂(gaussianE E) := by
        refine integral_mono_of_nonneg (Filter.Eventually.of_forall fun z => sq_nonneg _) hint
          (Filter.Eventually.of_forall hpt)
    _ = _ := by
        rw [integral_add (integrable_const _) ((integrable_const _).indicator measurableSet_ball),
          integral_const, integral_indicator measurableSet_ball, setIntegral_const, smul_eq_mul,
          smul_eq_mul, measureReal_def, measureReal_def, measure_univ, ENNReal.toReal_one, one_mul]
        ring

/-- **The sharp Gaussian Poincaré inequality for the squared distance to a finite codebook**:
`Var_γ f_C ≤ 4/(d − 2) · E_γ f_C`. -/
theorem variance_fC_le_sharp (hd : 3 ≤ Module.finrank ℝ E) (C : Finset E) (hC : C.Nonempty) :
    variance (fC C hC) (gaussianE E)
      ≤ 4 / ((Module.finrank ℝ E : ℝ) - 2) * ∫ x, fC C hC x ∂(gaussianE E) := by
  obtain ⟨Cc, hCc0, hCc⟩ := exists_bound_fderiv_cutoff (E := E)
  have hd' : (0 : ℝ) < (Module.finrank ℝ E : ℝ) - 2 := by
    have : (3 : ℝ) ≤ Module.finrank ℝ E := by exact_mod_cast hd
    linarith
  set I := ∫ x, fC C hC x ∂(gaussianE E) with hI
  set d := Module.finrank ℝ E with hd_def
  set L := Real.log C.card with hL
  set M₁ := codeBound C hC + L with hM₁
  set A := errA C hC Cc with hA
  set B := ballConst E with hB
  have hB0 : 0 ≤ B := ballConst_nonneg
  have hA0 : 0 ≤ A := errA_nonneg C hC hCc0
  have hlog : 0 ≤ L := Real.log_nonneg (by exact_mod_cast hC.card_pos)
  have hM1 : 0 ≤ M₁ := by have := codeBound_nonneg C hC; linarith
  have hcard : (0 : ℝ) ≤ C.card := by positivity
  -- the bound for `ε = δ ∈ (0, 1]` and any `η > 0`
  have hbound : ∀ η : ℝ, 0 < η → ∀ ε : ℝ, 0 < ε → ε ≤ 1 →
      variance (fC C hC) (gaussianE E)
        ≤ (1 + η) * (4 * (I + C.card * ε) / ((d : ℝ) - 2) + A * B * 2 ^ d * ε ^ (d - 2))
          + (1 + 1 / η) * (2 * (ε * L) ^ 2 + 2 * M₁ ^ 2 * (B * 2 ^ d * ε ^ d)) := by
    intro η hη ε hε hε1
    have hball : (gaussianE E (Metric.ball (0 : E) (2 * ε))).toReal ≤ B * 2 ^ d * ε ^ d := by
      have := gaussianE_ball_le (E := E) (r := 2 * ε) (by positivity)
      rw [mul_pow] at this
      linarith [this]
    have hball0 : 0 ≤ (gaussianE E (Metric.ball (0 : E) (2 * ε))).toReal := ENNReal.toReal_nonneg
    have h1 := variance_gApprox_le hd C hC hCc hε hε1 hε
    have h2 := integral_sq_sub_gApprox_le C hC hε hε1 hε
    have h3 := variance_le_of_close (μ := gaussianE E) (measurable_fC C hC)
      (differentiable_gApprox C hC hε hε).continuous.measurable
      (fun z => by rw [abs_of_nonneg (fC_nonneg C hC z)]; exact fC_le C hC z)
      (abs_gApprox_le C hC hε hε1 ε) hη
    have hε2 : A / ε ^ 2 * (B * 2 ^ d * ε ^ d) = A * B * 2 ^ d * ε ^ (d - 2) := by
      have hd3 : d = (d - 2) + 2 := by omega
      rw [hd3, pow_add, Nat.add_sub_cancel]
      field_simp
      ring
    have hvarg : variance (gApprox C ε ε) (gaussianE E)
        ≤ 4 * (I + C.card * ε) / ((d : ℝ) - 2) + A * B * 2 ^ d * ε ^ (d - 2) := by
      refine h1.trans (add_le_add le_rfl ?_)
      rw [← hε2]
      exact mul_le_mul_of_nonneg_left hball (by positivity)
    have hL2 : ∫ z, (fC C hC z - gApprox C ε ε z) ^ 2 ∂(gaussianE E)
        ≤ 2 * (ε * L) ^ 2 + 2 * M₁ ^ 2 * (B * 2 ^ d * ε ^ d) := by
      refine h2.trans (add_le_add le_rfl ?_)
      exact mul_le_mul_of_nonneg_left hball (by positivity)
    have hvar0 := variance_nonneg (gApprox C ε ε) (gaussianE E)
    calc variance (fC C hC) (gaussianE E) ≤ _ := h3
      _ ≤ _ := add_le_add (mul_le_mul_of_nonneg_left hvarg (by linarith))
          (mul_le_mul_of_nonneg_left hL2 (by positivity))
  -- let `ε → 0⁺`
  have hlim : ∀ η : ℝ, 0 < η →
      variance (fC C hC) (gaussianE E) ≤ (1 + η) * (4 * I / ((d : ℝ) - 2)) := by
    intro η hη
    have hd2 : 1 ≤ d - 2 := by omega
    have hcont : Continuous fun ε : ℝ =>
        (1 + η) * (4 * (I + C.card * ε) / ((d : ℝ) - 2) + A * B * 2 ^ d * ε ^ (d - 2))
          + (1 + 1 / η) * (2 * (ε * L) ^ 2 + 2 * M₁ ^ 2 * (B * 2 ^ d * ε ^ d)) := by fun_prop
    have htend := hcont.tendsto 0
    have hval : (1 + η) * (4 * (I + C.card * 0) / ((d : ℝ) - 2) + A * B * 2 ^ d * 0 ^ (d - 2))
          + (1 + 1 / η) * (2 * (0 * L) ^ 2 + 2 * M₁ ^ 2 * (B * 2 ^ d * 0 ^ d))
        = (1 + η) * (4 * I / ((d : ℝ) - 2)) := by
      rw [zero_pow (by omega), zero_pow (by omega)]; ring
    rw [hval] at htend
    refine ge_of_tendsto (tendsto_nhdsWithin_of_tendsto_nhds (s := Set.Ioi 0) htend) ?_
    filter_upwards [Ioc_mem_nhdsGT (zero_lt_one' ℝ)] with ε hε
    exact hbound η hη ε hε.1 hε.2
  -- let `η → 0⁺`
  rw [show (4 : ℝ) / ((d : ℝ) - 2) * I = 4 * I / ((d : ℝ) - 2) by ring]
  have hcont2 : Continuous fun η : ℝ => (1 + η) * (4 * I / ((d : ℝ) - 2)) := by fun_prop
  have htend2 := hcont2.tendsto 0
  simp only [add_zero, one_mul] at htend2
  refine ge_of_tendsto (tendsto_nhdsWithin_of_tendsto_nhds (s := Set.Ioi 0) htend2) ?_
  filter_upwards [self_mem_nhdsWithin] with η hη
  exact hlim η hη

/-- **The sharp variance bound for a finite codebook on the sphere**:
`Var_σ ‖u − q u‖² ≤ 4 ρ²/(d − 2)`. -/
theorem variance_stageErr_le_finite_sharp (hd : 3 ≤ Module.finrank ℝ E) {q : E → E} (C : Finset E)
    (hC : C.Nonempty) (hq : IsNearest (C : Set E) q) (hmeas : Measurable q) :
    variance (fun u => ‖u - q u‖ ^ 2) (sphereUniformAmbient E)
      ≤ 4 / ((Module.finrank ℝ E : ℝ) - 2) * stageDistortion q (sphereUniformAmbient E) := by
  have hF : Measurable fun x : E => ‖x - q x‖ ^ 2 := (measurable_id.sub hmeas).norm.pow_const 2
  have hcomp : (fun x : E => ‖x - q x‖ ^ 2) ∘ dir = fC C hC := by
    funext z; simp only [Function.comp]; rw [fC_eq_of_isNearest C hC hq]
  have hint : ∫ x, ‖dir x - q (dir x)‖ ^ 2 ∂(gaussianE E) = ∫ x, fC C hC x ∂(gaussianE E) :=
    integral_congr_ae (Filter.Eventually.of_forall fun z => (fC_eq_of_isNearest C hC hq z).symm)
  rw [← gaussianE_map_dir, variance_map (by rw [gaussianE_map_dir]; exact hF.aemeasurable)
    measurable_dir.aemeasurable, hcomp, stageDistortion, integral_map measurable_dir.aemeasurable
    (by rw [gaussianE_map_dir]; exact hF.aestronglyMeasurable), hint]
  exact variance_fC_le_sharp hd C hC

end SharpAssembly

section StreamFiniteSharp

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] [FiniteDimensional ℝ E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G] [MeasurableMul G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {q : E → E} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G}
  [κ.IsMulRightInvariant] {x₀ : E}

theorem variance_stageErrSq_le_finite_sharp (hd : 3 ≤ Module.finrank ℝ E)
    (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1) (C : Finset E) (hC : C.Nonempty)
    (hq : IsNearest (C : Set E) q) (p : ℕ) :
    variance (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p) μ
      ≤ 4 / ((Module.finrank ℝ E : ℝ) - 2) * stageDistortion q (sphereUniformAmbient E) := by
  haveI : Nontrivial E := ⟨⟨x₀, 0, by rintro rfl; simp at hx₀⟩⟩
  have hmp := measurePreserving_unitVec_stream h hx₀ p
  have hF : Measurable fun x : E => ‖x - q x‖ ^ 2 := (measurable_id.sub h.meas_q).norm.pow_const 2
  have heq : (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p)
      = (fun x : E => ‖x - q x‖ ^ 2) ∘ (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p) := rfl
  rw [heq, ← variance_map (by rw [hmp.map_eq]; exact hF.aemeasurable) hmp.measurable.aemeasurable,
    hmp.map_eq]
  exact variance_stageErr_le_finite_sharp hd C hC hq h.meas_q

/-- **Concentration of the decay law for a finite codebook, with the sharp constant.**
The `P`-pass squared error lies within the compounded band `(1 ± ε)^P ρ^{2P}‖w‖²` except on an
event of probability at most `4P/((d − 2) ε² ρ²)`. -/
theorem meas_not_decayWithin_le_finite_sharp (hd : 3 ≤ Module.finrank ℝ E)
    (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1) (C : Finset E) (hC : C.Nonempty)
    (hq : IsNearest (C : Set E) q) {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε ≤ 1)
    (hρ : 0 < stageDistortion q (sphereUniformAmbient E)) (P : ℕ) :
    μ {ω | ¬ DecayWithin (rotScheme ι q (R ω)) (W ω)
        (Real.sqrt (stageDistortion q (sphereUniformAmbient E))) ε P}
      ≤ P * ENNReal.ofReal (4 / (((Module.finrank ℝ E : ℝ) - 2) * ε ^ 2 *
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
      ≤ 4 / ((Module.finrank ℝ E : ℝ) - 2) * D := fun p _ =>
    variance_stageErrSq_le_finite_sharp hd h hx₀ C hC hq p
  have hρne : Real.sqrt D ≠ 0 := (Real.sqrt_pos.mpr hρ).ne'
  have key := meas_not_decayWithin_le_of_variance hε0 hε1 hρne hmem hmean hvar
  have hD0 : D ≠ 0 := hρ.ne'
  have hε : ε ≠ 0 := hε0.ne'
  have hd' : ((Module.finrank ℝ E : ℝ) - 2) ≠ 0 := by
    have : (3 : ℝ) ≤ Module.finrank ℝ E := by exact_mod_cast hd
    linarith
  have hconst : 4 / ((Module.finrank ℝ E : ℝ) - 2) * D / (ε * Real.sqrt D ^ 2) ^ 2
      = 4 / (((Module.finrank ℝ E : ℝ) - 2) * ε ^ 2 * D) := by
    rw [hρ2]; field_simp
  rw [← hconst]; exact key

end StreamFiniteSharp
