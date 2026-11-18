/-! ## §35  Bounded codebooks: the Gaussian Poincaré hypothesis discharged

§34 proved the sharp variance bound for finite codebooks.  A bounded codebook `C`, finite or
not, contains a finite `η`-net `C₀` (`exists_finite_net`), and then
`dist(u, C) ≤ dist(u, C₀) ≤ dist(u, C) + η`, so the squared distances to `C` and to `C₀` are
uniformly close on the sphere.  The comparison `Var f ≤ (1 + κ) Var g + (1 + 1/κ) E(f − g)²`
of §34 and the bound for `C₀` give, as `η → 0` and then `κ → 0`,
`Var_γ dist(Z/‖Z‖, C)² ≤ 4/(d − 2) · E dist(Z/‖Z‖, C)²` (`variance_infDist_sq_le_bounded`),
hence `Var_σ ‖u − q u‖² ≤ 4ρ²/(d − 2)` for every nearest-codeword stage with a bounded codebook
(`variance_stageErr_le_bounded`), and `meas_not_decayWithin_le_bounded` is the concentration
remark at rate `4P/((d − 2) ε² ρ²)` for every bounded codebook, with no unproved hypothesis.
This supersedes `meas_not_decayWithin_le_gauss` (§26): the hypothesis `GaussPoincare` is no
longer needed for any statement of the paper.  It remains unproved in its general two-point
form, and §25 and §26 keep it only as a record of the route the development took. -/

section InfiniteCodebook

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- `f_{C}` for a finite `C` is the squared distance from the direction to `C`. -/
theorem fC_eq_infDist (C : Finset E) (hC : C.Nonempty) (z : E) :
    fC C hC z = Metric.infDist (dir z) (C : Set E) ^ 2 := by
  obtain ⟨c₀, hc₀, hmin⟩ := Finset.exists_mem_eq_inf' hC (fun c => hc c z)
  have hfC : fC C hC z = hc c₀ z := hmin
  rw [hfC]
  unfold hc
  congr 1
  refine le_antisymm ?_ ?_
  · rw [Metric.infDist_eq_iInf]
    haveI : Nonempty (C : Set E) := ⟨⟨c₀, hc₀⟩⟩
    refine le_ciInf fun c => ?_
    rw [dist_eq_norm]
    have h : hc c₀ z ≤ hc (c : E) z := by
      rw [← hfC]; exact Finset.inf'_le _ (Finset.mem_coe.mp c.2)
    exact (pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).mp h
  · have := Metric.infDist_le_dist_of_mem (x := dir z) (Finset.mem_coe.mpr hc₀ : c₀ ∈ (C : Set E))
    rwa [dist_eq_norm] at this

theorem variance_infDist_sq_le_finite (hd : 3 ≤ Module.finrank ℝ E) (C : Finset E)
    (hC : C.Nonempty) :
    variance (fun z => Metric.infDist (dir z) (C : Set E) ^ 2) (gaussianE E)
      ≤ 4 / ((Module.finrank ℝ E : ℝ) - 2) *
          ∫ z, Metric.infDist (dir z) (C : Set E) ^ 2 ∂(gaussianE E) := by
  have e : (fun z => Metric.infDist (dir z) (C : Set E) ^ 2) = fC C hC :=
    funext fun z => (fC_eq_infDist C hC z).symm
  rw [e]; exact variance_fC_le_sharp hd C hC

/-- **A finite `η`-net inside a bounded set**, with the distance functions `η`-close. -/
theorem exists_finite_net {C : Set E} (hC : Bornology.IsBounded C) (hne : C.Nonempty) {η : ℝ}
    (hη : 0 < η) :
    ∃ C₀ : Finset E, C₀.Nonempty ∧ (C₀ : Set E) ⊆ C ∧
      ∀ x, Metric.infDist x (C₀ : Set E) ≤ Metric.infDist x C + η := by
  have htb : TotallyBounded C :=
    hC.isCompact_closure.totallyBounded.subset subset_closure
  obtain ⟨t, htC, htfin, hcov⟩ := htb.exists_subset (Metric.dist_mem_uniformity hη)
  refine ⟨htfin.toFinset, ?_, by simpa using htC, fun x => ?_⟩
  · obtain ⟨c, hc⟩ := hne
    have := hcov hc
    simp only [Set.mem_iUnion, Set.mem_setOf_eq] at this
    obtain ⟨y, hy, -⟩ := this
    exact ⟨y, htfin.mem_toFinset.mpr hy⟩
  · rw [Metric.infDist_eq_iInf (s := C)]
    haveI : Nonempty C := ⟨⟨_, hne.some_mem⟩⟩
    rw [← _root_.sub_le_iff_le_add]
    refine le_ciInf fun c => ?_
    have := hcov c.2
    simp only [Set.mem_iUnion, Set.mem_setOf_eq] at this
    obtain ⟨y, hy, hdy⟩ := this
    have h1 : Metric.infDist x (htfin.toFinset : Set E) ≤ dist x y :=
      Metric.infDist_le_dist_of_mem (by simpa using hy)
    have h2 : dist x y ≤ dist x c + dist (c : E) y := dist_triangle _ _ _
    linarith

theorem measurable_infDist_dir (C : Set E) : Measurable fun z : E => Metric.infDist (dir z) C :=
  (Metric.continuous_infDist_pt C).measurable.comp measurable_dir

theorem infDist_dir_le {C : Set E} {R : ℝ} (hR : ∀ c ∈ C, ‖c‖ ≤ R) {c₀ : E} (hc₀ : c₀ ∈ C) (z : E) :
    Metric.infDist (dir z) C ≤ 1 + R := by
  calc Metric.infDist (dir z) C ≤ dist (dir z) c₀ := Metric.infDist_le_dist_of_mem hc₀
    _ ≤ ‖dir z‖ + ‖c₀‖ := by rw [dist_eq_norm]; exact norm_sub_le _ _
    _ ≤ 1 + R := add_le_add (norm_dir_le_one z) (hR c₀ hc₀)

/-- **The sharp variance bound for a bounded codebook**, by finite nets:
`Var_γ dist(Z/‖Z‖, C)² ≤ 4/(d − 2) · E dist(Z/‖Z‖, C)²`. -/
theorem variance_infDist_sq_le_bounded (hd : 3 ≤ Module.finrank ℝ E) {C : Set E}
    (hC : Bornology.IsBounded C) (hne : C.Nonempty) :
    variance (fun z => Metric.infDist (dir z) C ^ 2) (gaussianE E)
      ≤ 4 / ((Module.finrank ℝ E : ℝ) - 2) *
          ∫ z, Metric.infDist (dir z) C ^ 2 ∂(gaussianE E) := by
  obtain ⟨R₀, hR₀⟩ := hC.exists_norm_le
  set R := max R₀ 0 with hRdef
  have hR : ∀ c ∈ C, ‖c‖ ≤ R := fun c hc => (hR₀ c hc).trans (le_max_left _ _)
  have hR0 : 0 ≤ R := le_max_right _ _
  have hd' : (0 : ℝ) < (Module.finrank ℝ E : ℝ) - 2 := by
    have : (3 : ℝ) ≤ Module.finrank ℝ E := by exact_mod_cast hd
    linarith
  set d := Module.finrank ℝ E with hd_def
  set f : E → ℝ := fun z => Metric.infDist (dir z) C ^ 2 with hf
  set I := ∫ z, f z ∂(gaussianE E) with hI
  have hfm : Measurable f := (measurable_infDist_dir C).pow_const 2
  have hfb : ∀ z, |f z| ≤ (1 + R) ^ 2 := fun z => by
    rw [hf, abs_of_nonneg (sq_nonneg _)]
    exact pow_le_pow_left₀ Metric.infDist_nonneg (infDist_dir_le hR hne.some_mem z) 2
  -- the bound with a net at scale `η ∈ (0, 1]` and comparison parameter `κ > 0`
  have hbound : ∀ κ : ℝ, 0 < κ → ∀ η : ℝ, 0 < η → η ≤ 1 →
      variance f (gaussianE E)
        ≤ (1 + κ) * (4 / ((d : ℝ) - 2) * (I + η * (2 * R + 3)))
          + (1 + 1 / κ) * (η * (2 * R + 3)) ^ 2 := by
    intro κ hκ η hη hη1
    obtain ⟨C₀, hC₀, hsub, hnet⟩ := exists_finite_net hC hne hη
    set f₀ : E → ℝ := fun z => Metric.infDist (dir z) (C₀ : Set E) ^ 2 with hf₀
    have hf₀m : Measurable f₀ := (measurable_infDist_dir _).pow_const 2
    have hR' : ∀ c ∈ (C₀ : Set E), ‖c‖ ≤ R := fun c hc => hR c (hsub hc)
    obtain ⟨c₀, hc₀⟩ := hC₀
    have hf₀b : ∀ z, |f₀ z| ≤ (1 + R) ^ 2 := fun z => by
      rw [hf₀, abs_of_nonneg (sq_nonneg _)]
      exact pow_le_pow_left₀ Metric.infDist_nonneg
        (infDist_dir_le hR' (Finset.mem_coe.mpr hc₀) z) 2
    -- pointwise closeness
    have hclose : ∀ z, |f z - f₀ z| ≤ η * (2 * R + 3) := fun z => by
      have ha := Metric.infDist_nonneg (x := dir z) (s := C)
      have hab : Metric.infDist (dir z) C ≤ Metric.infDist (dir z) (C₀ : Set E) :=
        Metric.infDist_le_infDist_of_subset hsub ⟨c₀, Finset.mem_coe.mpr hc₀⟩
      have hba := hnet (dir z)
      have hbnd := infDist_dir_le hR' (Finset.mem_coe.mpr hc₀) z
      rw [hf, hf₀]
      simp only
      rw [abs_le]
      constructor <;> nlinarith
    have hmean : ∫ z, f₀ z ∂(gaussianE E) ≤ I + η * (2 * R + 3) := by
      have hfi : Integrable f (gaussianE E) := integrable_of_bound_gauss hfm.aestronglyMeasurable _ hfb
      have hf₀i : Integrable f₀ (gaussianE E) :=
        integrable_of_bound_gauss hf₀m.aestronglyMeasurable _ hf₀b
      have : ∫ z, f₀ z ∂(gaussianE E) ≤ ∫ z, (f z + η * (2 * R + 3)) ∂(gaussianE E) :=
        integral_mono hf₀i (hfi.add (integrable_const _)) fun z => by
          have := (abs_le.mp (hclose z)).1; linarith
      rwa [integral_add hfi (integrable_const _), integral_const, measureReal_def, measure_univ,
        ENNReal.toReal_one, one_smul] at this
    have hvar₀ := variance_infDist_sq_le_finite hd C₀ ⟨c₀, hc₀⟩
    have hsq : ∫ z, (f z - f₀ z) ^ 2 ∂(gaussianE E) ≤ (η * (2 * R + 3)) ^ 2 := by
      have hi : Integrable (fun z => (f z - f₀ z) ^ 2) (gaussianE E) :=
        integrable_of_bound_gauss ((hfm.sub hf₀m).pow_const 2).aestronglyMeasurable
          ((η * (2 * R + 3)) ^ 2) fun z => by
            rw [abs_of_nonneg (sq_nonneg _), ← sq_abs]
            exact pow_le_pow_left₀ (abs_nonneg _) (hclose z) 2
      have := integral_mono hi (integrable_const _) fun z => by
        show (f z - f₀ z) ^ 2 ≤ (η * (2 * R + 3)) ^ 2
        rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) (hclose z) 2
      rwa [integral_const, measureReal_def, measure_univ, ENNReal.toReal_one, one_smul] at this
    have h3 := variance_le_of_close (μ := gaussianE E) hfm hf₀m hfb hf₀b hκ
    have hvar₀' : variance f₀ (gaussianE E) ≤ 4 / ((d : ℝ) - 2) * (I + η * (2 * R + 3)) :=
      hvar₀.trans (mul_le_mul_of_nonneg_left hmean (by positivity))
    calc variance f (gaussianE E) ≤ _ := h3
      _ ≤ _ := add_le_add (mul_le_mul_of_nonneg_left hvar₀' (by linarith))
          (mul_le_mul_of_nonneg_left hsq (by positivity))
  -- `η → 0⁺`
  have hlim : ∀ κ : ℝ, 0 < κ → variance f (gaussianE E) ≤ (1 + κ) * (4 / ((d : ℝ) - 2) * I) := by
    intro κ hκ
    have hcont : Continuous fun η : ℝ => (1 + κ) * (4 / ((d : ℝ) - 2) * (I + η * (2 * R + 3)))
        + (1 + 1 / κ) * (η * (2 * R + 3)) ^ 2 := by fun_prop
    have htend := hcont.tendsto 0
    simp only [zero_mul, add_zero, zero_pow two_ne_zero, mul_zero] at htend
    refine ge_of_tendsto (tendsto_nhdsWithin_of_tendsto_nhds (s := Set.Ioi 0) htend) ?_
    filter_upwards [Ioc_mem_nhdsGT (zero_lt_one' ℝ)] with η hη
    exact hbound κ hκ η hη.1 hη.2
  -- `κ → 0⁺`
  have hcont2 : Continuous fun κ : ℝ => (1 + κ) * (4 / ((d : ℝ) - 2) * I) := by fun_prop
  have htend2 := hcont2.tendsto 0
  simp only [add_zero, one_mul] at htend2
  refine ge_of_tendsto (tendsto_nhdsWithin_of_tendsto_nhds (s := Set.Ioi 0) htend2) ?_
  filter_upwards [self_mem_nhdsWithin] with κ hκ
  exact hlim κ hκ

/-- **The sharp variance bound on the sphere for any bounded codebook.** -/
theorem variance_stageErr_le_bounded (hd : 3 ≤ Module.finrank ℝ E) {q : E → E} {C : Set E}
    (hC : Bornology.IsBounded C) (hq : IsNearest C q) (hmeas : Measurable q) :
    variance (fun u => ‖u - q u‖ ^ 2) (sphereUniformAmbient E)
      ≤ 4 / ((Module.finrank ℝ E : ℝ) - 2) * stageDistortion q (sphereUniformAmbient E) := by
  have hne : C.Nonempty := ⟨q 0, (hq 0).1⟩
  have hF : Measurable fun x : E => ‖x - q x‖ ^ 2 := (measurable_id.sub hmeas).norm.pow_const 2
  have hcomp : (fun x : E => ‖x - q x‖ ^ 2) ∘ dir = fun z => Metric.infDist (dir z) C ^ 2 := by
    funext z
    show ‖dir z - q (dir z)‖ ^ 2 = Metric.infDist (dir z) C ^ 2
    rw [norm_sub_eq_infDist hq]
  have hint : ∫ x, ‖dir x - q (dir x)‖ ^ 2 ∂(gaussianE E)
      = ∫ z, Metric.infDist (dir z) C ^ 2 ∂(gaussianE E) :=
    integral_congr_ae (Filter.Eventually.of_forall fun z => by
      show ‖dir z - q (dir z)‖ ^ 2 = Metric.infDist (dir z) C ^ 2
      rw [norm_sub_eq_infDist hq])
  rw [← gaussianE_map_dir, variance_map (by rw [gaussianE_map_dir]; exact hF.aemeasurable)
    measurable_dir.aemeasurable, hcomp, stageDistortion, integral_map measurable_dir.aemeasurable
    (by rw [gaussianE_map_dir]; exact hF.aestronglyMeasurable), hint]
  exact variance_infDist_sq_le_bounded hd hC hne

end InfiniteCodebook

section StreamBounded

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] [FiniteDimensional ℝ E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G] [MeasurableMul G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {q : E → E} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G}
  [κ.IsMulRightInvariant] {x₀ : E}

theorem variance_stageErrSq_le_bounded (hd : 3 ≤ Module.finrank ℝ E)
    (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1) {C : Set E} (hC : Bornology.IsBounded C)
    (hq : IsNearest C q) (p : ℕ) :
    variance (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p) μ
      ≤ 4 / ((Module.finrank ℝ E : ℝ) - 2) * stageDistortion q (sphereUniformAmbient E) := by
  haveI : Nontrivial E := ⟨⟨x₀, 0, by rintro rfl; simp at hx₀⟩⟩
  have hmp := measurePreserving_unitVec_stream h hx₀ p
  have hF : Measurable fun x : E => ‖x - q x‖ ^ 2 := (measurable_id.sub h.meas_q).norm.pow_const 2
  have heq : (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p)
      = (fun x : E => ‖x - q x‖ ^ 2) ∘ (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p) := rfl
  rw [heq, ← variance_map (by rw [hmp.map_eq]; exact hF.aemeasurable) hmp.measurable.aemeasurable,
    hmp.map_eq]
  exact variance_stageErr_le_bounded hd hC hq h.meas_q

/-- **Concentration of the decay law for any bounded codebook, with the sharp constant and no
unproved hypothesis.**  This is `meas_not_decayWithin_le_gauss` with the Gaussian Poincaré
hypothesis discharged, at the better rate `4P/((d − 2) ε² ρ²)`. -/
theorem meas_not_decayWithin_le_bounded (hd : 3 ≤ Module.finrank ℝ E)
    (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1) {C : Set E} (hC : Bornology.IsBounded C)
    (hq : IsNearest C q) {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε ≤ 1)
    (hρ : 0 < stageDistortion q (sphereUniformAmbient E)) (P : ℕ) :
    μ {ω | ¬ DecayWithin (rotScheme ι q (R ω)) (W ω)
        (Real.sqrt (stageDistortion q (sphereUniformAmbient E))) ε P}
      ≤ P * ENNReal.ofReal (4 / (((Module.finrank ℝ E : ℝ) - 2) * ε ^ 2 *
          stageDistortion q (sphereUniformAmbient E))) := by
  haveI : Nontrivial E := ⟨⟨x₀, 0, by rintro rfl; simp at hx₀⟩⟩
  set D := stageDistortion q (sphereUniformAmbient E) with hD
  have hρ2 : Real.sqrt D ^ 2 = D := Real.sq_sqrt hρ.le
  have hmem : ∀ p < P, MemLp (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p) 2 μ := fun p _ =>
    (memLp_stageErr_of_bounded hq hC h.meas_q).comp_measurePreserving
      (measurePreserving_unitVec_stream h hx₀ p)
  have hmean : ∀ p < P, ∫ ω, stageErrSq (rotScheme ι q (R ω)) (W ω) p ∂μ = Real.sqrt D ^ 2 := by
    intro p _
    rw [hρ2, hD, ← h.orbitLaw_eq hx₀]
    exact integral_stageErr_eq_stageDistortion h p
  have hvar : ∀ p < P, variance (fun ω => stageErrSq (rotScheme ι q (R ω)) (W ω) p) μ
      ≤ 4 / ((Module.finrank ℝ E : ℝ) - 2) * D := fun p _ =>
    variance_stageErrSq_le_bounded hd h hx₀ hC hq p
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

end StreamBounded
