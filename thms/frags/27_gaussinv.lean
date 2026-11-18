/-! ## §27  Gaussian invariance, moments, and the variance identity

Groundwork for proving the Poincaré inequality of §26 outright, for the one function the paper
needs.  The Gaussian measure is invariant under linear isometries (`gaussianE_map_isometry`),
its second moments are `E‖Z‖² = d` and `E⟨Z, v⟩² = ‖v‖²` (`lintegral_normSq_gaussianE`,
`lintegral_inner_sq_gaussianE`), and the product of two copies is the Gaussian measure of the
`ℓ²` product (`gaussianE_prod_map_ofLp`), so the rotation
`(x, y) ↦ (cos θ·x + sin θ·y, −sin θ·x + cos θ·y)` preserves `γ ⊗ γ` (`rotMap`, `inner_rotMap`).
The variance identity `Var f = ½ E(f(X) − f(Y))²` for independent copies
(`variance_eq_half_integral_prod`) and the pointwise derivative bound from a two-point Lipschitz
inequality (`norm_fderiv_le_of_twoPoint`) complete the toolkit. -/

open Metric
open scoped InnerProductSpace

section GaussianInvariance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

theorem gaussDensity_isometry (L : E ≃ₗᵢ[ℝ] E) (x : E) : gaussDensity (L x) = gaussDensity x := by
  simp [gaussDensity, L.norm_map]

/-- The unnormalised Gaussian is invariant under linear isometries. -/
theorem gaussRaw_map_isometry (L : E ≃ₗᵢ[ℝ] E) : (gaussRaw E).map L = gaussRaw E := by
  ext s hs
  rw [Measure.map_apply L.continuous.measurable hs, gaussRaw,
    withDensity_apply _ (L.continuous.measurable hs), withDensity_apply _ hs,
    ← L.measurePreserving.setLIntegral_comp_preimage_emb L.toHomeomorph.measurableEmbedding
      gaussDensity s]
  exact setLIntegral_congr_fun (L.continuous.measurable hs) fun x _ =>
    (gaussDensity_isometry L x).symm

/-- **The Gaussian measure is invariant under linear isometries.** -/
theorem gaussianE_map_isometry (L : E ≃ₗᵢ[ℝ] E) : (gaussianE E).map L = gaussianE E := by
  rw [gaussianE, Measure.map_smul, gaussRaw_map_isometry]

theorem measurePreserving_isometry_gaussianE (L : E ≃ₗᵢ[ℝ] E) :
    MeasurePreserving L (gaussianE E) (gaussianE E) :=
  ⟨L.continuous.measurable, gaussianE_map_isometry L⟩

end GaussianInvariance

section GaussianMoments

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- `E‖Z‖² = d` for the standard Gaussian, from the radial recursion. -/
theorem lintegral_normSq_gaussianE :
    ∫⁻ x, ENNReal.ofReal (‖x‖ ^ 2) ∂(gaussianE E) = (Module.finrank ℝ E : ℝ≥0∞) := by
  set d := Module.finrank ℝ E with hd_def
  have hd : 1 ≤ d := Module.finrank_pos
  have hφ : Measurable fun r : ℝ => ENNReal.ofReal (r ^ 2) := by fun_prop
  have h := lintegral_gaussRaw_polar (E := E) (ψ := fun _ => 1) measurable_const hφ
  simp only [one_mul] at h
  rw [lintegral_one] at h
  have hrad : ∫⁻ r in Set.Ioi (0 : ℝ), ENNReal.ofReal (r ^ (d - 1)) *
      (ENNReal.ofReal (r ^ 2) * ENNReal.ofReal (Real.exp (-(r ^ 2) / 2)))
      = ENNReal.ofReal (radGauss (d + 1)) := by
    rw [← lintegral_radial]
    refine setLIntegral_congr_fun measurableSet_Ioi fun r hr => ?_
    rw [← mul_assoc, ← ENNReal.ofReal_mul (pow_nonneg hr.le _), ← pow_add]
    congr 3
    omega
  have hrec : radGauss (d + 1) = (d : ℝ) * radGauss (d - 1) := by
    have h2 := radGauss_succ_succ (d - 1)
    rw [show d - 1 + 2 = d + 1 by omega] at h2
    rw [h2, Nat.cast_sub hd, Nat.cast_one]
    ring
  rw [gaussianE, lintegral_smul_measure, smul_eq_mul, h, hrad, hrec, gaussRaw_univ,
    ENNReal.ofReal_mul (Nat.cast_nonneg d), ENNReal.ofReal_natCast]
  set A := volume.toSphere (Set.univ : Set (sphere (0 : E) 1))
  set B := ENNReal.ofReal (radGauss (d - 1))
  have hA0 : A ≠ 0 := toSphere_univ_ne_zero E
  have hAt : A ≠ ⊤ := measure_ne_top _ _
  have hB0 : B ≠ 0 := by simp [B, radGauss_pos]
  have hBt : B ≠ ⊤ := ENNReal.ofReal_ne_top
  rw [ENNReal.mul_inv (Or.inl hA0) (Or.inl hAt)]
  calc A⁻¹ * B⁻¹ * (A * ((d : ℝ≥0∞) * B)) = (d : ℝ≥0∞) * ((A⁻¹ * A) * (B⁻¹ * B)) := by ring
    _ = (d : ℝ≥0∞) := by
      rw [ENNReal.inv_mul_cancel hA0 hAt, ENNReal.inv_mul_cancel hB0 hBt, mul_one, mul_one]

end GaussianMoments


section GaussianMoments2

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

theorem integrable_sq_sphericalCoordLaw {d : ℕ} (hd : 2 ≤ d) :
    Integrable (fun t : ℝ => t ^ 2) (sphericalCoordLaw d) := by
  haveI := isProbabilityMeasure_sphericalCoordLaw hd
  have hsupp : ∀ᵐ t ∂(sphericalCoordLaw d), t ∈ Set.Icc (-1 : ℝ) 1 :=
    (withDensity_absolutelyContinuous _ _).ae_le (ae_restrict_mem measurableSet_Icc)
  refine (MemLp.of_bound (by fun_prop : Measurable fun t : ℝ => t ^ 2).aestronglyMeasurable 1
    (hsupp.mono fun t ht => ?_)).integrable le_rfl
  rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg t)]
  nlinarith [ht.1, ht.2]

/-- `E⟨v, u⟩² = ‖v‖²/d` for `u` uniform on the sphere. -/
theorem lintegral_inner_sq_sphereUniform (hd : 2 ≤ Module.finrank ℝ E) (v : E) :
    ∫⁻ u, ENNReal.ofReal (⟪v, u⟫_ℝ ^ 2) ∂(sphereUniformAmbient E)
      = ENNReal.ofReal (‖v‖ ^ 2 / Module.finrank ℝ E) := by
  rcases eq_or_ne v 0 with rfl | hv
  · simp
  have hv' : 0 < ‖v‖ := norm_pos_iff.mpr hv
  set e : E := ‖v‖⁻¹ • v with he
  have hen : ‖e‖ = 1 := by
    rw [he, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hv'.ne']
  have hinner : ∀ u : E, ⟪v, u⟫_ℝ = ‖v‖ * ⟪e, u⟫_ℝ := by
    intro u
    rw [he, real_inner_smul_left, ← mul_assoc, mul_inv_cancel₀ hv'.ne', one_mul]
  have hlaw := sphereUniformAmbient_map_inner hd hen
  have hmeas : Measurable fun x : E => ⟪e, x⟫_ℝ := (continuous_const.inner continuous_id).measurable
  have h1 : ∫⁻ u, ENNReal.ofReal (⟪e, u⟫_ℝ ^ 2) ∂(sphereUniformAmbient E)
      = ENNReal.ofReal (1 / Module.finrank ℝ E) := by
    rw [← lintegral_map (f := fun t : ℝ => ENNReal.ofReal (t ^ 2)) (by fun_prop) hmeas, hlaw,
      ← ofReal_integral_eq_lintegral_ofReal (integrable_sq_sphericalCoordLaw hd)
        (Filter.Eventually.of_forall fun t => sq_nonneg t), integral_sq_sphericalCoordLaw hd]
  simp_rw [hinner, mul_pow, ENNReal.ofReal_mul (sq_nonneg _)]
  rw [lintegral_const_mul _ (by fun_prop), h1, ← ENNReal.ofReal_mul (sq_nonneg _)]
  congr 1; ring

/-- **`E⟨v, Z⟩² = ‖v‖²`** for the standard Gaussian. -/
theorem lintegral_inner_sq_gaussianE (hd : 2 ≤ Module.finrank ℝ E) (v : E) :
    ∫⁻ x, ENNReal.ofReal (⟪v, x⟫_ℝ ^ 2) ∂(gaussianE E) = ENNReal.ofReal (‖v‖ ^ 2) := by
  set d := Module.finrank ℝ E with hd_def
  have hφ : Measurable fun r : ℝ => ENNReal.ofReal (r ^ 2) := by fun_prop
  have hψ : Measurable fun u : E => ENNReal.ofReal (⟪v, u⟫_ℝ ^ 2) := by fun_prop
  have hpt : ∀ x : E, ENNReal.ofReal (⟪v, x⟫_ℝ ^ 2)
      = ENNReal.ofReal (⟪v, dir x⟫_ℝ ^ 2) * ENNReal.ofReal (‖x‖ ^ 2) := by
    intro x
    rcases eq_or_ne x 0 with rfl | hx
    · simp
    have hn : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
    rw [← ENNReal.ofReal_mul (sq_nonneg _)]
    congr 1
    rw [dir, real_inner_smul_right, mul_pow, inv_pow]
    field_simp
  rw [lintegral_congr hpt]
  have h := lintegral_gaussRaw_polar (E := E) hψ hφ
  have hrad : ∫⁻ r in Set.Ioi (0 : ℝ), ENNReal.ofReal (r ^ (d - 1)) *
      (ENNReal.ofReal (r ^ 2) * ENNReal.ofReal (Real.exp (-(r ^ 2) / 2)))
      = ENNReal.ofReal (radGauss (d + 1)) := by
    rw [← lintegral_radial]
    refine setLIntegral_congr_fun measurableSet_Ioi fun r hr => ?_
    rw [← mul_assoc, ← ENNReal.ofReal_mul (pow_nonneg hr.le _), ← pow_add]
    congr 3
    have : 1 ≤ d := Module.finrank_pos
    omega
  have hrec : radGauss (d + 1) = (d : ℝ) * radGauss (d - 1) := by
    have hd1 : 1 ≤ d := Module.finrank_pos
    have h2 := radGauss_succ_succ (d - 1)
    rw [show d - 1 + 2 = d + 1 by omega] at h2
    rw [h2, Nat.cast_sub hd1, Nat.cast_one]
    ring
  have hsph := lintegral_inner_sq_sphereUniform hd v
  rw [sphereUniformAmbient, lintegral_map hψ measurable_subtype_coe, sphereUniform,
    lintegral_smul_measure, smul_eq_mul] at hsph
  rw [gaussianE, lintegral_smul_measure, smul_eq_mul, h, hrad, hrec, gaussRaw_univ,
    ENNReal.ofReal_mul (Nat.cast_nonneg d), ENNReal.ofReal_natCast]
  set A := volume.toSphere (Set.univ : Set (sphere (0 : E) 1))
  set B := ENNReal.ofReal (radGauss (d - 1))
  set I := ∫⁻ u : sphere (0 : E) 1, ENNReal.ofReal (⟪v, (u : E)⟫_ℝ ^ 2) ∂volume.toSphere
  have hA0 : A ≠ 0 := toSphere_univ_ne_zero E
  have hAt : A ≠ ⊤ := measure_ne_top _ _
  have hB0 : B ≠ 0 := by simp [B, radGauss_pos]
  have hBt : B ≠ ⊤ := ENNReal.ofReal_ne_top
  have hdpos : (0 : ℝ) < d := by exact_mod_cast (show 0 < d by omega)
  rw [ENNReal.mul_inv (Or.inl hA0) (Or.inl hAt)]
  calc A⁻¹ * B⁻¹ * (I * ((d : ℝ≥0∞) * B)) = (A⁻¹ * I) * (d : ℝ≥0∞) * (B⁻¹ * B) := by ring
    _ = ENNReal.ofReal (‖v‖ ^ 2 / d) * (d : ℝ≥0∞) := by
        rw [ENNReal.inv_mul_cancel hB0 hBt, mul_one, hsph]
    _ = ENNReal.ofReal (‖v‖ ^ 2) := by
        rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (by positivity)]
        congr 1
        field_simp

end GaussianMoments2

section GaussianProd

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

instance : IsFiniteMeasure (gaussRaw E) := ⟨lt_top_iff_ne_top.mpr gaussRaw_univ_ne_top⟩

theorem gaussDensity_toLp (x y : E) :
    gaussDensity (WithLp.toLp 2 (x, y) : WithLp 2 (E × E)) = gaussDensity x * gaussDensity y := by
  simp only [gaussDensity]
  rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add, WithLp.prod_norm_sq_eq_of_L2]
  congr 2
  simp only [WithLp.fst, WithLp.snd]
  ring

/-- The unnormalised Gaussian on `E × E` is the product of the unnormalised Gaussians. -/
theorem gaussRaw_prod_map_ofLp :
    (gaussRaw (WithLp 2 (E × E))).map WithLp.ofLp = (gaussRaw E).prod (gaussRaw E) := by
  refine (Measure.prod_eq fun s t hs ht => ?_).symm
  rw [Measure.map_apply (WithLp.measurable_ofLp 2 _) (hs.prod ht), gaussRaw,
    withDensity_apply _ ((WithLp.measurable_ofLp 2 _) (hs.prod ht))]
  have hemb : MeasurableEmbedding (WithLp.ofLp : WithLp 2 (E × E) → E × E) :=
    (MeasurableEquiv.toLp 2 (E × E)).symm.measurableEmbedding
  have hpt : ∀ z : WithLp 2 (E × E), gaussDensity z
      = (fun p : E × E => gaussDensity p.1 * gaussDensity p.2) (WithLp.ofLp z) := by
    intro z
    have := gaussDensity_toLp (WithLp.ofLp z).1 (WithLp.ofLp z).2
    rwa [Prod.mk.eta, WithLp.toLp_ofLp] at this
  have hkey := (WithLp.volume_preserving_ofLp E E).setLIntegral_comp_preimage_emb hemb
    (fun p : E × E => gaussDensity p.1 * gaussDensity p.2) (s ×ˢ t)
  rw [setLIntegral_congr_fun ((WithLp.measurable_ofLp 2 _) (hs.prod ht)) (fun z _ => hpt z), hkey,
    volume_eq_prod, ← Measure.prod_restrict, lintegral_prod_mul measurable_gaussDensity.aemeasurable
      measurable_gaussDensity.aemeasurable, gaussRaw, withDensity_apply _ hs,
    withDensity_apply _ ht]

theorem gaussRaw_prod_univ :
    gaussRaw (WithLp 2 (E × E)) Set.univ = gaussRaw E Set.univ * gaussRaw E Set.univ := by
  have h : (gaussRaw (WithLp 2 (E × E))).map WithLp.ofLp Set.univ
      = (gaussRaw E).prod (gaussRaw E) Set.univ := by rw [gaussRaw_prod_map_ofLp]
  rw [Measure.map_apply (WithLp.measurable_ofLp 2 _) MeasurableSet.univ, Set.preimage_univ,
    ← Set.univ_prod_univ, Measure.prod_prod] at h
  exact h

/-- **The Gaussian pair is a product.** -/
theorem gaussianE_prod_map_ofLp :
    (gaussianE (WithLp 2 (E × E))).map WithLp.ofLp = (gaussianE E).prod (gaussianE E) := by
  rw [gaussianE, gaussianE, Measure.map_smul, gaussRaw_prod_map_ofLp, gaussRaw_prod_univ,
    Measure.prod_smul_left, Measure.prod_smul_right, smul_smul,
    ENNReal.mul_inv (Or.inl gaussRaw_univ_ne_zero) (Or.inl gaussRaw_univ_ne_top)]

end GaussianProd

section VarianceIdentity

variable {X : Type*} [MeasurableSpace X]

/-- `Var f = ½ ∬ (f x − f y)² dμ dμ` for a probability measure `μ`. -/
theorem variance_eq_half_integral_prod {μ : Measure X} [IsProbabilityMeasure μ] {f : X → ℝ}
    (hf : MemLp f 2 μ) :
    variance f μ = (1 / 2) * ∫ p : X × X, (f p.1 - f p.2) ^ 2 ∂(μ.prod μ) := by
  have hf1 : Integrable f μ := hf.integrable one_le_two
  have hf2 : Integrable (fun x => f x ^ 2) μ := hf.integrable_sq
  have h1 : ∫ p : X × X, (f p.1 - f p.2) ^ 2 ∂(μ.prod μ)
      = ∫ p : X × X, (f p.1 ^ 2 * (1 : ℝ) + (1 : ℝ) * f p.2 ^ 2 - 2 * (f p.1 * f p.2))
          ∂(μ.prod μ) :=
    integral_congr_ae (Filter.Eventually.of_forall fun p => by ring)
  have hA : ∫ a : X × X, f a.1 ^ 2 * (1 : ℝ) ∂(μ.prod μ)
      = (∫ x, f x ^ 2 ∂μ) * ∫ _y, (1 : ℝ) ∂μ := integral_prod_mul (fun x => f x ^ 2) (fun _ => (1 : ℝ))
  have hB : ∫ a : X × X, (1 : ℝ) * f a.2 ^ 2 ∂(μ.prod μ)
      = (∫ _x, (1 : ℝ) ∂μ) * ∫ y, f y ^ 2 ∂μ := integral_prod_mul (fun _ => (1 : ℝ)) (fun y => f y ^ 2)
  have hC : ∫ a : X × X, f a.1 * f a.2 ∂(μ.prod μ) = (∫ x, f x ∂μ) * ∫ y, f y ∂μ :=
    integral_prod_mul f f
  rw [h1, integral_sub, integral_add, hA, hB, integral_const_mul, hC, variance_eq_sub hf]
  · simp; ring
  · exact hf2.mul_prod (integrable_const (μ := μ) (1 : ℝ))
  · exact (integrable_const (μ := μ) (1 : ℝ)).mul_prod hf2
  · exact (hf2.mul_prod (integrable_const (μ := μ) (1 : ℝ))).add ((integrable_const (μ := μ) (1 : ℝ)).mul_prod hf2)
  · exact (hf1.mul_prod hf1).const_mul 2

end VarianceIdentity

section RotationPair

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- The rotation of a pair by angle `θ`, as a linear map on `E × E` with the `ℓ²` norm:
`(x, y) ↦ (cos θ · x + sin θ · y, −sin θ · x + cos θ · y)`. -/
noncomputable def rotMap (θ : ℝ) : WithLp 2 (E × E) →ₗ[ℝ] WithLp 2 (E × E) :=
  (WithLp.linearEquiv 2 ℝ (E × E)).symm.toLinearMap ∘ₗ
    (LinearMap.prod (Real.cos θ • LinearMap.fst ℝ E E + Real.sin θ • LinearMap.snd ℝ E E)
      (-Real.sin θ • LinearMap.fst ℝ E E + Real.cos θ • LinearMap.snd ℝ E E)) ∘ₗ
    (WithLp.linearEquiv 2 ℝ (E × E)).toLinearMap

theorem rotMap_apply (θ : ℝ) (p : WithLp 2 (E × E)) :
    rotMap θ p = WithLp.toLp 2 (Real.cos θ • p.fst + Real.sin θ • p.snd,
      -Real.sin θ • p.fst + Real.cos θ • p.snd) := rfl

theorem inner_rotMap (θ : ℝ) (p q : WithLp 2 (E × E)) : ⟪rotMap θ p, rotMap θ q⟫_ℝ = ⟪p, q⟫_ℝ := by
  rw [rotMap_apply, rotMap_apply, WithLp.prod_inner_apply, WithLp.prod_inner_apply]
  simp only [WithLp.fst, WithLp.snd, WithLp.ofLp_toLp, inner_add_left, inner_add_right,
    real_inner_smul_left, real_inner_smul_right, neg_smul, inner_neg_left, inner_neg_right]
  linear_combination (⟪p.ofLp.1, q.ofLp.1⟫_ℝ + ⟪p.ofLp.2, q.ofLp.2⟫_ℝ) * Real.cos_sq_add_sin_sq θ

/-- The pair rotation as a linear isometry equivalence. -/
noncomputable def rotPair (θ : ℝ) : WithLp 2 (E × E) ≃ₗᵢ[ℝ] WithLp 2 (E × E) :=
  LinearIsometry.toLinearIsometryEquiv
    ⟨rotMap θ, fun p => by rw [norm_eq_sqrt_real_inner, norm_eq_sqrt_real_inner, inner_rotMap]⟩ rfl

theorem rotPair_apply (θ : ℝ) (p : WithLp 2 (E × E)) : rotPair θ p = rotMap θ p := by
  rw [rotPair, LinearIsometry.coe_toLinearIsometryEquiv]; rfl

end RotationPair

section DerivBound

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A two-point Lipschitz bound with continuous slope function `G` bounds the derivative:
`‖f'(x)‖ ≤ 2 G x`. -/
theorem norm_fderiv_le_of_twoPoint {f G : E → ℝ} {x : E} (hx : x ≠ 0) (hG : ContinuousAt G x)
    (hGnn : 0 ≤ G x)
    (htwo : ∀ z w, z ≠ 0 → w ≠ 0 → |f z - f w| ≤ ‖z - w‖ * (G z + G w))
    {f' : E →L[ℝ] ℝ} (hf : HasFDerivAt f f' x) : ‖f'‖ ≤ 2 * G x := by
  refine le_of_forall_pos_le_add fun ε hε => ?_
  have hG' : ∀ᶠ z in 𝓝 x, G z < G x + ε / 2 := hG (Iio_mem_nhds (by linarith))
  have hne : ∀ᶠ z in 𝓝 x, z ≠ 0 := isOpen_ne.mem_nhds hx
  obtain ⟨s, hs, hsG⟩ := (hG'.and hne).exists_mem
  have hK : 0 ≤ 2 * G x + ε := by linarith
  have hlip : LipschitzOnWith ⟨2 * G x + ε, hK⟩ f s := by
    refine LipschitzOnWith.of_dist_le_mul fun z hz w hw => ?_
    rw [Real.dist_eq, dist_eq_norm]
    calc |f z - f w| ≤ ‖z - w‖ * (G z + G w) := htwo z w (hsG z hz).2 (hsG w hw).2
      _ ≤ ‖z - w‖ * (2 * G x + ε) := by
          have h1 := (hsG z hz).1; have h2 := (hsG w hw).1
          exact mul_le_mul_of_nonneg_left (by linarith) (norm_nonneg _)
      _ = ((⟨2 * G x + ε, hK⟩ : NNReal) : ℝ) * ‖z - w‖ := by simp [mul_comm]
  have := hf.le_of_lipschitzOn hs hlip
  calc ‖f'‖ ≤ ((⟨2 * G x + ε, hK⟩ : NNReal) : ℝ) := this
    _ = 2 * G x + ε := rfl

end DerivBound
