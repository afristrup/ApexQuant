/-! ## §26  Sphere Poincaré from Gaussian Poincaré

§25 takes the Poincaré inequality on the sphere, `SpherePoincare`, as a hypothesis.  This
section derives it, with constant `4/(d − 2)`, from the Poincaré inequality for the standard
Gaussian measure, stated as `GaussPoincare` in the same two-point Lipschitz form.  The mechanism
is the classical one: a function `F` of the direction is the `0`-homogeneous function
`f = F ∘ dir` of a Gaussian vector `Z`, whose slope at `z` is the sphere slope at `z/‖z‖` divided
by `‖z‖` (`norm_dir_sub_dir_le`), and `E[1/‖Z‖²] = 1/(d − 2)` because the direction of `Z` is
uniform and independent of the radius, whose law has the radial density `r^{d−1} e^{−r²/2}`.

Everything is proved with the polar machinery of §19: the Gaussian measure `gaussianE E` is
`e^{−‖x‖²/2} dx` normalised, its direction is uniform (`gaussianE_map_dir`), the weighted
integral `E[ψ(Z/‖Z‖)/‖Z‖²]` factorises (`lintegral_gaussianE_dir_div_normSq`), and the radial
integrals satisfy `R_{n+2} = (n+1) R_n` by integration by parts on `(0, ∞)`
(`radGauss_succ_succ`).  `GaussPoincare` is taken as a hypothesis here and proved in §39
(`gaussPoincare_gaussianE`).  With it, `meas_not_decayWithin_le_gauss` is the concentration
remark at rate `16 P/((d − 2) ε² ρ²)`. -/

open Filter
open scoped Topology

section RadialGauss

/-- The radial Gaussian integral `R_n = ∫₀^∞ rⁿ e^{−r²/2} dr`. -/
noncomputable def radGauss (n : ℕ) : ℝ :=
  ∫ r in Set.Ioi (0 : ℝ), r ^ n * Real.exp (-(r ^ 2) / 2)

theorem integrableOn_radGauss (n : ℕ) :
    IntegrableOn (fun r : ℝ => r ^ n * Real.exp (-(r ^ 2) / 2)) (Set.Ioi 0) := by
  have h := integrableOn_rpow_mul_exp_neg_mul_sq (b := (1 / 2 : ℝ)) (by norm_num) (s := n)
    (neg_one_lt_zero.trans_le (Nat.cast_nonneg n))
  refine h.congr_fun (fun r hr => ?_) measurableSet_Ioi
  show r ^ (n : ℝ) * Real.exp (-(1 / 2) * r ^ 2) = r ^ n * Real.exp (-(r ^ 2) / 2)
  rw [Real.rpow_natCast]
  congr 2
  ring

theorem radGauss_pos (n : ℕ) : 0 < radGauss n := by
  unfold radGauss
  rw [setIntegral_pos_iff_support_of_nonneg_ae]
  · have hsupp : (Function.support fun r : ℝ => r ^ n * Real.exp (-(r ^ 2) / 2)) ∩ Set.Ioi 0
        = Set.Ioi 0 := by
      ext r
      simp only [Set.mem_inter_iff, Function.mem_support, Set.mem_Ioi, ne_eq, mul_eq_zero,
        pow_eq_zero_iff', Real.exp_ne_zero, or_false, not_and, Decidable.not_not]
      constructor
      · exact fun h => h.2
      · intro hr; exact ⟨fun h => absurd h hr.ne', hr⟩
    rw [hsupp]; simp
  · exact (ae_restrict_mem measurableSet_Ioi).mono fun r hr =>
      mul_nonneg (pow_nonneg hr.le n) (Real.exp_pos _).le
  · exact integrableOn_radGauss n

/-- `rⁿ⁺¹ e^{−r²/2} → 0` at infinity. -/
theorem tendsto_pow_mul_exp_neg_sq (n : ℕ) :
    Tendsto (fun r : ℝ => r ^ n * Real.exp (-(r ^ 2) / 2)) atTop (𝓝 0) := by
  have h := Real.tendsto_pow_mul_exp_neg_atTop_nhds_zero n
  refine squeeze_zero' ((eventually_ge_atTop 0).mono fun r hr => ?_)
    ((eventually_ge_atTop 2).mono fun r hr => ?_) h
  · exact mul_nonneg (pow_nonneg hr n) (Real.exp_pos _).le
  · exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by nlinarith)) (pow_nonneg (by linarith) n)

/-- **The recursion `R_{n+2} = (n+1) R_n`**, by integration by parts on `(0, ∞)`. -/
theorem radGauss_succ_succ (n : ℕ) : radGauss (n + 2) = (n + 1) * radGauss n := by
  unfold radGauss
  have hu : ∀ r ∈ Set.Ioi (0 : ℝ), HasDerivAt (fun r : ℝ => r ^ (n + 1)) ((n + 1 : ℝ) * r ^ n) r := by
    intro r _
    simpa [Nat.cast_succ] using hasDerivAt_pow (n + 1) r
  have hv : ∀ r ∈ Set.Ioi (0 : ℝ), HasDerivAt (fun r : ℝ => -Real.exp (-(r ^ 2) / 2))
      (r * Real.exp (-(r ^ 2) / 2)) r := by
    intro r _
    have := (((hasDerivAt_pow 2 r).neg.div_const 2).exp).neg
    refine this.congr_deriv ?_
    norm_num <;> ring
  have h1 : IntegrableOn ((fun r : ℝ => r ^ (n + 1)) * fun r => r * Real.exp (-(r ^ 2) / 2))
      (Set.Ioi 0) :=
    (integrableOn_radGauss (n + 2)).congr_fun (fun r _ => by simp only [Pi.mul_apply]; ring)
      measurableSet_Ioi
  have h2 : IntegrableOn ((fun r : ℝ => (n + 1 : ℝ) * r ^ n) * fun r => -Real.exp (-(r ^ 2) / 2))
      (Set.Ioi 0) :=
    IntegrableOn.congr_fun (Integrable.const_mul (integrableOn_radGauss n) (-(n + 1 : ℝ)))
      (fun r _ => by simp only [Pi.mul_apply]; ring) measurableSet_Ioi
  have h0 : Tendsto ((fun r : ℝ => r ^ (n + 1)) * fun r => -Real.exp (-(r ^ 2) / 2)) (𝓝[>] 0)
      (𝓝 0) := by
    have hc : Continuous ((fun r : ℝ => r ^ (n + 1)) * fun r => -Real.exp (-(r ^ 2) / 2)) := by
      fun_prop
    have := (hc.tendsto 0).mono_left (nhdsWithin_le_nhds (s := Set.Ioi (0 : ℝ)))
    simpa using this
  have hinf : Tendsto ((fun r : ℝ => r ^ (n + 1)) * fun r => -Real.exp (-(r ^ 2) / 2)) atTop
      (𝓝 0) := by
    have := (tendsto_pow_mul_exp_neg_sq (n + 1)).neg
    rw [neg_zero] at this
    refine this.congr fun r => ?_
    simp only [Pi.mul_apply]; ring
  have key := integral_Ioi_mul_deriv_eq_deriv_mul hu hv h1 h2 h0 hinf
  rw [sub_zero, zero_sub] at key
  have hl : ∫ r in Set.Ioi (0 : ℝ), r ^ (n + 1) * (r * Real.exp (-(r ^ 2) / 2))
      = ∫ r in Set.Ioi (0 : ℝ), r ^ (n + 2) * Real.exp (-(r ^ 2) / 2) :=
    setIntegral_congr_fun measurableSet_Ioi fun r _ => by ring
  have hr : ∫ r in Set.Ioi (0 : ℝ), (n + 1 : ℝ) * r ^ n * -Real.exp (-(r ^ 2) / 2)
      = -((n + 1 : ℝ) * ∫ r in Set.Ioi (0 : ℝ), r ^ n * Real.exp (-(r ^ 2) / 2)) := by
    rw [← integral_const_mul, ← integral_neg]
    exact setIntegral_congr_fun measurableSet_Ioi fun r _ => by ring
  rw [hl, hr, neg_neg] at key
  exact key

end RadialGauss

section Direction

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- `‖z/‖z‖ − w/‖w‖‖ ≤ 2‖z − w‖/‖z‖` for `z ≠ 0`. -/
theorem norm_dir_sub_dir_le {z w : E} (hz : z ≠ 0) :
    ‖‖z‖⁻¹ • z - ‖w‖⁻¹ • w‖ ≤ 2 * ‖z - w‖ / ‖z‖ := by
  have hz' : 0 < ‖z‖ := norm_pos_iff.mpr hz
  have h1 : ‖‖z‖⁻¹ • z - ‖w‖⁻¹ • w‖ ≤ ‖‖z‖⁻¹ • z - ‖z‖⁻¹ • w‖ + ‖‖z‖⁻¹ • w - ‖w‖⁻¹ • w‖ :=
    norm_sub_le_norm_sub_add_norm_sub _ _ _
  have h2 : ‖‖z‖⁻¹ • z - ‖z‖⁻¹ • w‖ = ‖z - w‖ / ‖z‖ := by
    rw [← smul_sub, norm_smul, norm_inv, norm_norm, div_eq_inv_mul]
  have h3 : ‖‖z‖⁻¹ • w - ‖w‖⁻¹ • w‖ ≤ ‖z - w‖ / ‖z‖ := by
    rcases eq_or_ne w 0 with hw | hw
    · simp [hw]; positivity
    · have hw' : 0 < ‖w‖ := norm_pos_iff.mpr hw
      rw [← sub_smul, norm_smul, Real.norm_eq_abs]
      have hrew : |‖z‖⁻¹ - ‖w‖⁻¹| * ‖w‖ = |‖w‖ - ‖z‖| / ‖z‖ := by
        rw [show ‖z‖⁻¹ - ‖w‖⁻¹ = (‖w‖ - ‖z‖) / (‖z‖ * ‖w‖) by field_simp, abs_div,
          abs_of_pos (mul_pos hz' hw')]
        field_simp
      rw [hrew]
      gcongr
      rw [abs_sub_comm]
      exact abs_norm_sub_norm_le z w
  calc ‖‖z‖⁻¹ • z - ‖w‖⁻¹ • w‖
      ≤ ‖‖z‖⁻¹ • z - ‖z‖⁻¹ • w‖ + ‖‖z‖⁻¹ • w - ‖w‖⁻¹ • w‖ := h1
    _ ≤ ‖z - w‖ / ‖z‖ + ‖z - w‖ / ‖z‖ := add_le_add h2.le h3
    _ = 2 * ‖z - w‖ / ‖z‖ := by ring

end Direction


section GaussianMeasure

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- The unnormalised Gaussian density `e^{−‖x‖²/2}`. -/
noncomputable def gaussDensity (x : E) : ℝ≥0∞ := ENNReal.ofReal (Real.exp (-(‖x‖ ^ 2) / 2))

theorem measurable_gaussDensity : Measurable (gaussDensity : E → ℝ≥0∞) := by
  unfold gaussDensity; fun_prop

/-- The unnormalised Gaussian measure `e^{−‖x‖²/2} dx`. -/
noncomputable def gaussRaw (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [MeasurableSpace E] [BorelSpace E] [FiniteDimensional ℝ E] : Measure E :=
  volume.withDensity gaussDensity

/-- The direction map `x ↦ x/‖x‖` (sending `0` to `0`). -/
noncomputable def dir (x : E) : E := ‖x‖⁻¹ • x

theorem measurable_dir : Measurable (dir : E → E) := by unfold dir; fun_prop

/-- The radial integral in `lintegral` form. -/
theorem lintegral_radial (n : ℕ) :
    ∫⁻ r in Set.Ioi (0 : ℝ), ENNReal.ofReal (r ^ n) * ENNReal.ofReal (Real.exp (-(r ^ 2) / 2))
      = ENNReal.ofReal (radGauss n) := by
  rw [radGauss, ofReal_integral_eq_lintegral_ofReal (integrableOn_radGauss n)
    ((ae_restrict_mem measurableSet_Ioi).mono fun r hr =>
      mul_nonneg (pow_nonneg hr.le n) (Real.exp_pos _).le)]
  refine setLIntegral_congr_fun measurableSet_Ioi fun r hr => ?_
  rw [ENNReal.ofReal_mul (pow_nonneg hr.le n)]

/-- **Polar form of Gaussian integrals**: a function of the direction times a function of the
radius integrates to a product. -/
theorem lintegral_gaussRaw_polar {ψ : E → ℝ≥0∞} (hψ : Measurable ψ) {φ : ℝ → ℝ≥0∞}
    (hφ : Measurable φ) :
    ∫⁻ x, ψ (dir x) * φ ‖x‖ ∂(gaussRaw E)
      = (∫⁻ u : sphere (0 : E) 1, ψ u ∂volume.toSphere) *
          ∫⁻ r in Set.Ioi (0 : ℝ), ENNReal.ofReal (r ^ (Module.finrank ℝ E - 1)) *
            (φ r * ENNReal.ofReal (Real.exp (-(r ^ 2) / 2))) := by
  have hF : Measurable fun x : E => gaussDensity x * (ψ (dir x) * φ ‖x‖) :=
    measurable_gaussDensity.mul ((hψ.comp measurable_dir).mul (hφ.comp measurable_norm))
  rw [gaussRaw, lintegral_withDensity_eq_lintegral_mul _ measurable_gaussDensity
    (g := fun x : E => ψ (dir x) * φ ‖x‖) (by exact (hψ.comp measurable_dir).mul (hφ.comp measurable_norm))]
  simp only [Pi.mul_apply]
  rw [lintegral_eq_polar _ hF, ← lintegral_mul_const _ (by fun_prop)]
  refine lintegral_congr fun u => ?_
  rw [← lintegral_const_mul _ (by fun_prop)]
  refine setLIntegral_congr_fun measurableSet_Ioi fun r hr => ?_
  have hu : ‖(u : E)‖ = 1 := mem_sphere_zero_iff_norm.mp u.2
  have hn : ‖r • (u : E)‖ = r := by
    rw [norm_smul, hu, mul_one, Real.norm_eq_abs, abs_of_pos hr]
  rw [gaussDensity, dir, hn, inv_smul_smul₀ hr.ne']
  ring

/-- The total mass of the unnormalised Gaussian: `σ(S) · R_{d−1}`. -/
theorem gaussRaw_univ :
    gaussRaw E Set.univ = volume.toSphere (Set.univ : Set (sphere (0 : E) 1)) *
      ENNReal.ofReal (radGauss (Module.finrank ℝ E - 1)) := by
  have h := lintegral_gaussRaw_polar (E := E) (ψ := fun _ => 1) measurable_const
    (φ := fun _ => 1) measurable_const
  simp only [one_mul, mul_one] at h
  rw [lintegral_one, lintegral_one] at h
  rw [h, lintegral_radial]

theorem gaussRaw_univ_ne_zero : gaussRaw E Set.univ ≠ 0 := by
  rw [gaussRaw_univ]
  exact mul_ne_zero (toSphere_univ_ne_zero E) (by simp [radGauss_pos])

theorem gaussRaw_univ_ne_top : gaussRaw E Set.univ ≠ ⊤ := by
  rw [gaussRaw_univ]
  exact ENNReal.mul_ne_top (measure_ne_top _ _) ENNReal.ofReal_ne_top

/-- The standard Gaussian probability measure on `E`. -/
noncomputable def gaussianE (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [MeasurableSpace E] [BorelSpace E] [FiniteDimensional ℝ E] : Measure E :=
  (gaussRaw E Set.univ)⁻¹ • gaussRaw E

instance : IsProbabilityMeasure (gaussianE E) := by
  constructor
  rw [gaussianE, Measure.smul_apply, smul_eq_mul,
    ENNReal.inv_mul_cancel gaussRaw_univ_ne_zero gaussRaw_univ_ne_top]

/-- **The direction of a Gaussian vector is uniform on the sphere.** -/
theorem gaussianE_map_dir : (gaussianE E).map dir = sphereUniformAmbient E := by
  ext s hs
  have hψ : Measurable (s.indicator (1 : E → ℝ≥0∞)) := measurable_const.indicator hs
  have hL : gaussRaw E (dir ⁻¹' s)
      = ∫⁻ x, s.indicator (1 : E → ℝ≥0∞) (dir x) * (fun _ : ℝ => (1 : ℝ≥0∞)) ‖x‖
          ∂(gaussRaw E) := by
    rw [← lintegral_indicator_one (measurable_dir hs)]
    refine lintegral_congr fun x => ?_
    by_cases h : dir x ∈ s
    · simp [Set.indicator_apply, Set.mem_preimage, h]
    · simp [Set.indicator_apply, Set.mem_preimage, h]
  have hR : (sphereUniformAmbient E) s
      = (volume.toSphere (Set.univ : Set (sphere (0 : E) 1)))⁻¹ *
          ∫⁻ u : sphere (0 : E) 1, s.indicator (1 : E → ℝ≥0∞) u ∂volume.toSphere := by
    have hind : ∫⁻ u : sphere (0 : E) 1, (Subtype.val ⁻¹' s).indicator (1 : sphere (0 : E) 1 → ℝ≥0∞) u
        ∂volume.toSphere
        = ∫⁻ u : sphere (0 : E) 1, s.indicator (1 : E → ℝ≥0∞) u ∂volume.toSphere := by
      refine lintegral_congr fun u => ?_
      by_cases h : (u : E) ∈ s
      · simp [Set.indicator_apply, Set.mem_preimage, h]
      · simp [Set.indicator_apply, Set.mem_preimage, h]
    rw [sphereUniformAmbient, Measure.map_apply measurable_subtype_coe hs, sphereUniform,
      Measure.smul_apply, smul_eq_mul, ← lintegral_indicator_one (measurable_subtype_coe hs), hind]
  rw [Measure.map_apply measurable_dir hs, gaussianE, Measure.smul_apply, smul_eq_mul, hL,
    lintegral_gaussRaw_polar hψ measurable_const, hR, gaussRaw_univ]
  simp only [one_mul]
  rw [lintegral_radial]
  set A := volume.toSphere (Set.univ : Set (sphere (0 : E) 1))
  set B := ENNReal.ofReal (radGauss (Module.finrank ℝ E - 1))
  set I := ∫⁻ u : sphere (0 : E) 1, s.indicator (1 : E → ℝ≥0∞) u ∂volume.toSphere
  have hB0 : B ≠ 0 := by simp [B, radGauss_pos]
  have hBt : B ≠ ⊤ := ENNReal.ofReal_ne_top
  have hA0 : A ≠ 0 := toSphere_univ_ne_zero E
  have hAt : A ≠ ⊤ := measure_ne_top _ _
  rw [ENNReal.mul_inv (Or.inl hA0) (Or.inl hAt)]
  calc A⁻¹ * B⁻¹ * (I * B) = A⁻¹ * I * (B⁻¹ * B) := by ring
    _ = A⁻¹ * I := by rw [ENNReal.inv_mul_cancel hB0 hBt, mul_one]

theorem measurePreserving_dir : MeasurePreserving (dir : E → E) (gaussianE E) (sphereUniformAmbient E) :=
  ⟨measurable_dir, gaussianE_map_dir⟩

/-- **`E[ψ(Z/‖Z‖) / ‖Z‖²] = E_σ[ψ] / (d − 2)`** for a Gaussian `Z` in dimension `d ≥ 3`. -/
theorem lintegral_gaussianE_dir_div_normSq (hd : 3 ≤ Module.finrank ℝ E) {ψ : E → ℝ≥0∞}
    (hψ : Measurable ψ) :
    ∫⁻ x, ψ (dir x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹) ∂(gaussianE E)
      = (∫⁻ u, ψ u ∂(sphereUniformAmbient E)) *
          ENNReal.ofReal (1 / ((Module.finrank ℝ E : ℝ) - 2)) := by
  set d := Module.finrank ℝ E with hd_def
  have hφ : Measurable fun r : ℝ => ENNReal.ofReal ((r ^ 2)⁻¹) := by fun_prop
  have hrad : ∫⁻ r in Set.Ioi (0 : ℝ), ENNReal.ofReal (r ^ (d - 1)) *
      (ENNReal.ofReal ((r ^ 2)⁻¹) * ENNReal.ofReal (Real.exp (-(r ^ 2) / 2)))
      = ENNReal.ofReal (radGauss (d - 3)) := by
    rw [← lintegral_radial]
    refine setLIntegral_congr_fun measurableSet_Ioi fun r hr => ?_
    rw [← mul_assoc, ← ENNReal.ofReal_mul (pow_nonneg hr.le _)]
    congr 2
    have : d - 1 = (d - 3) + 2 := by omega
    rw [this, pow_add, mul_assoc, mul_inv_cancel₀ (pow_ne_zero 2 hr.ne'), mul_one]
  have hrec : radGauss (d - 1) = ((d : ℝ) - 2) * radGauss (d - 3) := by
    have h := radGauss_succ_succ (d - 3)
    rw [show d - 3 + 2 = d - 1 by omega] at h
    rw [h]
    congr 1
    rw [Nat.cast_sub (by omega)]
    push_cast
    ring
  have hL : ∫⁻ x, ψ (dir x) * ENNReal.ofReal ((‖x‖ ^ 2)⁻¹) ∂(gaussianE E)
      = (gaussRaw E Set.univ)⁻¹ *
          ((∫⁻ u : sphere (0 : E) 1, ψ u ∂volume.toSphere) * ENNReal.ofReal (radGauss (d - 3))) := by
    rw [gaussianE, lintegral_smul_measure, smul_eq_mul, lintegral_gaussRaw_polar hψ hφ, hrad]
  have hR : ∫⁻ u, ψ u ∂(sphereUniformAmbient E)
      = (volume.toSphere (Set.univ : Set (sphere (0 : E) 1)))⁻¹ *
          ∫⁻ u : sphere (0 : E) 1, ψ u ∂volume.toSphere := by
    rw [sphereUniformAmbient, lintegral_map hψ measurable_subtype_coe, sphereUniform,
      lintegral_smul_measure, smul_eq_mul]
  rw [hL, hR, gaussRaw_univ, hrec]
  set A := volume.toSphere (Set.univ : Set (sphere (0 : E) 1))
  set I := ∫⁻ u : sphere (0 : E) 1, ψ u ∂volume.toSphere
  set R := radGauss (d - 3)
  have hd2 : (0 : ℝ) < (d : ℝ) - 2 := by
    have : (3 : ℝ) ≤ d := by exact_mod_cast hd
    linarith
  have hR0 : 0 < R := radGauss_pos _
  have hA0 : A ≠ 0 := toSphere_univ_ne_zero E
  have hAt : A ≠ ⊤ := measure_ne_top _ _
  have hB : ENNReal.ofReal (((d : ℝ) - 2) * R) = ENNReal.ofReal ((d : ℝ) - 2) * ENNReal.ofReal R :=
    ENNReal.ofReal_mul hd2.le
  have hB0 : ENNReal.ofReal (((d : ℝ) - 2) * R) ≠ 0 := by
    simp [mul_pos hd2 hR0]
  have hBt : ENNReal.ofReal (((d : ℝ) - 2) * R) ≠ ⊤ := ENNReal.ofReal_ne_top
  have hinv : ENNReal.ofReal (1 / ((d : ℝ) - 2)) = (ENNReal.ofReal ((d : ℝ) - 2))⁻¹ := by
    rw [one_div, ENNReal.ofReal_inv_of_pos hd2]
  rw [ENNReal.mul_inv (Or.inl hA0) (Or.inl hAt), hinv, hB,
    ENNReal.mul_inv (Or.inl (ENNReal.ofReal_pos.mpr hd2).ne') (Or.inl ENNReal.ofReal_ne_top)]
  have hR0' : ENNReal.ofReal R ≠ 0 := by simp [hR0]
  calc A⁻¹ * ((ENNReal.ofReal ((d : ℝ) - 2))⁻¹ * (ENNReal.ofReal R)⁻¹) * (I * ENNReal.ofReal R)
      = A⁻¹ * I * (ENNReal.ofReal ((d : ℝ) - 2))⁻¹ * ((ENNReal.ofReal R)⁻¹ * ENNReal.ofReal R) := by
        ring
    _ = A⁻¹ * I * (ENNReal.ofReal ((d : ℝ) - 2))⁻¹ := by
        rw [ENNReal.inv_mul_cancel hR0' ENNReal.ofReal_ne_top, mul_one]

end GaussianMeasure

section GaussianCoordinates

open Orthogonal

/-- **A coordinate of the direction of a Gaussian vector has the spherical coordinate law.**
Composes `gaussianE_map_dir`, that the direction `dir Z` of a Gaussian vector `Z` is uniform on
the sphere, with `sphereUniformAmbient_map_coord`, that a coordinate of the uniform direction has
law `sphericalCoordLaw d` (§22): the pushforward of `gaussianE (Ed d)` along `x ↦ (dir x).ofLp i`
is `sphericalCoordLaw d`, for every coordinate `i` and every `d ≥ 2`.  This says nothing about the
radius `‖Z‖`; it is used only to divide by it. -/
theorem gaussianE_map_dir_coord {d : ℕ} (hd : 2 ≤ d) (i : Fin d) :
    (gaussianE (Ed d)).map (fun x => (dir x).ofLp i) = sphericalCoordLaw d := by
  haveI : Nontrivial (Ed d) := Module.nontrivial_of_finrank_pos (R := ℝ) (by
    have : (0 : ℕ) < d := by omega
    simpa [Ed] using this)
  have hmeas : Measurable (fun v : Ed d => v.ofLp i) :=
    (measurable_pi_apply i).comp (WithLp.measurable_ofLp 2 _)
  have h1 : (fun x : Ed d => (dir x).ofLp i) = (fun v : Ed d => v.ofLp i) ∘ dir := rfl
  rw [h1, ← Measure.map_map hmeas measurable_dir, gaussianE_map_dir]
  exact sphereUniformAmbient_map_coord hd i

end GaussianCoordinates

section GaussianPoincare

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E]

/-- **The Gaussian Poincaré inequality**, in the two-point Lipschitz form and with the origin
removed: for `f` with `|f x − f y| ≤ ‖x − y‖ (G x + G y)` off the origin and `G` continuous off
the origin (so that the pointwise Lipschitz constant of `f` is at most `2G` almost everywhere),
`Var_γ f ≤ ∫ (2G)² dγ`.  This is the classical inequality with constant `1` for the standard
Gaussian measure; the dimension restriction `d ≥ 2` is needed because a point has positive
capacity in dimension one. -/
structure GaussPoincare (γ : Measure E) : Prop where
  bound : 2 ≤ Module.finrank ℝ E → ∀ (f G : E → ℝ), ContinuousOn G {x | x ≠ 0} →
    (∀ x y, x ≠ 0 → y ≠ 0 → |f x - f y| ≤ ‖x - y‖ * (G x + G y)) →
    MemLp f 2 γ → Integrable (fun x => G x ^ 2) γ →
    variance f γ ≤ ∫ x, (2 * G x) ^ 2 ∂γ

theorem continuousOn_dir : ContinuousOn (dir : E → E) {x | x ≠ 0} := by
  unfold dir
  exact (continuous_norm.continuousOn.inv₀ fun x hx => norm_ne_zero_iff.mpr hx).smul
    continuousOn_id

/-- **Sphere Poincaré from Gaussian Poincaré.**  If the standard Gaussian on `E` satisfies the
Poincaré inequality, the uniform law on the sphere satisfies it with constant `4/(d − 2)`: a
function of the direction is a `0`-homogeneous function of the Gaussian vector, its slope is the
sphere slope divided by the radius, and `E[1/‖Z‖²] = 1/(d − 2)`. -/
theorem spherePoincare_of_gaussPoincare (hd : 3 ≤ Module.finrank ℝ E)
    (hG : GaussPoincare (gaussianE E)) :
    SpherePoincare (sphereUniformAmbient E) (4 / ((Module.finrank ℝ E : ℝ) - 2)) := by
  haveI : Nontrivial E := Module.nontrivial_of_finrank_pos (R := ℝ) (by omega)
  constructor
  intro F g hg hFg hF hg2
  set d := Module.finrank ℝ E with hd_def
  have hd2 : (0 : ℝ) < (d : ℝ) - 2 := by
    have : (3 : ℝ) ≤ d := by exact_mod_cast hd
    linarith
  -- pass to `|g|`
  set g' : E → ℝ := fun x => |g x| with hg'
  have hg'c : Continuous g' := hg.abs
  have hFg' : ∀ x y, |F x - F y| ≤ ‖x - y‖ * (g' x + g' y) := fun x y =>
    (hFg x y).trans (mul_le_mul_of_nonneg_left (add_le_add (le_abs_self _) (le_abs_self _))
      (norm_nonneg _))
  have hg2' : Integrable (fun x => g' x ^ 2) (sphereUniformAmbient E) := by
    simpa [g', sq_abs] using hg2
  have hsq : ∀ x, (2 * g x) ^ 2 = (2 * g' x) ^ 2 := fun x => by simp [g', mul_pow, sq_abs]
  -- the Gaussian lift
  set f : E → ℝ := F ∘ dir with hf
  set G : E → ℝ := fun z => 2 * g' (dir z) / ‖z‖ with hGdef
  have hdir := measurePreserving_dir (E := E)
  have htwo : ∀ z w : E, z ≠ 0 → w ≠ 0 → |f z - f w| ≤ ‖z - w‖ * (G z + G w) := by
    intro z w hz hw
    have hz' : 0 < ‖z‖ := norm_pos_iff.mpr hz
    have hw' : 0 < ‖w‖ := norm_pos_iff.mpr hw
    have h1 := hFg' (dir z) (dir w)
    have hzw : ‖dir z - dir w‖ ≤ 2 * ‖z - w‖ / ‖z‖ := norm_dir_sub_dir_le hz
    have hwz : ‖dir z - dir w‖ ≤ 2 * ‖z - w‖ / ‖w‖ :=
      calc ‖dir z - dir w‖ = ‖dir w - dir z‖ := norm_sub_rev _ _
        _ ≤ 2 * ‖w - z‖ / ‖w‖ := norm_dir_sub_dir_le hw
        _ = 2 * ‖z - w‖ / ‖w‖ := by rw [norm_sub_rev]
    have hg'z : 0 ≤ g' (dir z) := abs_nonneg _
    have hg'w : 0 ≤ g' (dir w) := abs_nonneg _
    calc |f z - f w| = |F (dir z) - F (dir w)| := rfl
      _ ≤ ‖dir z - dir w‖ * (g' (dir z) + g' (dir w)) := h1
      _ = ‖dir z - dir w‖ * g' (dir z) + ‖dir z - dir w‖ * g' (dir w) := by ring
      _ ≤ (2 * ‖z - w‖ / ‖z‖) * g' (dir z) + (2 * ‖z - w‖ / ‖w‖) * g' (dir w) :=
          add_le_add (mul_le_mul_of_nonneg_right hzw hg'z) (mul_le_mul_of_nonneg_right hwz hg'w)
      _ = ‖z - w‖ * (G z + G w) := by
          simp only [G]; field_simp
  have hGcont : ContinuousOn G {x | x ≠ 0} := by
    refine ContinuousOn.div (continuousOn_const.mul (hg'c.comp_continuousOn continuousOn_dir))
      continuous_norm.continuousOn fun x hx => norm_ne_zero_iff.mpr hx
  have hfmem : MemLp f 2 (gaussianE E) := hF.comp_measurePreserving hdir
  -- the Gaussian integral of `G²`, via the polar formula
  have hψ : Measurable fun x : E => ENNReal.ofReal (4 * g' x ^ 2) := by
    exact ENNReal.measurable_ofReal.comp (by fun_prop)
  have hG2 : ∀ z : E, ENNReal.ofReal (G z ^ 2)
      = ENNReal.ofReal (4 * g' (dir z) ^ 2) * ENNReal.ofReal ((‖z‖ ^ 2)⁻¹) := by
    intro z
    rw [← ENNReal.ofReal_mul (by positivity)]
    congr 1
    simp only [G]
    rw [div_pow, mul_pow, div_eq_mul_inv]
    ring
  have hlint : ∫⁻ z, ENNReal.ofReal (G z ^ 2) ∂(gaussianE E)
      = ENNReal.ofReal (∫ x, 4 * g' x ^ 2 ∂(sphereUniformAmbient E)) *
          ENNReal.ofReal (1 / ((d : ℝ) - 2)) := by
    simp_rw [hG2]
    rw [lintegral_gaussianE_dir_div_normSq hd hψ,
      ← ofReal_integral_eq_lintegral_ofReal (hg2'.const_mul 4)
        (Filter.Eventually.of_forall fun x => by positivity)]
  have hGmeas : Measurable G := by
    simp only [G]
    exact ((measurable_const.mul (hg'c.measurable.comp measurable_dir)).div measurable_norm)
  have hG2int : Integrable (fun z => G z ^ 2) (gaussianE E) := by
    refine ⟨(hGmeas.pow_const 2).aestronglyMeasurable, ?_⟩
    rw [hasFiniteIntegral_iff_ofReal (Filter.Eventually.of_forall fun z => sq_nonneg _), hlint]
    exact ENNReal.mul_lt_top ENNReal.ofReal_lt_top ENNReal.ofReal_lt_top
  have hint : ∫ z, (2 * G z) ^ 2 ∂(gaussianE E)
      = 4 / ((d : ℝ) - 2) * ∫ x, (2 * g' x) ^ 2 ∂(sphereUniformAmbient E) := by
    have h1 : ∫ z, (2 * G z) ^ 2 ∂(gaussianE E) = 4 * ∫ z, G z ^ 2 ∂(gaussianE E) := by
      rw [← integral_const_mul]
      exact integral_congr_ae (Filter.Eventually.of_forall fun z => by ring)
    have h2 : ∫ z, G z ^ 2 ∂(gaussianE E)
        = (∫ x, 4 * g' x ^ 2 ∂(sphereUniformAmbient E)) * (1 / ((d : ℝ) - 2)) := by
      rw [integral_eq_lintegral_of_nonneg_ae (Filter.Eventually.of_forall fun z => sq_nonneg _)
        (hGmeas.pow_const 2).aestronglyMeasurable, hlint, ← ENNReal.ofReal_mul
        (integral_nonneg fun x => by positivity), ENNReal.toReal_ofReal
        (mul_nonneg (integral_nonneg fun x => by positivity) (by positivity))]
    have h3 : ∫ x, (2 * g' x) ^ 2 ∂(sphereUniformAmbient E)
        = ∫ x, 4 * g' x ^ 2 ∂(sphereUniformAmbient E) :=
      integral_congr_ae (Filter.Eventually.of_forall fun x => by ring)
    rw [h1, h2, h3]; ring
  have hvar : variance F (sphereUniformAmbient E) = variance f (gaussianE E) := by
    rw [← hdir.map_eq, variance_map (by rw [hdir.map_eq]; exact hF.aestronglyMeasurable.aemeasurable)
      measurable_dir.aemeasurable]
  have key := hG.bound (by omega) f G hGcont htwo hfmem hG2int
  rw [hvar]
  simp_rw [hsq]
  rw [← hint]
  exact key

end GaussianPoincare

section GaussianConcentration

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] [FiniteDimensional ℝ E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G] [MeasurableMul G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {q : E → E} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G}
  [κ.IsMulRightInvariant] {x₀ : E}

/-- **Concentration of the decay law under the Gaussian Poincaré inequality.**  The `P`-pass
squared error lies within the compounded band `(1 ± ε)^P ρ^{2P}‖w‖²` except with probability at
most `16 P / ((d − 2) ε² ρ²)`. -/
theorem meas_not_decayWithin_le_gauss (hd : 3 ≤ Module.finrank ℝ E)
    (hG : GaussPoincare (gaussianE E)) (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1)
    {C : Set E} (hq : IsNearest C q) (hC : Bornology.IsBounded C) {ε : ℝ} (hε0 : 0 < ε)
    (hε1 : ε ≤ 1) (hρ : 0 < stageDistortion q (sphereUniformAmbient E)) (P : ℕ) :
    μ {ω | ¬ DecayWithin (rotScheme ι q (R ω)) (W ω)
        (Real.sqrt (stageDistortion q (sphereUniformAmbient E))) ε P}
      ≤ P * ENNReal.ofReal (16 / (((Module.finrank ℝ E : ℝ) - 2) * ε ^ 2 *
          stageDistortion q (sphereUniformAmbient E))) := by
  have key := meas_not_decayWithin_le_poincare h hx₀ (spherePoincare_of_gaussPoincare hd hG)
    hq hC hε0 hε1 hρ P
  have hd2 : ((Module.finrank ℝ E : ℝ) - 2) ≠ 0 := by
    have : (3 : ℝ) ≤ Module.finrank ℝ E := by exact_mod_cast hd
    linarith
  have hD : stageDistortion q (sphereUniformAmbient E) ≠ 0 := hρ.ne'
  have hε : ε ≠ 0 := hε0.ne'
  have heq : 4 * (4 / ((Module.finrank ℝ E : ℝ) - 2)) /
      (ε ^ 2 * stageDistortion q (sphereUniformAmbient E))
      = 16 / (((Module.finrank ℝ E : ℝ) - 2) * ε ^ 2 * stageDistortion q (sphereUniformAmbient E)) := by
    field_simp; ring
  rw [← heq]; exact key

end GaussianConcentration
