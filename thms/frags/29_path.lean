/-! ## §29  The path inequality for a finite codebook

Fix a finite codebook `C` and let `f_C(z) = min_c ‖z/‖z‖ − c‖²`, the squared distance from the
direction of `z` to `C`.  Along the Maurey–Pisier path `X_θ = cos θ·x + sin θ·y` from `x` to
`y`, with `Y_θ = −sin θ·x + cos θ·y` its derivative, the path formula of §28 gives
`f_C(y) − f_C(x) = ∫₀^{π/2} f_C'₊(X_θ)[Y_θ] dθ`, so by Cauchy–Schwarz on the interval
`(f_C(y) − f_C(x))² ≤ (π/2) ∫₀^{π/2} S_C(X_θ, Y_θ) dθ`, with `S_C(u, v)` the sum over the
active codewords at `u` of `(h_c'(u)[v])²` (`sq_sub_fC_le`).  The path avoids the origin for
`γ ⊗ γ`-almost every pair (`ae_good`), two codewords tie on a `γ`-null set
(`gaussianE_tie_null`), and `E_y (L y)² = ‖L‖²` for a linear form `L` and Gaussian `y`
(`lintegral_clm_sq_gaussianE`). -/

section CodewordFunction

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- `h_c(z) = ‖z/‖z‖ − c‖²`, the squared distance from the direction of `z` to the codeword `c`. -/
noncomputable def hc (c z : E) : ℝ := ‖dir z - c‖ ^ 2

theorem contDiffAt_dir {z : E} (hz : z ≠ 0) : ContDiffAt ℝ 1 (dir : E → E) z := by
  show ContDiffAt ℝ 1 (fun x : E => ‖x‖⁻¹ • x) z
  exact (((contDiffAt_id (𝕜 := ℝ)).norm ℝ hz).inv (norm_ne_zero_iff.mpr hz)).smul contDiffAt_id

theorem contDiffAt_hc (c : E) {z : E} (hz : z ≠ 0) : ContDiffAt ℝ 1 (hc c) z :=
  ((contDiffAt_dir hz).sub contDiffAt_const).norm_sq ℝ

theorem differentiableAt_hc (c : E) {z : E} (hz : z ≠ 0) : DifferentiableAt ℝ (hc c) z :=
  (contDiffAt_hc c hz).differentiableAt one_ne_zero

/-- The two-point Lipschitz bound for `h_c`, with slope function `2‖dir z − c‖/‖z‖`. -/
theorem abs_hc_sub_le (c : E) {z w : E} (hz : z ≠ 0) (hw : w ≠ 0) :
    |hc c z - hc c w| ≤ ‖z - w‖ * (2 * ‖dir z - c‖ / ‖z‖ + 2 * ‖dir w - c‖ / ‖w‖) := by
  have hz' : 0 < ‖z‖ := norm_pos_iff.mpr hz
  have hw' : 0 < ‖w‖ := norm_pos_iff.mpr hw
  have h1 : |hc c z - hc c w| ≤ ‖dir z - dir w‖ * (‖dir z - c‖ + ‖dir w - c‖) := by
    unfold hc
    rw [sq_sub_sq, abs_mul, abs_of_nonneg (add_nonneg (norm_nonneg _) (norm_nonneg _)), mul_comm,
      add_comm]
    refine mul_le_mul_of_nonneg_right ?_ (add_nonneg (norm_nonneg _) (norm_nonneg _))
    have := abs_norm_sub_norm_le (dir z - c) (dir w - c)
    rwa [sub_sub_sub_cancel_right] at this
  have hzw : ‖dir z - dir w‖ ≤ 2 * ‖z - w‖ / ‖z‖ := norm_dir_sub_dir_le hz
  have hwz : ‖dir z - dir w‖ ≤ 2 * ‖z - w‖ / ‖w‖ :=
    calc ‖dir z - dir w‖ = ‖dir w - dir z‖ := norm_sub_rev _ _
      _ ≤ 2 * ‖w - z‖ / ‖w‖ := norm_dir_sub_dir_le hw
      _ = 2 * ‖z - w‖ / ‖w‖ := by rw [norm_sub_rev]
  calc |hc c z - hc c w| ≤ ‖dir z - dir w‖ * (‖dir z - c‖ + ‖dir w - c‖) := h1
    _ = ‖dir z - dir w‖ * ‖dir z - c‖ + ‖dir z - dir w‖ * ‖dir w - c‖ := by ring
    _ ≤ (2 * ‖z - w‖ / ‖z‖) * ‖dir z - c‖ + (2 * ‖z - w‖ / ‖w‖) * ‖dir w - c‖ :=
        add_le_add (mul_le_mul_of_nonneg_right hzw (norm_nonneg _))
          (mul_le_mul_of_nonneg_right hwz (norm_nonneg _))
    _ = ‖z - w‖ * (2 * ‖dir z - c‖ / ‖z‖ + 2 * ‖dir w - c‖ / ‖w‖) := by
        field_simp

/-- The derivative of `h_c` is bounded by twice the distance to the codeword over the radius. -/
theorem norm_fderiv_hc_le (c : E) {z : E} (hz : z ≠ 0) :
    ‖fderiv ℝ (hc c) z‖ ≤ 2 * (2 * ‖dir z - c‖ / ‖z‖) := by
  have hcont : ContinuousAt (fun w : E => 2 * ‖dir w - c‖ / ‖w‖) z := by
    have hd : ContinuousAt (dir : E → E) z := (contDiffAt_dir hz).continuousAt
    exact (continuousAt_const.mul (hd.sub continuousAt_const).norm).div continuous_norm.continuousAt
      (norm_ne_zero_iff.mpr hz)
  exact norm_fderiv_le_of_twoPoint hz hcont (by positivity) (fun z w hz hw => abs_hc_sub_le c hz hw)
    (differentiableAt_hc c hz).hasFDerivAt

/-! ### The rotation path -/

/-- The point at angle `θ` on the great circle through `x` (at `θ = 0`) and `y` (at `θ = π/2`). -/
noncomputable def pathX (x y : E) (θ : ℝ) : E := Real.cos θ • x + Real.sin θ • y

/-- Its velocity, which is the rotated second point. -/
noncomputable def pathY (x y : E) (θ : ℝ) : E := -Real.sin θ • x + Real.cos θ • y

theorem hasDerivAt_pathX (x y : E) (θ : ℝ) : HasDerivAt (pathX x y) (pathY x y θ) θ := by
  unfold pathX pathY
  have h1 := (Real.hasDerivAt_cos θ).smul_const x
  have h2 := (Real.hasDerivAt_sin θ).smul_const y
  exact h1.add h2

theorem contDiff_pathX (x y : E) : ContDiff ℝ 1 (pathX x y) := by
  unfold pathX
  exact (Real.contDiff_cos.smul contDiff_const).add (Real.contDiff_sin.smul contDiff_const)

theorem pathX_zero (x y : E) : pathX x y 0 = x := by simp [pathX]

theorem pathX_pi_div_two (x y : E) : pathX x y (Real.pi / 2) = y := by simp [pathX]

/-- `Good x y`: the great circle through `x` and `y` avoids the origin. -/
def Good (x y : E) : Prop := ∀ θ, pathX x y θ ≠ 0

/-- Along a good path, `h_c` composed with the path is `C¹` with the chain-rule derivative. -/
theorem hasDerivAt_hc_pathX (c : E) {x y : E} (hg : Good x y) (θ : ℝ) :
    HasDerivAt (fun t => hc c (pathX x y t)) (fderiv ℝ (hc c) (pathX x y θ) (pathY x y θ)) θ :=
  (differentiableAt_hc c (hg θ)).hasFDerivAt.comp_hasDerivAt θ (hasDerivAt_pathX x y θ)

theorem contDiff_hc_pathX (c : E) {x y : E} (hg : Good x y) :
    ContDiff ℝ 1 (fun t => hc c (pathX x y t)) :=
  contDiff_iff_contDiffAt.mpr fun θ =>
    (contDiffAt_hc c (hg θ)).comp θ (contDiff_pathX x y).contDiffAt

theorem continuous_hc_pathX_deriv (c : E) {x y : E} (hg : Good x y) :
    Continuous fun θ => fderiv ℝ (hc c) (pathX x y θ) (pathY x y θ) := by
  have h := (contDiff_hc_pathX c hg).continuous_deriv le_rfl
  refine h.congr fun θ => ?_
  exact (hasDerivAt_hc_pathX c hg θ).deriv

end CodewordFunction

section GoodSet

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- A bad pair satisfies the Cauchy–Schwarz equality. -/
theorem abs_inner_eq_of_not_good {x y : E} (h : ¬ Good x y) : |⟪x, y⟫_ℝ| = ‖x‖ * ‖y‖ := by
  unfold Good at h
  push Not at h
  obtain ⟨θ, hθ⟩ := h
  unfold pathX at hθ
  rcases eq_or_ne (Real.sin θ) 0 with hs | hs
  · have hc : Real.cos θ ≠ 0 := by
      intro hc; have := Real.cos_sq_add_sin_sq θ; rw [hc, hs] at this; norm_num at this
    have hx : x = 0 := by
      rw [hs, zero_smul, add_zero] at hθ
      exact (smul_eq_zero.mp hθ).resolve_left hc
    simp [hx]
  · have hy : y = -(Real.cos θ / Real.sin θ) • x := by
      have : Real.sin θ • y = -(Real.cos θ • x) := by rw [← sub_eq_zero, sub_neg_eq_add, add_comm]; exact hθ
      rw [neg_smul, div_eq_inv_mul, mul_smul, ← smul_neg, ← this, smul_smul, inv_mul_cancel₀ hs,
        one_smul]
    rw [hy, real_inner_smul_right, real_inner_self_eq_norm_sq, abs_mul, abs_of_nonneg (sq_nonneg ‖x‖),
      norm_smul, Real.norm_eq_abs]
    ring

/-- The set of bad pairs is contained in a closed set. -/
theorem measurableSet_bad :
    MeasurableSet {p : E × E | |⟪p.1, p.2⟫_ℝ| = ‖p.1‖ * ‖p.2‖} :=
  measurableSet_eq_fun (by fun_prop) (by fun_prop)

theorem gaussianE_absolutelyContinuous : gaussianE E ≪ (volume : Measure E) :=
  (withDensity_absolutelyContinuous _ _).smul_left _

/-- For `x ≠ 0` and `d ≥ 2`, the line through `x` is Gaussian-null. -/
theorem gaussianE_span_singleton (hd : 2 ≤ Module.finrank ℝ E) (x : E) :
    gaussianE E (Submodule.span ℝ {x} : Set E) = 0 := by
  refine gaussianE_absolutelyContinuous (Measure.addHaar_submodule _ _ ?_)
  intro htop
  have h1 := finrank_span_le_card (R := ℝ) ({x} : Set E)
  rw [htop, finrank_top] at h1
  simp at h1
  omega

end GoodSet

section BadNull

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

theorem gaussianE_ae_ne_zero : ∀ᵐ x ∂(gaussianE E), x ≠ 0 := by
  rw [ae_iff]
  have : {x : E | ¬ x ≠ 0} = {0} := by ext x; simp
  rw [this]
  exact gaussianE_absolutelyContinuous (measure_singleton 0)

/-- **Almost every pair is good**: the great circle through two independent Gaussian vectors
avoids the origin. -/
theorem ae_good (hd : 2 ≤ Module.finrank ℝ E) :
    ∀ᵐ p ∂((gaussianE E).prod (gaussianE E)), Good p.1 p.2 := by
  rw [ae_iff]
  refine measure_mono_null (t := {p : E × E | |⟪p.1, p.2⟫_ℝ| = ‖p.1‖ * ‖p.2‖})
    (fun p hp => abs_inner_eq_of_not_good hp) ?_
  rw [Measure.prod_apply measurableSet_bad]
  refine (lintegral_congr_ae ?_).trans lintegral_zero
  filter_upwards [gaussianE_ae_ne_zero (E := E)] with x hx
  refine measure_mono_null (fun y hy => ?_) (gaussianE_span_singleton hd x)
  simp only [Set.mem_preimage, Set.mem_setOf_eq] at hy
  rcases eq_or_ne y 0 with rfl | hy0
  · exact Submodule.zero_mem _
  · have := (norm_inner_eq_norm_iff (𝕜 := ℝ) hx hy0).mp (by rw [Real.norm_eq_abs]; exact hy)
    obtain ⟨r, -, rfl⟩ := this
    exact Submodule.smul_mem _ r (Submodule.mem_span_singleton_self x)

/-- **Ties are null**: two distinct codewords are equidistant from the direction of a Gaussian
vector with probability zero. -/
theorem gaussianE_tie_null (hd : 2 ≤ Module.finrank ℝ E) {c c' : E} (hne : c ≠ c') :
    gaussianE E {z | hc c z = hc c' z} = 0 := by
  have hpos : 0 < ‖c' - c‖ := norm_pos_iff.mpr (sub_ne_zero.mpr hne.symm)
  set e : E := ‖c' - c‖⁻¹ • (c' - c) with he
  have hen : ‖e‖ = 1 := by
    rw [he, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hpos.ne']
  set t : ℝ := (‖c'‖ ^ 2 - ‖c‖ ^ 2) / (2 * ‖c' - c‖) with ht
  have hiff : ∀ u : E, ‖u - c‖ ^ 2 = ‖u - c'‖ ^ 2 ↔ ⟪e, u⟫_ℝ = t := by
    intro u
    rw [norm_sub_sq_real, norm_sub_sq_real, he, real_inner_smul_left, inner_sub_left, ht,
      real_inner_comm u c, real_inner_comm u c']
    constructor
    · intro h; field_simp; linarith
    · intro h; field_simp at h; linarith
  have hset : {z : E | hc c z = hc c' z} = dir ⁻¹' {u : E | ⟪e, u⟫_ℝ = t} := by
    ext z; simp only [Set.mem_setOf_eq, Set.mem_preimage, hc]; exact hiff (dir z)
  have hA : MeasurableSet {u : E | ⟪e, u⟫_ℝ = t} :=
    measurableSet_eq_fun (continuous_const.inner continuous_id).measurable measurable_const
  rw [hset, ← Measure.map_apply measurable_dir hA, gaussianE_map_dir]
  have hmeas : Measurable fun u : E => ⟪e, u⟫_ℝ := (continuous_const.inner continuous_id).measurable
  have : {u : E | ⟪e, u⟫_ℝ = t} = (fun u : E => ⟪e, u⟫_ℝ) ⁻¹' {t} := by ext u; simp
  rw [this, ← Measure.map_apply hmeas (measurableSet_singleton t), sphereUniformAmbient_map_inner hd hen,
    sphericalCoordLaw]
  have h0 : (volume.restrict (Set.Icc (-1 : ℝ) 1)) {t} = 0 :=
    le_antisymm ((Measure.le_iff'.1 Measure.restrict_le_self {t}).trans_eq (measure_singleton t))
      zero_le
  exact withDensity_absolutelyContinuous _ _ h0

end BadNull

section DerivInner

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- **`E_y (f' y)² = ‖f'‖²`** for a continuous linear functional and a Gaussian `y`. -/
theorem lintegral_clm_sq_gaussianE (hd : 2 ≤ Module.finrank ℝ E) (f' : E →L[ℝ] ℝ) :
    ∫⁻ y, ENNReal.ofReal (f' y ^ 2) ∂(gaussianE E) = ENNReal.ofReal (‖f'‖ ^ 2) := by
  set g : E := (InnerProductSpace.toDual ℝ E).symm f' with hg
  have h1 : ∀ y, f' y = ⟪g, y⟫_ℝ := fun y => (InnerProductSpace.toDual_symm_apply).symm
  have h2 : ‖f'‖ = ‖g‖ := by
    rw [hg, LinearIsometryEquiv.norm_map]
  simp_rw [h1, h2]
  exact lintegral_inner_sq_gaussianE hd g

end DerivInner

section RotProd

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

theorem measurable_rotProd (θ : ℝ) :
    Measurable fun p : E × E => (pathX p.1 p.2 θ, pathY p.1 p.2 θ) := by
  unfold pathX pathY; fun_prop

/-- **Rotation invariance of the Gaussian pair.** -/
theorem measurePreserving_rotProd (θ : ℝ) :
    MeasurePreserving (fun p : E × E => (pathX p.1 p.2 θ, pathY p.1 p.2 θ))
      ((gaussianE E).prod (gaussianE E)) ((gaussianE E).prod (gaussianE E)) := by
  refine ⟨measurable_rotProd θ, ?_⟩
  have hcomp : (fun p : E × E => (pathX p.1 p.2 θ, pathY p.1 p.2 θ)) ∘ (WithLp.ofLp : WithLp 2 (E × E) → E × E)
      = WithLp.ofLp ∘ rotPair θ := by
    funext z
    simp only [Function.comp, rotPair_apply, rotMap_apply, WithLp.ofLp_toLp, pathX, pathY]
    rfl
  rw [← gaussianE_prod_map_ofLp, Measure.map_map (measurable_rotProd θ) (WithLp.measurable_ofLp 2 _),
    hcomp, ← Measure.map_map (WithLp.measurable_ofLp 2 _) (rotPair θ).continuous.measurable,
    gaussianE_map_isometry]

end RotProd

section CauchySchwarzInterval

/-- `(∫ₐᵇ D)² ≤ (b − a) ∫ₐᵇ D²`, from the nonnegativity of a variance. -/
theorem sq_intervalIntegral_le {a b : ℝ} (hab : a < b) {D : ℝ → ℝ}
    (hD : MemLp D 2 (volume.restrict (Set.Ioc a b))) :
    (∫ t in a..b, D t) ^ 2 ≤ (b - a) * ∫ t in a..b, D t ^ 2 := by
  set ν : Measure ℝ := (ENNReal.ofReal (b - a))⁻¹ • volume.restrict (Set.Ioc a b) with hν
  have hba : 0 < b - a := sub_pos.mpr hab
  haveI : IsProbabilityMeasure ν := by
    constructor
    rw [hν, Measure.smul_apply, smul_eq_mul, Measure.restrict_apply MeasurableSet.univ,
      Set.univ_inter, Real.volume_Ioc, ENNReal.inv_mul_cancel (by simp [hab]) ENNReal.ofReal_ne_top]
  have hDν : MemLp D 2 ν := hD.smul_measure (by simp [hab])
  have hvar := ProbabilityTheory.variance_nonneg D ν
  rw [ProbabilityTheory.variance_eq_sub hDν] at hvar
  have hsm : ∀ f : ℝ → ℝ, ∫ t, f t ∂ν = (b - a)⁻¹ * ∫ t in a..b, f t := by
    intro f
    rw [hν, integral_smul_measure, intervalIntegral.integral_of_le hab.le, smul_eq_mul,
      ENNReal.toReal_inv, ENNReal.toReal_ofReal hba.le]
  rw [hsm, hsm] at hvar
  have hc : 0 < (b - a)⁻¹ := inv_pos.mpr hba
  have := sub_nonneg.mp hvar
  simp only [Pi.pow_apply, mul_pow] at this
  have hne : (b - a) ≠ 0 := hba.ne'
  calc (∫ t in a..b, D t) ^ 2
      = (b - a) ^ 2 * ((b - a)⁻¹ ^ 2 * (∫ t in a..b, D t) ^ 2) := by field_simp
    _ ≤ (b - a) ^ 2 * ((b - a)⁻¹ * ∫ t in a..b, D t ^ 2) := by gcongr
    _ = (b - a) * ∫ t in a..b, D t ^ 2 := by field_simp

end CauchySchwarzInterval

section PathIneq

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

theorem measurable_finset_inf'_fun {ι : Type*} {s : Finset ι} (hs : s.Nonempty) {φ : ι → E → ℝ}
    (h : ∀ i ∈ s, Measurable (φ i)) : Measurable fun z => s.inf' hs (fun i => φ i z) := by
  revert h
  induction hs using Finset.Nonempty.cons_induction with
  | singleton a => intro h; simpa using h a (Finset.mem_singleton_self a)
  | cons a s ha hs' ih =>
      intro h
      have hfun : (fun z => (Finset.cons a s ha).inf' ⟨a, Finset.mem_cons_self a s⟩ (fun k => φ k z))
          = fun z => min (φ a z) (s.inf' hs' (fun k => φ k z)) := by
        funext z; rw [Finset.inf'_cons hs']
      rw [hfun]
      exact (h a (Finset.mem_cons_self a s)).min (ih fun i hi => h i (Finset.mem_cons_of_mem hi))

/-- The codebook function on `E`: `f z = min_c h_c z = dist(z/‖z‖, C)²`. -/
noncomputable def fC (C : Finset E) (hC : C.Nonempty) (z : E) : ℝ := C.inf' hC (fun c => hc c z)

/-- The set of pairs `(u, v)` at which the codeword `c` is active at `u`. -/
def active (C : Finset E) (hC : C.Nonempty) (c : E) : Set (E × E) :=
  {p | hc c p.1 = fC C hC p.1}

/-- The active-codeword derivative sum `S(u, v) = ∑_c 1[c active at u] (h_c'(u) v)²`. -/
noncomputable def SC (C : Finset E) (hC : C.Nonempty) (u v : E) : ℝ :=
  ∑ c ∈ C, (active C hC c).indicator (fun p => fderiv ℝ (hc c) p.1 p.2 ^ 2) (u, v)

theorem fC_nonneg (C : Finset E) (hC : C.Nonempty) (z : E) : 0 ≤ fC C hC z :=
  Finset.le_inf' hC _ fun _ _ => sq_nonneg _

theorem SC_nonneg (C : Finset E) (hC : C.Nonempty) (u v : E) : 0 ≤ SC C hC u v :=
  Finset.sum_nonneg fun c _ => Set.indicator_nonneg (fun _ _ => sq_nonneg _) _

theorem measurable_hc (c : E) : Measurable (hc c) := by
  unfold hc
  exact ((measurable_dir).sub measurable_const).norm.pow_const 2

theorem measurable_fC (C : Finset E) (hC : C.Nonempty) : Measurable (fC C hC) := by
  unfold fC
  exact measurable_finset_inf'_fun hC fun c _ => measurable_hc c

theorem measurableSet_active (C : Finset E) (hC : C.Nonempty) (c : E) :
    MeasurableSet (active C hC c) :=
  measurableSet_eq_fun ((measurable_hc c).comp measurable_fst) ((measurable_fC C hC).comp measurable_fst)

theorem measurable_fderiv_hc_apply (c : E) :
    Measurable fun p : E × E => fderiv ℝ (hc c) p.1 p.2 ^ 2 := by
  have heval : Measurable fun q : (E →L[ℝ] ℝ) × E => q.1 q.2 :=
    (isBoundedBilinearMap_apply (𝕜 := ℝ) (E := E) (F := ℝ)).continuous.measurable
  exact (heval.comp (((measurable_fderiv ℝ (hc c)).comp measurable_fst).prodMk measurable_snd)).pow_const 2

theorem measurable_SC (C : Finset E) (hC : C.Nonempty) :
    Measurable fun p : E × E => SC C hC p.1 p.2 := by
  unfold SC
  exact Finset.measurable_sum _ fun c _ =>
    (measurable_fderiv_hc_apply c).indicator (measurableSet_active C hC c)

theorem measurable_pathPair (x y : E) : Measurable fun t : ℝ => (pathX x y t, pathY x y t) := by
  unfold pathX pathY; fun_prop

theorem measurable_SC_path (C : Finset E) (hC : C.Nonempty) (x y : E) :
    Measurable fun t : ℝ => SC C hC (pathX x y t) (pathY x y t) := by
  have h := Measurable.comp (g := fun p : E × E => SC C hC p.1 p.2)
    (f := fun t : ℝ => (pathX x y t, pathY x y t)) (measurable_SC C hC) (measurable_pathPair x y)
  exact h

set_option maxHeartbeats 1000000 in
/-- **The pointwise path inequality** for a good pair**, together with the integrability of the
active-derivative sum along the path. -/
theorem sq_sub_fC_le (C : Finset E) (hC : C.Nonempty) {x y : E} (hg : Good x y) :
    IntervalIntegrable (fun θ => SC C hC (pathX x y θ) (pathY x y θ)) volume 0 (Real.pi / 2) ∧
    (fC C hC y - fC C hC x) ^ 2
      ≤ (Real.pi / 2) * ∫ θ in (0 : ℝ)..(Real.pi / 2), SC C hC (pathX x y θ) (pathY x y θ) := by
  have hφ : ∀ c ∈ C, ∀ t, HasDerivAt (fun θ => hc c (pathX x y θ))
      ((fun c t => fderiv ℝ (hc c) (pathX x y t) (pathY x y t)) c t) t :=
    fun c _ t => hasDerivAt_hc_pathX c hg t
  have hφ' : ∀ c ∈ C, Continuous ((fun c t => fderiv ℝ (hc c) (pathX x y t) (pathY x y t)) c) :=
    fun c _ => continuous_hc_pathX_deriv c hg
  have hpi : (0 : ℝ) < Real.pi / 2 := by positivity
  have hpath := finset_inf'_sub_eq_integral (φ := fun c θ => hc c (pathX x y θ)) hC hpi.le hφ hφ'
  have hg0 : C.inf' hC (fun j => hc j (pathX x y 0)) = fC C hC x := by rw [pathX_zero]; rfl
  have hg1 : C.inf' hC (fun j => hc j (pathX x y (Real.pi / 2))) = fC C hC y := by
    rw [pathX_pi_div_two]; rfl
  have hD : ∀ t, derivWithin (fun u => C.inf' hC (fun j => hc j (pathX x y u))) (Ioi t) t ^ 2
      ≤ SC C hC (pathX x y t) (pathY x y t) := by
    intro t
    obtain ⟨c, hc', heq, hd, -⟩ := exists_derivWithin_Ioi_finset_inf' hC hφ t
    rw [hd]
    unfold SC
    have hterm : (active C hC c).indicator (fun p : E × E => fderiv ℝ (hc c) p.1 p.2 ^ 2)
        (pathX x y t, pathY x y t) = fderiv ℝ (hc c) (pathX x y t) (pathY x y t) ^ 2 :=
      Set.indicator_of_mem (show (pathX x y t, pathY x y t) ∈ active C hC c from heq) _
    rw [← hterm]
    exact Finset.single_le_sum (f := fun c => (active C hC c).indicator
      (fun p : E × E => fderiv ℝ (hc c) p.1 p.2 ^ 2) (pathX x y t, pathY x y t))
      (fun c _ => Set.indicator_nonneg (fun _ _ => sq_nonneg _) _) hc'
  have hbound : ∀ t, |derivWithin (fun u => C.inf' hC (fun j => hc j (pathX x y u))) (Ioi t) t|
      ≤ ∑ c ∈ C, |fderiv ℝ (hc c) (pathX x y t) (pathY x y t)| := by
    intro t
    obtain ⟨c, hc', -, hd, -⟩ := exists_derivWithin_Ioi_finset_inf' hC hφ t
    rw [hd]
    exact Finset.single_le_sum (f := fun c => |fderiv ℝ (hc c) (pathX x y t) (pathY x y t)|)
      (fun c _ => abs_nonneg _) hc'
  have hcontsum : Continuous fun t => ∑ c ∈ C, |fderiv ℝ (hc c) (pathX x y t) (pathY x y t)| :=
    continuous_finsetSum C fun c hc' => (hφ' c hc').abs
  obtain ⟨M, hM⟩ := (isCompact_Icc (a := (0 : ℝ)) (b := Real.pi / 2)).exists_bound_of_continuousOn
    hcontsum.continuousOn
  have hDmeas : Measurable fun t =>
      derivWithin (fun u => C.inf' hC (fun j => hc j (pathX x y u))) (Ioi t) t :=
    measurable_derivWithin_Ioi _
  have hDmem : MemLp (fun t => derivWithin (fun u => C.inf' hC (fun j => hc j (pathX x y u))) (Ioi t) t)
      2 (volume.restrict (Set.Ioc 0 (Real.pi / 2))) := by
    refine MemLp.of_bound hDmeas.aestronglyMeasurable M ?_
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with t ht
    rw [Real.norm_eq_abs]
    refine (hbound t).trans ((le_abs_self _).trans ?_)
    have := hM t (Set.Ioc_subset_Icc_self ht)
    rwa [Real.norm_eq_abs] at this
  have hSCmeas : Measurable fun t => SC C hC (pathX x y t) (pathY x y t) :=
    measurable_SC_path C hC x y
  have hSCbound : ∀ t, SC C hC (pathX x y t) (pathY x y t)
      ≤ ∑ c ∈ C, fderiv ℝ (hc c) (pathX x y t) (pathY x y t) ^ 2 := by
    intro t
    unfold SC
    exact Finset.sum_le_sum fun c _ => Set.indicator_le_self' (fun _ _ => sq_nonneg _) _
  have hSCint : IntervalIntegrable (fun t => SC C hC (pathX x y t) (pathY x y t)) volume 0
      (Real.pi / 2) := by
    have hg : IntervalIntegrable (fun t => ∑ c ∈ C, fderiv ℝ (hc c) (pathX x y t) (pathY x y t) ^ 2)
        volume 0 (Real.pi / 2) :=
      (continuous_finsetSum C fun c hc' => (hφ' c hc').pow 2).intervalIntegrable _ _
    refine hg.mono_fun hSCmeas.aestronglyMeasurable (Filter.Eventually.of_forall fun t => ?_)
    show ‖SC C hC (pathX x y t) (pathY x y t)‖
      ≤ ‖∑ c ∈ C, fderiv ℝ (hc c) (pathX x y t) (pathY x y t) ^ 2‖
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (SC_nonneg _ _ _ _),
      abs_of_nonneg (Finset.sum_nonneg fun c _ => sq_nonneg _)]
    exact hSCbound t
  have hD2int : IntervalIntegrable (fun t =>
      derivWithin (fun u => C.inf' hC (fun j => hc j (pathX x y u))) (Ioi t) t ^ 2) volume 0
      (Real.pi / 2) := by
    refine hSCint.mono_fun (hDmeas.pow_const 2).aestronglyMeasurable
      (Filter.Eventually.of_forall fun t => ?_)
    show ‖derivWithin (fun u => C.inf' hC (fun j => hc j (pathX x y u))) (Ioi t) t ^ 2‖
      ≤ ‖SC C hC (pathX x y t) (pathY x y t)‖
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _),
      abs_of_nonneg (SC_nonneg _ _ _ _)]
    exact hD t
  refine ⟨hSCint, ?_⟩
  calc (fC C hC y - fC C hC x) ^ 2
      = (∫ t in (0 : ℝ)..(Real.pi / 2),
          derivWithin (fun u => C.inf' hC (fun j => hc j (pathX x y u))) (Ioi t) t) ^ 2 := by
        rw [← hg0, ← hg1, hpath]
    _ ≤ (Real.pi / 2 - 0) * ∫ t in (0 : ℝ)..(Real.pi / 2),
          derivWithin (fun u => C.inf' hC (fun j => hc j (pathX x y u))) (Ioi t) t ^ 2 :=
        sq_intervalIntegral_le hpi hDmem
    _ ≤ (Real.pi / 2) * ∫ θ in (0 : ℝ)..(Real.pi / 2), SC C hC (pathX x y θ) (pathY x y θ) := by
        rw [sub_zero]
        exact mul_le_mul_of_nonneg_left
          (intervalIntegral.integral_mono_on hpi.le hD2int hSCint fun t _ => hD t) hpi.le

end PathIneq
