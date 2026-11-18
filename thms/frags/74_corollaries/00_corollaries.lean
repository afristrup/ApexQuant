/-! ## §74  Corollaries of the decay law

Five additions, all consequences of the machinery of §1, §5, §11, §17, §22 and §23.

* Tail bound (paper Corollary 1): `measureReal_lt_norm_le_of_integral_sq_le` (generic Markov
  step), `tail_resid_core`, `tail_resid_paper`.
* Jensen: `integral_norm_le_sqrt_integral_sq` (generic), `integral_norm_resid_le_core`,
  `integral_norm_resid_le_paper`, `integral_norm_resid_le_paper_det`.
* Mixed stages (paper Proposition 2, last sentence): `MixedScheme`, `mresid`,
  `integral_norm_mresid_sq`, `integral_norm_mresid_sq_paper`.  The stage may change at every
  pass and the decay constant becomes the product of the per-pass stage distortions.  The
  derivation from the rotation stream is redone in full, it is not assumed per pass.
* Causal encoders (paper Lemma 1, generic form): `causalRun`, `causalRun_take`,
  `causalRun_take_eq`.
* Zero residuals: `RotationStream.exact_never` is not needed.  `RotationStreamCore` is
  `RotationStream` without that field (and without the stage), and every theorem of this
  section is proved from it.  The device is `guardDir`: on the event `r_p = 0` the direction
  is replaced by the base point, which keeps it in the orbit and changes nothing in the
  identity `‖r_{p+1}‖² = ‖r_p‖² ‖u_p - q(u_p)‖²`, both sides being zero there. -/

section CorGeneric

variable {E : Type*} [NormedAddCommGroup E]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- Auxiliary: `√(x^n) = (√x)^n` for `0 ≤ x`. -/
theorem sqrt_pow_of_nonneg {x : ℝ} (hx : 0 ≤ x) (n : ℕ) :
    Real.sqrt (x ^ n) = Real.sqrt x ^ n := by
  have h : x ^ n = (Real.sqrt x ^ n) ^ 2 := by
    rw [← pow_mul, mul_comm, pow_mul, Real.sq_sqrt hx]
  rw [h, Real.sqrt_sq (pow_nonneg (Real.sqrt_nonneg _) _)]

/-- **Markov on the squared norm.**  If `E‖X‖² ≤ M` then `P(‖X‖ > √M/√δ) ≤ δ` for every
`δ > 0`.  The case `M = 0` is included: then `X = 0` almost surely. -/
theorem measureReal_lt_norm_le_of_integral_sq_le [IsProbabilityMeasure μ] {X : Ω → E}
    (hint : Integrable (fun ω => ‖X ω‖ ^ 2) μ) {M δ : ℝ}
    (hM : ∫ ω, ‖X ω‖ ^ 2 ∂μ ≤ M) (hδ : 0 < δ) :
    μ.real {ω | Real.sqrt M / Real.sqrt δ < ‖X ω‖} ≤ δ := by
  have hI0 : 0 ≤ ∫ ω, ‖X ω‖ ^ 2 ∂μ := integral_nonneg fun _ => sq_nonneg _
  have hM0 : 0 ≤ M := hI0.trans hM
  rcases hM0.eq_or_lt with hz | hpos
  · subst hz
    have hI : ∫ ω, ‖X ω‖ ^ 2 ∂μ = 0 := le_antisymm hM hI0
    have hae : (fun ω => ‖X ω‖ ^ 2) =ᵐ[μ] 0 :=
      (integral_eq_zero_iff_of_nonneg (fun _ => sq_nonneg _) hint).mp hI
    have hnull : μ {ω | Real.sqrt 0 / Real.sqrt δ < ‖X ω‖} = 0 := by
      refine measure_mono_null ?_ (ae_iff.mp hae)
      intro ω hω
      have h0 : Real.sqrt 0 / Real.sqrt δ = 0 := by simp
      rw [Set.mem_setOf_eq, h0] at hω
      exact pow_ne_zero 2 hω.ne'
    rw [measureReal_def, hnull]
    simpa using hδ.le
  · have hsub : {ω | Real.sqrt M / Real.sqrt δ < ‖X ω‖} ⊆ {ω | M / δ ≤ ‖X ω‖ ^ 2} := by
      intro ω hω
      have ha : 0 ≤ Real.sqrt M / Real.sqrt δ := by positivity
      have h2 : (Real.sqrt M / Real.sqrt δ) ^ 2 ≤ ‖X ω‖ ^ 2 :=
        pow_le_pow_left₀ ha (le_of_lt hω) 2
      rwa [div_pow, Real.sq_sqrt hM0, Real.sq_sqrt hδ.le] at h2
    have hmk := mul_meas_ge_le_integral_of_nonneg (μ := μ)
      (Filter.Eventually.of_forall fun ω => sq_nonneg ‖X ω‖) hint (M / δ)
    have hle : μ.real {ω | Real.sqrt M / Real.sqrt δ < ‖X ω‖}
        ≤ μ.real {ω | M / δ ≤ ‖X ω‖ ^ 2} := measureReal_mono hsub
    have h3 : μ.real {ω | M / δ ≤ ‖X ω‖ ^ 2} ≤ δ := by
      have hMδ : 0 < M / δ := div_pos hpos hδ
      by_contra hc
      have h4 := mul_lt_mul_of_pos_left (not_le.mp hc) hMδ
      rw [div_mul_cancel₀ M hδ.ne'] at h4
      linarith [hmk.trans hM]
    exact hle.trans h3

/-- **Jensen for the norm.**  `E‖X‖ ≤ √(E‖X‖²)` on a probability space. -/
theorem integral_norm_le_sqrt_integral_sq [IsProbabilityMeasure μ] {X : Ω → E}
    (hint : Integrable (fun ω => ‖X ω‖ ^ 2) μ) :
    ∫ ω, ‖X ω‖ ∂μ ≤ Real.sqrt (∫ ω, ‖X ω‖ ^ 2 ∂μ) := by
  by_cases hX : Integrable (fun ω => ‖X ω‖) μ
  swap
  · rw [integral_undef hX]
    exact Real.sqrt_nonneg _
  have hI0 : 0 ≤ ∫ ω, ‖X ω‖ ^ 2 ∂μ := integral_nonneg fun _ => sq_nonneg _
  have hc0 : 0 ≤ Real.sqrt (∫ ω, ‖X ω‖ ^ 2 ∂μ) := Real.sqrt_nonneg _
  have hsq : ∫ ω, ‖X ω‖ ^ 2 ∂μ = Real.sqrt (∫ ω, ‖X ω‖ ^ 2 ∂μ) ^ 2 := (Real.sq_sqrt hI0).symm
  generalize Real.sqrt (∫ ω, ‖X ω‖ ^ 2 ∂μ) = c at hc0 hsq ⊢
  rcases hc0.eq_or_lt with hz | hpos
  · subst hz
    have hI : ∫ ω, ‖X ω‖ ^ 2 ∂μ = 0 := by rw [hsq]; norm_num
    have hae : (fun ω => ‖X ω‖ ^ 2) =ᵐ[μ] 0 :=
      (integral_eq_zero_iff_of_nonneg (fun _ => sq_nonneg _) hint).mp hI
    have hae' : (fun ω => ‖X ω‖) =ᵐ[μ] 0 := hae.mono fun ω h => by
      have h' : ‖X ω‖ ^ 2 = 0 := h
      exact pow_eq_zero_iff (two_ne_zero) |>.mp h'
    rw [integral_congr_ae hae']
    simp
  · have hpt : ∀ ω, ‖X ω‖ ≤ (‖X ω‖ ^ 2 / c + c) / 2 := by
      intro ω
      rw [le_div_iff₀ two_pos, div_add' _ _ _ hpos.ne', le_div_iff₀ hpos]
      nlinarith [sq_nonneg (‖X ω‖ - c)]
    have hgi : Integrable (fun ω => (‖X ω‖ ^ 2 / c + c) / 2) μ :=
      ((hint.div_const c).add (integrable_const c)).div_const 2
    have h1 := integral_mono hX hgi hpt
    have h2 : ∫ ω, (‖X ω‖ ^ 2 / c + c) / 2 ∂μ = c := by
      rw [integral_div, integral_add (hint.div_const c) (integrable_const c), integral_div,
        integral_const, probReal_univ, one_smul, hsq, sq, mul_div_assoc, div_self hpos.ne']
      ring
    exact h1.trans h2.le

end CorGeneric

/-! ### Mixed stages, deterministically -/

section CorMixedDet

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A progressive quantiser whose stage may change from pass to pass. -/
structure MixedScheme (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E] where
  /-- `rot p` is the rotation applied at pass `p`. -/
  rot : ℕ → (E ≃ₗᵢ[ℝ] E)
  /-- `q p` is the stage used at pass `p`. -/
  q : ℕ → E → E

/-- The residual of a mixed scheme: `resid` with `S.q` replaced by `S.q p`. -/
noncomputable def mresid (S : MixedScheme E) (w : E) : ℕ → E
  | 0 => w
  | p + 1 =>
      mresid S w p
        - ‖mresid S w p‖ • (S.rot p).symm (S.q p (‖mresid S w p‖⁻¹ • S.rot p (mresid S w p)))

@[simp] theorem mresid_zero (S : MixedScheme E) (w : E) : mresid S w 0 = w := rfl

theorem mresid_succ (S : MixedScheme E) (w : E) (p : ℕ) :
    mresid S w (p + 1) = mresid S w p
      - ‖mresid S w p‖ • (S.rot p).symm (S.q p (‖mresid S w p‖⁻¹ • S.rot p (mresid S w p))) :=
  rfl

/-- A single-stage scheme, read as a mixed scheme with a constant stage. -/
def Scheme.toMixed (S : Scheme E) : MixedScheme E where
  rot := S.rot
  q := fun _ => S.q

/-- The mixed residual of a constant-stage scheme is the residual of §1. -/
theorem mresid_toMixed (S : Scheme E) (w : E) (p : ℕ) :
    mresid S.toMixed w p = resid S w p := by
  induction p with
  | zero => rfl
  | succ p ih =>
      rw [mresid_succ, ih]
      rfl

variable {G : Type*} [Group G] [MulAction G E]

/-- The mixed scheme whose pass-`p` rotation is `ι (g p)` and whose pass-`p` stage is `q p`. -/
def mixedRotScheme (ι : G → (E ≃ₗᵢ[ℝ] E)) (q : ℕ → E → E) (g : ℕ → G) : MixedScheme E where
  rot p := ι (g p)
  q := q

omit [Group G] [MulAction G E] in
/-- The constant-stage mixed residual is the residual of `rotScheme`. -/
theorem resid_rotScheme_eq_mresid (ι : G → (E ≃ₗᵢ[ℝ] E)) (q : E → E) (g : ℕ → G) (w : E)
    (p : ℕ) :
    resid (rotScheme ι q g) w p = mresid (mixedRotScheme ι (fun _ => q) g) w p :=
  (mresid_toMixed (rotScheme ι q g) w p).symm

/-- `mresidPast q p` computes the mixed residual from `(w, (g_0, …, g_{p-1}))` alone. -/
noncomputable def mresidPast (q : ℕ → E → E) : ∀ p : ℕ, E × (Fin p → G) → E
  | 0, x => x.1
  | (p + 1), x =>
      let r := mresidPast q p (x.1, fun i => x.2 i.castSucc)
      r - ‖r‖ • ((x.2 (Fin.last p))⁻¹ • q p (‖r‖⁻¹ • ((x.2 (Fin.last p)) • r)))

open Classical in
/-- The guarded direction: `r/‖r‖` off zero, the base point `x₀` at zero. -/
noncomputable def guardDir (x₀ r : E) : E := if r = 0 then x₀ else ‖r‖⁻¹ • r

/-- The guarded direction always lies in the orbit of the base point. -/
theorem guardDir_mem_orbit {x₀ : E} (htrans : ∀ x : E, ‖x‖ = 1 → x ∈ MulAction.orbit G x₀)
    (r : E) : guardDir x₀ r ∈ MulAction.orbit G x₀ := by
  unfold guardDir
  split_ifs with h
  · exact MulAction.mem_orbit_self x₀
  · exact htrans _ (by
      rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr h)])

variable {ι : G → (E ≃ₗᵢ[ℝ] E)}

/-- The mixed residual factors through the past. -/
theorem mresid_mixedRotScheme_eq_mresidPast (hι : ∀ (g : G) (x : E), ι g x = g • x)
    (q : ℕ → E → E) (w : E) (g : ℕ → G) (p : ℕ) :
    mresid (mixedRotScheme ι q g) w p = mresidPast q p (w, fun i : Fin p => g i) := by
  induction p with
  | zero => rfl
  | succ p ih =>
      have hcast : (fun i : Fin p => (fun j : Fin (p + 1) => g j) i.castSucc)
          = fun i : Fin p => g i := by
        funext i; simp
      rw [mresid_succ, ih, mresidPast]
      simp only [hcast, Fin.val_last]
      rw [show (mixedRotScheme ι q g).rot p = ι (g p) from rfl,
        show (mixedRotScheme ι q g).q = q from rfl, symm_apply_eq_inv_smul hι, hι]

/-- **One pass, with the zero residual included.**  The squared norm of the next residual is
the squared scale times the squared stage error of the *guarded* direction.  At `r = 0` both
sides vanish, whatever the base point. -/
theorem norm_step_sq (hι : ∀ (g : G) (x : E), ι g x = g • x) (Q : E → E) (x₀ r : E) (g : G) :
    ‖r - ‖r‖ • (g⁻¹ • Q (‖r‖⁻¹ • (g • r)))‖ ^ 2
      = ‖r‖ ^ 2 * ‖g • guardDir x₀ r - Q (g • guardDir x₀ r)‖ ^ 2 := by
  rcases eq_or_ne r 0 with h | h
  · simp [h]
  · have hn : ‖r‖ ≠ 0 := norm_ne_zero_iff.mpr h
    have hu : g • guardDir x₀ r = ‖r‖⁻¹ • (g • r) := by
      rw [guardDir, if_neg h, smul_comm_real hι]
    have hr : ‖r‖ • (ι g).symm (‖r‖⁻¹ • (g • r)) = r := by
      rw [map_smul, smul_smul, mul_inv_cancel₀ hn, one_smul, ← hι,
        LinearIsometryEquiv.symm_apply_apply]
    have hkey : r - ‖r‖ • (ι g).symm (Q (‖r‖⁻¹ • (g • r)))
        = ‖r‖ • (ι g).symm (‖r‖⁻¹ • (g • r) - Q (‖r‖⁻¹ • (g • r))) := by
      rw [map_sub, smul_sub, hr]
    rw [hu, ← symm_apply_eq_inv_smul hι, hkey, norm_smul, norm_norm,
      LinearIsometryEquiv.norm_map, mul_pow]

/-- `norm_step_sq` along the recursion. -/
theorem norm_mresidPast_succ_sq (hι : ∀ (g : G) (x : E), ι g x = g • x) (q : ℕ → E → E)
    (x₀ w : E) (g : ℕ → G) (p : ℕ) :
    ‖mresidPast q (p + 1) (w, fun i : Fin (p + 1) => g i)‖ ^ 2
      = ‖mresidPast q p (w, fun i : Fin p => g i)‖ ^ 2
        * ‖g p • guardDir x₀ (mresidPast q p (w, fun i : Fin p => g i))
            - q p (g p • guardDir x₀ (mresidPast q p (w, fun i : Fin p => g i)))‖ ^ 2 := by
  have h := norm_step_sq hι (q p) x₀ (mresid (mixedRotScheme ι q g) w p) (g p)
  have hs : mresid (mixedRotScheme ι q g) w (p + 1)
      = mresid (mixedRotScheme ι q g) w p
        - ‖mresid (mixedRotScheme ι q g) w p‖ • ((g p)⁻¹ • q p
            (‖mresid (mixedRotScheme ι q g) w p‖⁻¹ • (g p • mresid (mixedRotScheme ι q g) w p))) := by
    rw [mresid_succ, show (mixedRotScheme ι q g).rot p = ι (g p) from rfl,
      show (mixedRotScheme ι q g).q = q from rfl, symm_apply_eq_inv_smul hι, hι]
  rw [← hs] at h
  simpa only [mresid_mixedRotScheme_eq_mresidPast hι] using h

end CorMixedDet

section CorMixedMeasurable

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G]

theorem measurable_mresidPast {q : ℕ → E → E} (hq : ∀ p, Measurable (q p)) (p : ℕ) :
    Measurable (mresidPast (G := G) q p) := by
  induction p with
  | zero => exact measurable_fst
  | succ p ih =>
      have h1 : Measurable (fun x : E × (Fin (p + 1) → G) =>
          (x.1, fun i : Fin p => x.2 i.castSucc)) :=
        measurable_fst.prodMk (measurable_pi_lambda _ fun i =>
          (measurable_pi_apply (i.castSucc)).comp measurable_snd)
      have hr : Measurable (fun x : E × (Fin (p + 1) → G) =>
          mresidPast (G := G) q p (x.1, fun i : Fin p => x.2 i.castSucc)) := ih.comp h1
      have hg : Measurable (fun x : E × (Fin (p + 1) → G) => x.2 (Fin.last p)) :=
        (measurable_pi_apply _).comp measurable_snd
      exact hr.sub ((hr.norm).smul
        ((hg.inv).smul ((hq p).comp (((hr.norm).inv).smul (hg.smul hr)))))

theorem measurable_guardDir (x₀ : E) : Measurable (guardDir x₀) := by
  unfold guardDir
  exact Measurable.ite (measurableSet_singleton (0 : E)) measurable_const
    (measurable_norm.inv.smul measurable_id)

end CorMixedMeasurable

/-! ### Mixed stages from a rotation stream, with no `exact_never` -/

section CorMixedAssembly

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- `RotationStream` without its stage and without `exact_never`: hypotheses about the weight
row and the rotations only. -/
structure RotationStreamCore (ι : G → (E ≃ₗᵢ[ℝ] E)) (W : Ω → E) (R : Ω → ℕ → G)
    (κ : Measure G) (x₀ : E) (μ : Measure Ω) : Prop where
  /-- `ι` implements the `G`-action. -/
  implements : ∀ (g : G) (x : E), ι g x = g • x
  /-- the weight row is measurable. -/
  meas_W : Measurable W
  /-- every rotation is measurable. -/
  meas_R : ∀ p, Measurable (fun ω => R ω p)
  /-- every pass draws from the same law `κ`. -/
  law_R : ∀ p, μ.map (fun ω => R ω p) = κ
  /-- the pass-`p` rotation is independent of the weight row and of all earlier rotations. -/
  fresh : ∀ p, IndepFun (fun ω => R ω p) (fun ω => (W ω, fun i : Fin p => R ω i)) μ
  /-- `G` is transitive on the unit sphere, with base point `x₀`. -/
  transitive : ∀ x : E, ‖x‖ = 1 → x ∈ MulAction.orbit G x₀

variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G} {x₀ : E}

omit [BorelSpace E] [SecondCountableTopology E] [MeasurableSMul₂ G E] [MeasurableInv G] in
/-- A `RotationStream` is in particular a `RotationStreamCore`: forget the stage and
`exact_never`. -/
theorem RotationStream.toCore {q : E → E} (h : RotationStream ι q W R κ x₀ μ) :
    RotationStreamCore ι W R κ x₀ μ :=
  ⟨h.implements, h.meas_W, h.meas_R, h.law_R, h.fresh, h.transitive⟩

omit [BorelSpace E] [SecondCountableTopology E] [MeasurableSMul₂ G E] [MeasurableInv G] in
theorem RotationStreamCore.isProbabilityMeasure_law (h : RotationStreamCore ι W R κ x₀ μ)
    [IsProbabilityMeasure μ] : IsProbabilityMeasure κ := by
  rw [← h.law_R 0]
  exact Measure.isProbabilityMeasure_map (h.meas_R 0).aemeasurable

/-- The guarded unit direction of pass `p`: the fresh rotation applied to the guarded
normalised residual.  It equals `u_p` whenever `r_p ≠ 0`. -/
noncomputable def mixedDir (x₀ : E) (q : ℕ → E → E) (W : Ω → E) (R : Ω → ℕ → G) (p : ℕ)
    (ω : Ω) : E :=
  R ω p • guardDir x₀ (mresidPast q p (W ω, fun i : Fin p => R ω i))

variable {q : ℕ → E → E}

theorem measurable_stream_mresidPast (h : RotationStreamCore ι W R κ x₀ μ)
    (hq : ∀ p, Measurable (q p)) (p : ℕ) :
    Measurable (fun ω => mresidPast q p (W ω, fun i : Fin p => R ω i)) :=
  (measurable_mresidPast hq p).comp
    (h.meas_W.prodMk (measurable_pi_lambda _ fun i => h.meas_R i))

theorem measurable_mixedDir (h : RotationStreamCore ι W R κ x₀ μ)
    (hq : ∀ p, Measurable (q p)) (p : ℕ) : Measurable (mixedDir x₀ q W R p) :=
  (h.meas_R p).smul ((measurable_guardDir x₀).comp (measurable_stream_mresidPast h hq p))

/-- The fresh rotation is independent of the pair (guarded direction, squared scale). -/
theorem indepFun_rot_guardScale (h : RotationStreamCore ι W R κ x₀ μ)
    (hq : ∀ p, Measurable (q p)) (p : ℕ) :
    IndepFun (fun ω => R ω p)
      (fun ω => (guardDir x₀ (mresidPast q p (W ω, fun i : Fin p => R ω i)),
        ‖mresidPast q p (W ω, fun i : Fin p => R ω i)‖ ^ 2)) μ := by
  have hΨ : Measurable (fun x : E × (Fin p → G) =>
      (guardDir x₀ (mresidPast (G := G) q p x), ‖mresidPast (G := G) q p x‖ ^ 2)) :=
    ((measurable_guardDir x₀).comp (measurable_mresidPast hq p)).prodMk
      ((measurable_mresidPast hq p).norm.pow_const 2)
  exact (h.fresh p).comp (measurable_id (α := G)) hΨ

variable [MeasurableMul G] [IsProbabilityMeasure μ] [κ.IsMulRightInvariant]

/-- **Re-isotropisation without `exact_never`.**  The guarded direction has the orbit law at
every pass, for every sequence of stages. -/
theorem law_mixedDir (h : RotationStreamCore ι W R κ x₀ μ) (hq : ∀ p, Measurable (q p))
    (p : ℕ) : μ.map (mixedDir x₀ q W R p) = orbitLaw κ x₀ := by
  haveI := h.isProbabilityMeasure_law
  have hindep : IndepFun (fun ω => R ω p)
      (fun ω => guardDir x₀ (mresidPast q p (W ω, fun i : Fin p => R ω i))) μ :=
    (indepFun_rot_guardScale h hq p).comp (measurable_id (α := G)) measurable_fst
  exact law_smul_eq_orbitLaw (R := fun ω => R ω p)
    (r := fun ω => guardDir x₀ (mresidPast q p (W ω, fun i : Fin p => R ω i)))
    (h.meas_R p) ((measurable_guardDir x₀).comp (measurable_stream_mresidPast h hq p))
    (h.law_R p) hindep
    (Filter.Eventually.of_forall fun ω => guardDir_mem_orbit h.transitive _)

/-- The squared scale is independent of the squared stage error of the guarded direction. -/
theorem indepFun_scale_mixedErr (h : RotationStreamCore ι W R κ x₀ μ)
    (hq : ∀ p, Measurable (q p)) (p : ℕ) :
    IndepFun (fun ω => ‖mresidPast q p (W ω, fun i : Fin p => R ω i)‖ ^ 2)
      (fun ω => ‖mixedDir x₀ q W R p ω - q p (mixedDir x₀ q W R p ω)‖ ^ 2) μ := by
  haveI := h.isProbabilityMeasure_law
  have key := indepFun_smul_of_indepPair (R := fun ω => R ω p)
    (r := fun ω => guardDir x₀ (mresidPast q p (W ω, fun i : Fin p => R ω i)))
    (Z := fun ω => ‖mresidPast q p (W ω, fun i : Fin p => R ω i)‖ ^ 2)
    (h.meas_R p) ((measurable_guardDir x₀).comp (measurable_stream_mresidPast h hq p))
    ((measurable_stream_mresidPast h hq p).norm.pow_const 2) (h.law_R p)
    (indepFun_rot_guardScale h hq p)
    (Filter.Eventually.of_forall fun ω => guardDir_mem_orbit h.transitive _)
  have hmeasφ : Measurable (fun u : E => ‖u - q p u‖ ^ 2) :=
    (measurable_id.sub (hq p)).norm.pow_const 2
  exact (key.comp hmeasφ (measurable_id (α := ℝ))).symm

/-- The expected squared stage error of pass `p` is the stage distortion of `q p`. -/
theorem integral_mixedErr (h : RotationStreamCore ι W R κ x₀ μ) (hq : ∀ p, Measurable (q p))
    (p : ℕ) :
    ∫ ω, ‖mixedDir x₀ q W R p ω - q p (mixedDir x₀ q W R p ω)‖ ^ 2 ∂μ
      = stageDistortion (q p) (orbitLaw κ x₀) := by
  have hasm : AEStronglyMeasurable (fun u : E => ‖u - q p u‖ ^ 2)
      (μ.map (mixedDir x₀ q W R p)) :=
    ((measurable_id.sub (hq p)).norm.pow_const 2).aestronglyMeasurable
  rw [stageDistortion, ← law_mixedDir h hq p,
    integral_map (measurable_mixedDir h hq p).aemeasurable hasm]

theorem integrable_mixedErr (h : RotationStreamCore ι W R κ x₀ μ) (hq : ∀ p, Measurable (q p))
    (p : ℕ) (hint : Integrable (fun u : E => ‖u - q p u‖ ^ 2) (orbitLaw κ x₀)) :
    Integrable (fun ω => ‖mixedDir x₀ q W R p ω - q p (mixedDir x₀ q W R p ω)‖ ^ 2) μ := by
  rw [← law_mixedDir h hq p] at hint
  have hasm : AEStronglyMeasurable (fun u : E => ‖u - q p u‖ ^ 2)
      (μ.map (mixedDir x₀ q W R p)) :=
    ((measurable_id.sub (hq p)).norm.pow_const 2).aestronglyMeasurable
  exact (integrable_map_measure hasm (measurable_mixedDir h hq p).aemeasurable).mp hint

theorem integrable_norm_mresidPast_sq (h : RotationStreamCore ι W R κ x₀ μ)
    (hq : ∀ p, Measurable (q p)) (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : ∀ p, Integrable (fun u : E => ‖u - q p u‖ ^ 2) (orbitLaw κ x₀)) (p : ℕ) :
    Integrable (fun ω => ‖mresidPast q p (W ω, fun i : Fin p => R ω i)‖ ^ 2) μ := by
  induction p with
  | zero => simpa [mresidPast] using hW
  | succ p ih =>
      have hmul := (indepFun_scale_mixedErr h hq p).integrable_mul ih
        (integrable_mixedErr h hq p (hint p))
      have heq : (fun ω => ‖mresidPast q (p + 1) (W ω, fun i : Fin (p + 1) => R ω i)‖ ^ 2)
          = (fun ω => ‖mresidPast q p (W ω, fun i : Fin p => R ω i)‖ ^ 2) *
            (fun ω => ‖mixedDir x₀ q W R p ω - q p (mixedDir x₀ q W R p ω)‖ ^ 2) := by
        funext ω
        exact norm_mresidPast_succ_sq h.implements q x₀ (W ω) (R ω) p
      rw [heq]
      exact hmul

theorem integral_norm_mresidPast_sq (h : RotationStreamCore ι W R κ x₀ μ)
    (hq : ∀ p, Measurable (q p)) (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : ∀ p, Integrable (fun u : E => ‖u - q p u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) :
    ∫ ω, ‖mresidPast q P (W ω, fun i : Fin P => R ω i)‖ ^ 2 ∂μ
      = (∏ p ∈ range P, stageDistortion (q p) (orbitLaw κ x₀)) * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  induction P with
  | zero => simp [mresidPast]
  | succ P ih =>
      have hpt : ∀ ω, ‖mresidPast q (P + 1) (W ω, fun i : Fin (P + 1) => R ω i)‖ ^ 2
          = ‖mresidPast q P (W ω, fun i : Fin P => R ω i)‖ ^ 2
            * ‖mixedDir x₀ q W R P ω - q P (mixedDir x₀ q W R P ω)‖ ^ 2 :=
        fun ω => norm_mresidPast_succ_sq h.implements q x₀ (W ω) (R ω) P
      rw [integral_congr_ae (Filter.Eventually.of_forall hpt),
        (indepFun_scale_mixedErr h hq P).integral_fun_mul_eq_mul_integral
          (integrable_norm_mresidPast_sq h hq hW hint P).aestronglyMeasurable
          (integrable_mixedErr h hq P (hint P)).aestronglyMeasurable,
        integral_mixedErr h hq P, ih, Finset.prod_range_succ]
      ring

/-- Integrability of the squared mixed residual, at every pass. -/
theorem integrable_norm_mresid_sq (h : RotationStreamCore ι W R κ x₀ μ)
    (hq : ∀ p, Measurable (q p)) (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : ∀ p, Integrable (fun u : E => ‖u - q p u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) :
    Integrable (fun ω => ‖mresid (mixedRotScheme ι q (R ω)) (W ω) P‖ ^ 2) μ := by
  simp only [mresid_mixedRotScheme_eq_mresidPast h.implements]
  exact integrable_norm_mresidPast_sq h hq hW hint P

/-- **Mixed-stage decay (paper Proposition 2, last sentence).**  With a different stage at every
pass, the expected squared residual after `P` passes is the product of the per-pass stage
distortions times the expected squared weight.  No assumption that the residual is nonzero. -/
theorem integral_norm_mresid_sq (h : RotationStreamCore ι W R κ x₀ μ)
    (hq : ∀ p, Measurable (q p)) (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : ∀ p, Integrable (fun u : E => ‖u - q p u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) :
    ∫ ω, ‖mresid (mixedRotScheme ι q (R ω)) (W ω) P‖ ^ 2 ∂μ
      = (∏ p ∈ range P, stageDistortion (q p) (orbitLaw κ x₀)) * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  simp only [mresid_mixedRotScheme_eq_mresidPast h.implements]
  exact integral_norm_mresidPast_sq h hq hW hint P

/-! ### The single-stage law without `exact_never`, and its corollaries -/

variable {q₁ : E → E}

theorem integrable_norm_resid_sq_core (h : RotationStreamCore ι W R κ x₀ μ)
    (hq : Measurable q₁) (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q₁ u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) :
    Integrable (fun ω => ‖resid (rotScheme ι q₁ (R ω)) (W ω) P‖ ^ 2) μ := by
  simp only [resid_rotScheme_eq_mresid]
  exact integrable_norm_mresid_sq h (fun _ => hq) hW (fun _ => hint) P

/-- **`integral_norm_resid_sq_rotationStream` without `exact_never`.** -/
theorem integral_norm_resid_sq_core (h : RotationStreamCore ι W R κ x₀ μ)
    (hq : Measurable q₁) (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q₁ u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι q₁ (R ω)) (W ω) P‖ ^ 2 ∂μ
      = stageDistortion q₁ (orbitLaw κ x₀) ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  simp only [resid_rotScheme_eq_mresid]
  rw [integral_norm_mresid_sq h (fun _ => hq) hW (fun _ => hint) P, Finset.prod_const,
    Finset.card_range]

/-- **Tail bound (paper Corollary 1)**, abstract group, deterministic weight row. -/
theorem tail_resid_core {w : E} {R : Ω → ℕ → G}
    (h : RotationStreamCore ι (fun _ : Ω => w) R κ x₀ μ) (hq : Measurable q₁)
    (hint : Integrable (fun u : E => ‖u - q₁ u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) {δ : ℝ}
    (hδ : 0 < δ) :
    μ.real {ω | ‖w‖ * Real.sqrt (stageDistortion q₁ (orbitLaw κ x₀)) ^ P / Real.sqrt δ
        < ‖resid (rotScheme ι q₁ (R ω)) w P‖} ≤ δ := by
  have hc : ∫ _ω : Ω, ‖w‖ ^ 2 ∂μ = ‖w‖ ^ 2 := by rw [integral_const, probReal_univ, one_smul]
  have hI := integral_norm_resid_sq_core h hq (integrable_const _) hint P
  rw [hc] at hI
  have hT := measureReal_lt_norm_le_of_integral_sq_le
    (integrable_norm_resid_sq_core h hq (integrable_const _) hint P) hI.le hδ
  have hs : Real.sqrt (stageDistortion q₁ (orbitLaw κ x₀) ^ P * ‖w‖ ^ 2)
      = ‖w‖ * Real.sqrt (stageDistortion q₁ (orbitLaw κ x₀)) ^ P := by
    rw [Real.sqrt_mul (pow_nonneg (stageDistortion_nonneg _ _) P), Real.sqrt_sq (norm_nonneg w),
      sqrt_pow_of_nonneg (stageDistortion_nonneg _ _), mul_comm]
  rw [hs] at hT
  exact hT

/-- **Jensen**, abstract group, random weight row:
`E‖r_P‖ ≤ ρ^P · √(E‖W‖²)` with `ρ = √(stage distortion)`. -/
theorem integral_norm_resid_le_core (h : RotationStreamCore ι W R κ x₀ μ)
    (hq : Measurable q₁) (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q₁ u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι q₁ (R ω)) (W ω) P‖ ∂μ
      ≤ Real.sqrt (stageDistortion q₁ (orbitLaw κ x₀)) ^ P
        * Real.sqrt (∫ ω, ‖W ω‖ ^ 2 ∂μ) := by
  have hJ := integral_norm_le_sqrt_integral_sq (integrable_norm_resid_sq_core h hq hW hint P)
  rw [integral_norm_resid_sq_core h hq hW hint P,
    Real.sqrt_mul (pow_nonneg (stageDistortion_nonneg _ _) P),
    sqrt_pow_of_nonneg (stageDistortion_nonneg _ _)] at hJ
  exact hJ

end CorMixedAssembly

/-! ### The paper's setting: Haar rotations on `O(d)`, scalar codebooks -/

section CorPaper
open Orthogonal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- An independent Haar rotation stream on `O(d)` and a weight row: the hypotheses of the
paper's Proposition 2 that concern the randomness, with no stage and no `exact_never`. -/
structure HaarStream (d : ℕ) (W : Ω → Ed d) (R : Ω → ℕ → Od d) (μ : Measure Ω) : Prop where
  /-- the weight row is measurable. -/
  meas_W : Measurable W
  /-- every rotation is measurable. -/
  meas_R : ∀ p, Measurable (fun ω => R ω p)
  /-- every rotation is Haar distributed. -/
  law_R : ∀ p, μ.map (fun ω => R ω p) = haarOd d
  /-- the pass-`p` rotation is independent of the weight row and of all earlier rotations. -/
  fresh : ∀ p, IndepFun (fun ω => R ω p) (fun ω => (W ω, fun i : Fin p => R ω i)) μ

variable {d : ℕ} {W : Ω → Ed d} {R : Ω → ℕ → Od d}

/-- The hypotheses of `integral_norm_resid_sq_paper` contain a `HaarStream`. -/
theorem RotationStream.toHaarStream {q : Ed d → Ed d} {x₀ : Ed d}
    (h : RotationStream (iso d) q W R (haarOd d) x₀ μ) : HaarStream d W R μ :=
  ⟨h.meas_W, h.meas_R, h.law_R, h.fresh⟩

theorem HaarStream.toCore (h : HaarStream d W R μ) {x₀ : Ed d} (hx₀ : ‖x₀‖ = 1) :
    RotationStreamCore (iso d) W R (haarOd d) x₀ μ :=
  ⟨fun g x => iso_apply d g x, h.meas_W, h.meas_R, h.law_R, h.fresh,
    fun x hx => transitive d hx₀ x hx⟩

/-- Auxiliary: a base point for which the orbit law is the uniform law, so that every scalar
codebook has its stage distortion given by the quadrature `rhoSq`. -/
theorem HaarStream.exists_core (hd : 2 ≤ d) (h : HaarStream d W R μ) :
    ∃ x₀ : Ed d, RotationStreamCore (iso d) W R (haarOd d) x₀ μ ∧
      ∀ cb : ℝ → ℝ, Integrable (fun t => (t - cb t) ^ 2) (sphericalCoordLaw d) →
        Integrable (fun u : Ed d => ‖u - blockStage cb u‖ ^ 2) (orbitLaw (haarOd d) x₀) ∧
          stageDistortion (blockStage cb) (orbitLaw (haarOd d) x₀)
            = rhoSq d cb (sphericalCoordLaw d) := by
  have hx₀ : ‖(EuclideanSpace.single (⟨0, by omega⟩ : Fin d) (1 : ℝ) : Ed d)‖ = 1 := by
    rw [PiLp.norm_single 2 _, norm_one]
  refine ⟨EuclideanSpace.single (⟨0, by omega⟩ : Fin d) (1 : ℝ), h.toCore hx₀, ?_⟩
  intro cb hint
  haveI : Nontrivial (Ed d) :=
    ⟨⟨EuclideanSpace.single (⟨0, by omega⟩ : Fin d) (1 : ℝ), 0, by
      intro h0; rw [h0, norm_zero] at hx₀; exact zero_ne_one hx₀⟩⟩
  have hσ : orbitLaw (haarOd d) (EuclideanSpace.single (⟨0, by omega⟩ : Fin d) (1 : ℝ) : Ed d)
      = sphereUniformAmbient (Ed d) :=
    orbitLaw_eq_sphereUniformAmbient (fun g x => iso_apply d g x) (haarOd d)
      (fun x hx => transitive d hx₀ x hx)
  have hmarg : ∀ j : Fin d,
      (sphereUniformAmbient (Ed d)).map (fun v => v j) = sphericalCoordLaw d :=
    fun j => sphereUniformAmbient_map_coord hd j
  have hint' : Integrable (fun t : ℝ => ‖t - cb t‖ ^ 2) (sphericalCoordLaw d) := by
    simpa [Real.norm_eq_abs, sq_abs] using hint
  rw [hσ]
  exact ⟨integrable_blockStage_err cb hmarg hint', stageDistortion_blockStage_real cb hmarg hint⟩

variable [IsProbabilityMeasure μ]

/-- **Mixed-stage decay in the paper's setting.**  Haar rotations on `O(d)`, a scalar codebook
`cb p` at pass `p` applied coordinate-wise: the expected squared residual after `P` passes is
`∏_{p<P} ρ_p²` times the expected squared weight, `ρ_p²` the quadrature of `cb p`. -/
theorem integral_norm_mresid_sq_paper (hd : 2 ≤ d) {cb : ℕ → ℝ → ℝ} (h : HaarStream d W R μ)
    (hq : ∀ p, Measurable (blockStage (m := d) (cb p)))
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : ∀ p, Integrable (fun t => (t - cb p t) ^ 2) (sphericalCoordLaw d)) (P : ℕ) :
    ∫ ω, ‖mresid (mixedRotScheme (iso d) (fun p => blockStage (cb p)) (R ω)) (W ω) P‖ ^ 2 ∂μ
      = (∏ p ∈ range P, rhoSq d (cb p) (sphericalCoordLaw d)) * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  obtain ⟨x₀, hcore, hall⟩ := h.exists_core hd
  rw [integral_norm_mresid_sq hcore hq hW (fun p => (hall (cb p) (hint p)).1) P]
  congr 1
  exact Finset.prod_congr rfl fun p _ => (hall (cb p) (hint p)).2

/-- **Proposition 2 as the paper states it, without `exact_never`.** -/
theorem integral_norm_resid_sq_paper_haarStream (hd : 2 ≤ d) {cb : ℝ → ℝ}
    (h : HaarStream d W R μ) (hq : Measurable (blockStage (m := d) cb))
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun t => (t - cb t) ^ 2) (sphericalCoordLaw d)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme (iso d) (blockStage cb) (R ω)) (W ω) P‖ ^ 2 ∂μ
      = rhoSq d cb (sphericalCoordLaw d) ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  obtain ⟨x₀, hcore, hall⟩ := h.exists_core hd
  rw [integral_norm_resid_sq_core hcore hq hW (hall cb hint).1 P, (hall cb hint).2]

/-- **Tail bound (paper Corollary 1).**  For a deterministic weight row `w`,
`P(‖r_P‖ > ‖w‖ ρ^P / √δ) ≤ δ` with `ρ = √(rhoSq)`.  Holds for every `δ > 0`. -/
theorem tail_resid_paper (hd : 2 ≤ d) {cb : ℝ → ℝ} {w : Ed d}
    (h : HaarStream d (fun _ : Ω => w) R μ) (hq : Measurable (blockStage (m := d) cb))
    (hint : Integrable (fun t => (t - cb t) ^ 2) (sphericalCoordLaw d)) (P : ℕ) {δ : ℝ}
    (hδ : 0 < δ) :
    μ.real {ω | ‖w‖ * Real.sqrt (rhoSq d cb (sphericalCoordLaw d)) ^ P / Real.sqrt δ
        < ‖resid (rotScheme (iso d) (blockStage cb) (R ω)) w P‖} ≤ δ := by
  obtain ⟨x₀, hcore, hall⟩ := h.exists_core hd
  have hT := tail_resid_core hcore hq (hall cb hint).1 P hδ
  rw [(hall cb hint).2] at hT
  exact hT

/-- **Jensen**, random weight row: `E‖r_P‖ ≤ ρ^P · √(E‖W‖²)`. -/
theorem integral_norm_resid_le_paper (hd : 2 ≤ d) {cb : ℝ → ℝ} (h : HaarStream d W R μ)
    (hq : Measurable (blockStage (m := d) cb)) (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun t => (t - cb t) ^ 2) (sphericalCoordLaw d)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme (iso d) (blockStage cb) (R ω)) (W ω) P‖ ∂μ
      ≤ Real.sqrt (rhoSq d cb (sphericalCoordLaw d)) ^ P
        * Real.sqrt (∫ ω, ‖W ω‖ ^ 2 ∂μ) := by
  obtain ⟨x₀, hcore, hall⟩ := h.exists_core hd
  have hJ := integral_norm_resid_le_core hcore hq hW (hall cb hint).1 P
  rw [(hall cb hint).2] at hJ
  exact hJ

/-- **Jensen**, deterministic weight row: `E‖r_P‖ ≤ ρ^P · ‖w‖`. -/
theorem integral_norm_resid_le_paper_det (hd : 2 ≤ d) {cb : ℝ → ℝ} {w : Ed d}
    (h : HaarStream d (fun _ : Ω => w) R μ) (hq : Measurable (blockStage (m := d) cb))
    (hint : Integrable (fun t => (t - cb t) ^ 2) (sphericalCoordLaw d)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme (iso d) (blockStage cb) (R ω)) w P‖ ∂μ
      ≤ Real.sqrt (rhoSq d cb (sphericalCoordLaw d)) ^ P * ‖w‖ := by
  have hJ := integral_norm_resid_le_paper hd h hq (integrable_const _) hint P
  rwa [integral_const, probReal_univ, one_smul, Real.sqrt_sq (norm_nonneg w)] at hJ

/-- The tail bound under the hypotheses of `integral_norm_resid_sq_paper` itself. -/
theorem tail_resid_paper_rotationStream (hd : 2 ≤ d) {cb : ℝ → ℝ} {w x₀ : Ed d}
    (h : RotationStream (iso d) (blockStage cb) (fun _ : Ω => w) R (haarOd d) x₀ μ)
    (hint : Integrable (fun t => (t - cb t) ^ 2) (sphericalCoordLaw d)) (P : ℕ) {δ : ℝ}
    (hδ : 0 < δ) :
    μ.real {ω | ‖w‖ * Real.sqrt (rhoSq d cb (sphericalCoordLaw d)) ^ P / Real.sqrt δ
        < ‖resid (rotScheme (iso d) (blockStage cb) (R ω)) w P‖} ≤ δ :=
  tail_resid_paper hd h.toHaarStream h.meas_q hint P hδ

/-- Jensen under the hypotheses of `integral_norm_resid_sq_paper` itself. -/
theorem integral_norm_resid_le_paper_rotationStream (hd : 2 ≤ d) {cb : ℝ → ℝ} {x₀ : Ed d}
    (h : RotationStream (iso d) (blockStage cb) W R (haarOd d) x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun t => (t - cb t) ^ 2) (sphericalCoordLaw d)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme (iso d) (blockStage cb) (R ω)) (W ω) P‖ ∂μ
      ≤ Real.sqrt (rhoSq d cb (sphericalCoordLaw d)) ^ P
        * Real.sqrt (∫ ω, ‖W ω‖ ^ 2 ∂μ) :=
  integral_norm_resid_le_paper hd h.toHaarStream h.meas_q hW hint P

end CorPaper

/-! ### Causal encoders: the prefix identity, generically -/

section CausalEncoder

variable {St Bit Pay : Type*}

/-- A causal encoder run: from state `s₀`, step `n` reads the state and the `n`-th random
input, and returns the next state and one payload.  The run returns the final state and the
list of payloads in emission order. -/
def causalRun (step : ℕ → St → Bit → St × Pay) (bits : ℕ → Bit) (s₀ : St) : ℕ → St × List Pay
  | 0 => (s₀, [])
  | n + 1 =>
      ((step n (causalRun step bits s₀ n).1 (bits n)).1,
        (causalRun step bits s₀ n).2 ++ [(step n (causalRun step bits s₀ n).1 (bits n)).2])

theorem causalRun_length (step : ℕ → St → Bit → St × Pay) (bits : ℕ → Bit) (s₀ : St) (P : ℕ) :
    (causalRun step bits s₀ P).2.length = P := by
  induction P with
  | zero => rfl
  | succ P ih => simp [causalRun, ih]

/-- **Prefix identity.**  The first `P'` payloads of a `P`-step run are the payloads of the
`P'`-step run. -/
theorem causalRun_take (step : ℕ → St → Bit → St × Pay) (bits : ℕ → Bit) (s₀ : St) {P' P : ℕ}
    (h : P' ≤ P) :
    (causalRun step bits s₀ P).2.take P' = (causalRun step bits s₀ P').2 := by
  induction P, h using Nat.le_induction with
  | base => exact List.take_of_length_le (causalRun_length step bits s₀ P').le
  | succ P hP ih =>
      rw [← ih]
      show ((causalRun step bits s₀ P).2
        ++ [(step P (causalRun step bits s₀ P).1 (bits P)).2]).take P' = _
      exact List.take_append_of_le_length (by rw [causalRun_length]; exact hP)

/-- Runs that agree on the step functions and inputs below `P'` agree up to step `P'`. -/
theorem causalRun_congr {step₁ step₂ : ℕ → St → Bit → St × Pay} {bits₁ bits₂ : ℕ → Bit}
    (s₀ : St) {P' : ℕ} (hstep : ∀ n < P', step₁ n = step₂ n)
    (hbits : ∀ n < P', bits₁ n = bits₂ n) :
    ∀ n ≤ P', causalRun step₁ bits₁ s₀ n = causalRun step₂ bits₂ s₀ n := by
  intro n
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
      intro hn
      have hlt : n < P' := Nat.lt_of_succ_le hn
      have ihn := ih (Nat.le_of_lt hlt)
      simp only [causalRun, ihn, hstep n hlt, hbits n hlt]

/-- **Prefix identity across runs.**  The first `P'` payloads of a `P`-step run equal the
payloads of an independently executed `P'`-step run whose step functions and inputs agree
with it below `P'`. -/
theorem causalRun_take_eq {step₁ step₂ : ℕ → St → Bit → St × Pay} {bits₁ bits₂ : ℕ → Bit}
    (s₀ : St) {P' P : ℕ} (hP : P' ≤ P) (hstep : ∀ n < P', step₁ n = step₂ n)
    (hbits : ∀ n < P', bits₁ n = bits₂ n) :
    (causalRun step₁ bits₁ s₀ P).2.take P' = (causalRun step₂ bits₂ s₀ P').2 := by
  rw [causalRun_take step₁ bits₁ s₀ hP, causalRun_congr s₀ hstep hbits P' le_rfl]

end CausalEncoder
