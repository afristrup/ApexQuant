/-! ## §12  A witness: the hypotheses of §11 are satisfiable

`isotropicPasses_of_rotationStream` derives `IsotropicPasses` from `RotationStream`.  That is
only worth something if some object *satisfies* `RotationStream`: a bundle of contradictory
hypotheses would make every theorem of §11 vacuously true, and the file would certify nothing.
This section rules that out by exhibiting one.

The witness is `E = ℝ` with the two-element rotation group `{+1,-1}`, which is exactly `O(1)`,
and which is genuinely transitive on the unit sphere `{±1}`, the uniform law on it, and the
halving quantiser `q(x) = x/2`.  Every field of `RotationStream` is discharged, including the
two that carry the content: `law_R` (one draw from a right-invariant law per pass) and `fresh`
(the pass-`p` rotation is independent of the weight row and all earlier rotations), the latter
obtained from a genuinely i.i.d. rotation stream via `Measure.exists_iid`.

**What this does and does not settle.**  It settles that §11 is not vacuous, and it pins `ρ` to
a number: `witness_stageDistortion` computes the stage distortion to be `1/4`, so §11's
general theorem returns `err(P) = (1/2)^P` for this scheme, which `exists_witness_decay`
reads off.  It
does **not** close the instantiation gap of §5: `d = 1` is the one dimension in which the
orthogonal group is finite, and Mathlib still has neither a `MeasurableSpace` on `E ≃ₗᵢ[ℝ] E`
nor a Haar measure on `O(d)` for `d ≥ 2`, so the paper's `R ∼ Haar(O(d))` remains uninstantiable
here.  A witness proves consistency, not generality. -/

namespace Witness

/-- The rotation group of the line: two elements acting by `±1`.  This is `O(1)`. -/
inductive Sign : Type
  | pos : Sign
  | neg : Sign
  deriving DecidableEq

instance : Fintype Sign where
  elems := {Sign.pos, Sign.neg}
  complete := by rintro (_ | _) <;> simp

instance : MeasurableSpace Sign := ⊤

instance : MeasurableSingletonClass Sign := ⟨fun _ => trivial⟩

instance : Mul Sign :=
  ⟨fun a b => match a, b with
    | .pos, b => b
    | .neg, .pos => .neg
    | .neg, .neg => .pos⟩

instance : One Sign := ⟨.pos⟩

instance : Inv Sign := ⟨id⟩

instance : Group Sign where
  mul_assoc := by decide
  one_mul := by decide
  mul_one := by decide
  inv_mul_cancel := by decide

instance : MeasurableMul Sign :=
  ⟨fun _ => measurable_of_countable _, fun _ => measurable_of_countable _⟩

instance : MeasurableInv Sign := ⟨measurable_of_countable _⟩

/-- The scalar `±1` that a sign denotes. -/
def val : Sign → ℝ
  | .pos => 1
  | .neg => -1

@[simp] theorem val_pos : val .pos = 1 := rfl

@[simp] theorem val_neg : val .neg = -1 := rfl

instance : SMul Sign ℝ := ⟨fun s x => val s * x⟩

@[simp] theorem smul_def (s : Sign) (x : ℝ) : s • x = val s * x := rfl

theorem val_mul (a b : Sign) : val (a * b) = val a * val b := by
  cases a <;> cases b <;> norm_num [val]

@[simp] theorem abs_val (s : Sign) : |val s| = 1 := by cases s <;> norm_num

instance : MulAction Sign ℝ where
  one_smul x := by simp [show val 1 = 1 from rfl]
  mul_smul a b x := by simp [val_mul, mul_assoc]

instance : MeasurableSMul₂ Sign ℝ :=
  ⟨measurable_from_prod_countable_right fun s => by
    simpa using measurable_const_mul (val s)⟩

/-- The uniform law on the two rotations: the counting measure halved. -/
noncomputable def unif : Measure Sign := (2 : ℝ≥0∞)⁻¹ • Measure.count

instance : IsProbabilityMeasure unif := by
  constructor
  have hcard : (ENat.card Sign) = 2 := by
    rw [ENat.card_eq_coe_natCard, Nat.card_eq_fintype_card]
    rfl
  rw [unif, Measure.smul_apply, Measure.count_univ, hcard, smul_eq_mul,
    show ((2 : ℕ∞) : ℝ≥0∞) = (2 : ℝ≥0∞) by simp]
  exact ENNReal.inv_mul_cancel (by norm_num) (by norm_num)

instance : unif.IsMulRightInvariant := by
  rw [unif]; infer_instance

/-- The rotation `ι` of §11: multiplication by `±1`, as a linear isometry of `ℝ`. -/
def rot : Sign → (ℝ ≃ₗᵢ[ℝ] ℝ)
  | .pos => LinearIsometryEquiv.refl ℝ ℝ
  | .neg => LinearIsometryEquiv.neg ℝ

theorem rot_implements (s : Sign) (x : ℝ) : rot s x = s • x := by
  cases s <;> simp [rot]

/-- The stage quantiser: halve.  Chosen because it is measurable, never exact on a nonzero
residual (so `RotationStream.exact_never` holds pointwise, as §5's orbit hypothesis demands),
and has an explicitly computable stage distortion. -/
noncomputable def half (x : ℝ) : ℝ := x / 2

theorem measurable_half : Measurable half := by unfold half; fun_prop

theorem measurable_err : Measurable (fun u : ℝ => ‖u - half u‖ ^ 2) := by
  unfold half; fun_prop

theorem norm_sub_half (u : ℝ) : ‖u - half u‖ = ‖u‖ / 2 := by
  have : u - half u = u / 2 := by rw [half]; ring
  rw [this, norm_div]
  norm_num

/-! ### The residual never vanishes, and shrinks by exactly one half per pass -/

theorem norm_resid_witness (g : ℕ → Sign) {w : ℝ} (hw : w ≠ 0) (p : ℕ) :
    ‖resid (rotScheme rot half g) w p‖ = ‖w‖ / 2 ^ p := by
  induction p with
  | zero => simp
  | succ p ih =>
      have hpos : (0 : ℝ) < ‖w‖ / 2 ^ p := by
        have := norm_pos_iff.mpr hw
        positivity
      have hne : resid (rotScheme rot half g) w p ≠ 0 := by
        rw [← norm_ne_zero_iff, ih]
        exact ne_of_gt hpos
      rw [norm_resid_succ]
      have hq : (rotScheme rot half g).q = half := rfl
      rw [hq, norm_sub_half, norm_unitVec _ _ _ hne, ih]
      ring

theorem resid_witness_ne_zero (g : ℕ → Sign) {w : ℝ} (hw : w ≠ 0) (p : ℕ) :
    resid (rotScheme rot half g) w p ≠ 0 := by
  rw [← norm_ne_zero_iff, norm_resid_witness g hw p]
  have := norm_pos_iff.mpr hw
  positivity

/-- `Sign` is transitive on the unit sphere of `ℝ`, which is `{±1}`.  This is the one dimension
in which the orthogonal group is finite and the transitivity is provable by cases. -/
theorem transitive_on_sphere (x : ℝ) (hx : ‖x‖ = 1) : x ∈ MulAction.orbit Sign (1 : ℝ) := by
  rw [Real.norm_eq_abs, abs_eq (by norm_num : (0:ℝ) ≤ 1)] at hx
  rcases hx with h | h
  · exact ⟨.pos, by simp [h]⟩
  · exact ⟨.neg, by simp [h]⟩

/-! ### The stage distortion is exactly `1/4`, so `ρ = 1/2` -/

private theorem aux_meas_orbit : Measurable (fun s : Sign => s • (1 : ℝ)) :=
  measurable_of_countable _

private theorem aux_err_const (s : Sign) : ‖s • (1 : ℝ) - half (s • (1 : ℝ))‖ ^ 2 = 1 / 4 := by
  rw [norm_sub_half, smul_def, mul_one, Real.norm_eq_abs, abs_val]
  norm_num

theorem witness_integrable :
    Integrable (fun u : ℝ => ‖u - half u‖ ^ 2) (orbitLaw unif (1 : ℝ)) := by
  rw [orbitLaw,
    integrable_map_measure measurable_err.aestronglyMeasurable aux_meas_orbit.aemeasurable]
  exact (integrable_const (1 / 4 : ℝ)).congr
    (Filter.Eventually.of_forall fun s => (aux_err_const s).symm)

theorem witness_stageDistortion : stageDistortion half (orbitLaw unif (1 : ℝ)) = 1 / 4 := by
  rw [stageDistortion, orbitLaw,
    integral_map aux_meas_orbit.aemeasurable measurable_err.aestronglyMeasurable]
  simp only [aux_err_const]
  simp

theorem witness_rho : Real.sqrt (stageDistortion half (orbitLaw unif (1 : ℝ))) = 1 / 2 := by
  rw [witness_stageDistortion]
  rw [show (1 / 4 : ℝ) = (1 / 2) ^ 2 by norm_num]
  exact Real.sqrt_sq (by norm_num)

/-! ### Assembling the witness -/

/-- **`RotationStream` is satisfiable.**  Everything is concrete except the probability space,
which is produced by `Measure.exists_iid`: an i.i.d. stream of rotations, each with law `unif`.
`fresh` is then the disjoint-index-set consequence of mutual independence, exactly as
`indepFun_rot_past_of_iIndepFun` obtains it in §11b, here over `ℕ` rather than `Option ℕ`,
because the weight row is a constant and carries no randomness to separate. -/
theorem exists_rotationStream :
    ∃ (Ω : Type) (_ : MeasurableSpace Ω) (μ : Measure Ω) (R : Ω → ℕ → Sign),
      IsProbabilityMeasure μ ∧
        RotationStream rot half (fun _ => (1 : ℝ)) R unif (1 : ℝ) μ := by
  obtain ⟨Ω, mΩ, P, X, hmeas, hlaw, hindep, hprob⟩ := ProbabilityTheory.exists_iid ℕ unif
  refine ⟨Ω, mΩ, P, fun ω p => X p ω, hprob, ?_⟩
  refine
    { implements := rot_implements
      meas_q := measurable_half
      meas_W := measurable_const
      meas_R := hmeas
      law_R := fun p => (hlaw p).map_eq
      fresh := ?_
      exact_never := fun p =>
        Filter.Eventually.of_forall fun ω => resid_witness_ne_zero _ one_ne_zero p
      transitive := transitive_on_sphere }
  intro p
  have hdisj : Disjoint ({p} : Finset ℕ) (Finset.range p) := by
    simp [Finset.disjoint_singleton_left]
  have key := hindep.indepFun_finset ({p} : Finset ℕ) (Finset.range p) hdisj hmeas
  have hmemp : p ∈ ({p} : Finset ℕ) := Finset.mem_singleton_self p
  have hmemi : ∀ i : Fin p, (i : ℕ) ∈ Finset.range p := fun i => Finset.mem_range.mpr i.isLt
  have hφ : Measurable (fun v : (({p} : Finset ℕ) → Sign) => v ⟨p, hmemp⟩) :=
    measurable_pi_apply _
  have hψ : Measurable (fun v : (↥(Finset.range p) → Sign) =>
      ((1 : ℝ), fun i : Fin p => v ⟨(i : ℕ), hmemi i⟩)) :=
    measurable_const.prodMk (measurable_pi_lambda _ fun _ => measurable_pi_apply _)
  exact key.comp hφ hψ

/-- **The witness, with its decay constant.**  For this scheme §11's general theorem returns
`err(P) = (1/2)^P`, not assumed, but computed from the codebook by `witness_stageDistortion`
and pushed through `rms_relative_error_rotationStream`.  The existence of this object is what
makes §11 a theorem about something. -/
theorem exists_witness_decay :
    ∃ (Ω : Type) (_ : MeasurableSpace Ω) (μ : Measure Ω) (R : Ω → ℕ → Sign),
      IsProbabilityMeasure μ ∧
        RotationStream rot half (fun _ => (1 : ℝ)) R unif (1 : ℝ) μ ∧
          ∀ P : ℕ,
            Real.sqrt (∫ ω, ‖resid (rotScheme rot half (R ω)) (1 : ℝ) P‖ ^ 2 ∂μ) / ‖(1 : ℝ)‖
              = (1 / 2 : ℝ) ^ P := by
  obtain ⟨Ω, mΩ, μ, R, hprob, hstream⟩ := exists_rotationStream
  refine ⟨Ω, mΩ, μ, R, hprob, hstream, fun P => ?_⟩
  rw [rms_relative_error_rotationStream one_ne_zero hstream witness_integrable P, witness_rho]

end Witness
