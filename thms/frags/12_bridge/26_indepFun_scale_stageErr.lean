
/-- **`IsotropicPasses.indep`, derived.**  The squared scale `‖r_p‖²`, a function of the past,
is independent of the squared normalised quantisation error at pass `p`, because the latter is a
function of `R_p • r̂_p` and `R_p` is fresh.  This is the inductive discharge across all passes
that the header records as missing. -/
theorem indepFun_scale_stageErr (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    IndepFun (fun ω => ‖resid (rotScheme ι q (R ω)) (W ω) p‖ ^ 2)
      (fun ω => ‖unitVec (rotScheme ι q (R ω)) (W ω) p
        - q (unitVec (rotScheme ι q (R ω)) (W ω) p)‖ ^ 2) μ := by
  haveI := aux_isProb h
  have key := indepFun_smul_of_indepPair (R := fun ω => R ω p)
    (r := fun ω => residDirPast q p (W ω, fun i : Fin p => R ω i))
    (Z := fun ω => ‖resid (rotScheme ι q (R ω)) (W ω) p‖ ^ 2)
    (h.meas_R p) (aux_meas_dir h p) (aux_meas_scale h p) (h.law_R p)
    (indepFun_rot_dirScale h p) (aux_orbit h p)
  have hmeasφ : Measurable (fun u : E => ‖u - q u‖ ^ 2) :=
    (measurable_id.sub h.meas_q).norm.pow_const 2
  have hrw : ∀ ω, R ω p • residDirPast q p (W ω, fun i : Fin p => R ω i)
      = unitVec (rotScheme ι q (R ω)) (W ω) p :=
    fun ω => (unitVec_rotScheme_eq_smul h.implements q (W ω) (R ω) p).symm
  simp only [hrw] at key
  exact (key.comp hmeasφ (measurable_id (α := ℝ))).symm
