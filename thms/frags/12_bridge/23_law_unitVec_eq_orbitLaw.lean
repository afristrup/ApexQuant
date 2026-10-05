
/-- **Re-isotropisation, the statement the paper is really making.**  The unit direction `u_p`
has *the same law* `orbitLaw κ x₀` at every pass `p` — no matter what the residual `r_p` looks
like, and no matter what the quantiser `q` has done to it in the previous `p` passes.

This is the fact `IsotropicPasses.distortion` was assuming.  It is the payoff of §5's
`law_smul_eq_orbitLaw` combined with `resid_rotScheme_eq_residPast`: the residual direction is a
function of the past, the fresh rotation is independent of the past, and the orbit law is
constant along the orbit. -/
theorem law_unitVec_eq_orbitLaw (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    μ.map (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p) = orbitLaw κ x₀ := by
  haveI := aux_isProb h
  have hindep : IndepFun (fun ω => R ω p)
      (fun ω => residDirPast q p (W ω, fun i : Fin p => R ω i)) μ := by
    simpa [Function.comp_def] using
      (indepFun_rot_dirScale h p).comp (measurable_id (α := G)) measurable_fst
  have hkey := law_smul_eq_orbitLaw (R := fun ω => R ω p)
      (r := fun ω => residDirPast q p (W ω, fun i : Fin p => R ω i))
      (h.meas_R p) (aux_meas_dir h p) (h.law_R p) hindep (aux_orbit h p)
  simpa only [unitVec_rotScheme_eq_smul h.implements] using hkey
