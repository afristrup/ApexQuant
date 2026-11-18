
/-- **The hypotheses of the re-isotropised scheme.**  A rotation stream `R` drawn from a
measurable group `G` that acts on `E` through isometries `ι`, together with the regularity the
measure-theoretic argument needs.

Every field is a statement about the *inputs* of the algorithm, never about a residual law, and
never about a per-pass expectation.  That is the whole point: `IsotropicPasses.indep` and
`IsotropicPasses.distortion`, which are statements about residuals, become theorems. -/
structure RotationStream (ι : G → (E ≃ₗᵢ[ℝ] E)) (q : E → E) (W : Ω → E) (R : Ω → ℕ → G)
    (κ : Measure G) (x₀ : E) (μ : Measure Ω) : Prop where
  /-- `ι` implements the `G`-action: this is what makes `G` "a group of rotations". -/
  implements : ∀ (g : G) (x : E), ι g x = g • x
  /-- the quantiser is measurable. -/
  meas_q : Measurable q
  /-- the weight row is measurable. -/
  meas_W : Measurable W
  /-- every rotation is measurable. -/
  meas_R : ∀ p, Measurable (fun ω => R ω p)
  /-- **every pass draws from the same law `κ`**, one Haar draw per pass. -/
  law_R : ∀ p, μ.map (fun ω => R ω p) = κ
  /-- **the pass-`p` rotation is fresh**: independent of the weight row and of all earlier
  rotations, which by `resid_rotScheme_eq_residPast` is everything `r_p` can depend on.
  `indepFun_rot_past_of_iIndepFun` derives this from mutual independence of `(W, R_0, R_1, …)`. -/
  fresh : ∀ p, IndepFun (fun ω => R ω p) (fun ω => (W ω, fun i : Fin p => R ω i)) μ
  /-- no pass is exact, almost surely.  §5's orbit hypothesis is stated `∀ᵐ ω ∂μ`, which is the
  honest measure-theoretic form and lets this field be a.e.\ rather than pointwise. -/
  exact_never : ∀ p, ∀ᵐ ω ∂μ, resid (rotScheme ι q (R ω)) (W ω) p ≠ 0
  /-- `G` is transitive on the unit sphere, with base point `x₀`.  For `G = O(d)` this is
  `orbit_eq_sphere` of §5b; for a finite subgroup it is false, and the conclusions below then
  genuinely do not apply (cf. header item 4). -/
  transitive : ∀ x : E, ‖x‖ = 1 → x ∈ MulAction.orbit G x₀

variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {q : E → E} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G} {x₀ : E}

omit [BorelSpace E] [SecondCountableTopology E] [MeasurableSMul₂ G E] [MeasurableInv G] in
