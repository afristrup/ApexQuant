
/-- The two hypotheses Proposition 2 actually consumes, for a random weight row `W`. -/
structure IsotropicPasses (S : Ω → Scheme E) (W : Ω → E) (ρ : ℝ) (μ : Measure Ω) : Prop where
  /-- integrability of the squared residual norms (implicit in the paper). -/
  integrable : ∀ p, Integrable (fun ω => ‖resid (S ω) (W ω) p‖ ^ 2) μ
  /-- the pass-`p` squared normalised distortion is integrable. -/
  integrable_err : ∀ p,
    Integrable (fun ω => ‖unitVec (S ω) (W ω) p - (S ω).q (unitVec (S ω) (W ω) p)‖ ^ 2) μ
  /-- **direction ⟂ scale** — supplied by Theorem 1. -/
  indep : ∀ p, IndepFun (fun ω => ‖resid (S ω) (W ω) p‖ ^ 2)
    (fun ω => ‖unitVec (S ω) (W ω) p - (S ω).q (unitVec (S ω) (W ω) p)‖ ^ 2) μ
  /-- **every pass has the same normalised distortion `ρ²`** — supplied by Theorem 1 +
  Lemma 1: the coordinate marginals are `f_d` at every pass, so this expectation does not
  depend on `p`. -/
  distortion : ∀ p,
    ∫ ω, ‖unitVec (S ω) (W ω) p - (S ω).q (unitVec (S ω) (W ω) p)‖ ^ 2 ∂μ = ρ ^ 2

variable {S : Ω → Scheme E} {W : Ω → E} {ρ : ℝ}
