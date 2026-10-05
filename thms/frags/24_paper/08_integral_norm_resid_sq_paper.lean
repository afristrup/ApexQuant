
/-- **Proposition 2, exactly as the paper states it.**  Haar rotations on `O(d)`, a scalar
codebook `cb` applied coordinate-wise, and `ρ²` the quadrature `d·E_{f_d}[(t − cb t)²]` over the
spherical marginal: the expected squared residual after `P` passes is `ρ^{2P}` times the
expected squared weight.  Every hypothesis is about the inputs; nothing about any residual law
is assumed. -/
theorem integral_norm_resid_sq_paper {d : ℕ} (hd : 2 ≤ d) {cb : ℝ → ℝ} {W : Ω → Ed d}
    {R : Ω → ℕ → Od d} {x₀ : Ed d}
    (h : RotationStream (iso d) (blockStage cb) W R (haarOd d) x₀ μ) (hx₀ : ‖x₀‖ = 1)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun t => (t - cb t) ^ 2) (sphericalCoordLaw d)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme (iso d) (blockStage cb) (R ω)) (W ω) P‖ ^ 2 ∂μ
      = rhoSq d cb (sphericalCoordLaw d) ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  have hmarg : ∀ j : Fin d,
      (sphereUniformAmbient (Ed d)).map (fun v => v j) = sphericalCoordLaw d :=
    fun j => sphereUniformAmbient_map_coord hd j
  have hint' : Integrable (fun t : ℝ => ‖t - cb t‖ ^ 2) (sphericalCoordLaw d) := by
    simpa [Real.norm_eq_abs, sq_abs] using hint
  rw [integral_norm_resid_sq_uniform h hx₀ hW (integrable_blockStage_err cb hmarg hint') P,
    stageDistortion_blockStage_real cb hmarg hint]

end PaperStatement

section WitnessOd
open Orthogonal

variable (d : ℕ)
