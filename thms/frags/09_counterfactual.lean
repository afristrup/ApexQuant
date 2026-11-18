/-! ## §7  The counterfactual: freshness of the rotation is necessary

The paper's ablation A1 reports that *"reusing a fixed rotation across passes stalls the decay
precisely as the proposition predicts"*.  Formally, what fails when the rotation is reused is not
a rate but the *existence of a rate*: with a fixed rotation the per-pass ratio is a function of
the starting direction, and no single constant describes it.

Here `d = 2` and `cb` is the two-level codebook `{-1/2, 1/2}`; the rotation stream is constantly
the identity.  Two starting rows give two different one-pass ratios, so no `ρ` can satisfy
`‖r_1‖ = ρ‖w‖` for both. -/

section Counterfactual

/-- The fixed-rotation scheme in `ℝ²` with the two-level codebook `{-1/2, 1/2}`. -/
noncomputable def stallScheme : Scheme (EuclideanSpace ℝ (Fin 2)) where
  rot := fun _ => LinearIsometryEquiv.refl ℝ _
  q := fun v => WithLp.toLp 2 (fun i => if 0 ≤ v i then (1 : ℝ) / 2 else -(1 / 2))

private theorem aux_sq2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)

private theorem aux_sq2_lb : (1 : ℝ) ≤ Real.sqrt 2 := by
  nlinarith [aux_sq2, Real.sqrt_nonneg 2]

/-- One pass of `stallScheme`, unfolded.  The rotation is the identity, so the pull-back `R₀ᵀ`
disappears and `r₁ = w - ‖w‖ · q(w/‖w‖)`. -/
private theorem aux_resid_eq (w : EuclideanSpace ℝ (Fin 2)) :
    resid stallScheme w 1 = w - ‖w‖ • stallScheme.q (‖w‖⁻¹ • w) := rfl

private theorem aux_norm_w1 :
    ‖(WithLp.toLp 2 ![(1 : ℝ), 1] : EuclideanSpace ℝ (Fin 2))‖ = Real.sqrt 2 := by
  rw [EuclideanSpace.norm_eq]
  norm_num [Fin.sum_univ_two]

private theorem aux_norm_w2 :
    ‖(WithLp.toLp 2 ![(1 : ℝ), 0] : EuclideanSpace ℝ (Fin 2))‖ = 1 := by
  rw [EuclideanSpace.norm_eq]
  norm_num [Fin.sum_univ_two]

/-- `w = (1,1)`: `‖w‖ = √2`, `u = (1/√2)(1,1)`, both coordinates are `≥ 0` so `q(u) = (½,½)`,
and `r₁ = (1 - √2/2)(1,1)` has norm `√2 - 1`.  The one-pass ratio is `1 - 1/√2 ≈ 0.293`. -/
private theorem aux_resid_w1 :
    ‖resid stallScheme (WithLp.toLp 2 ![(1 : ℝ), 1]) 1‖ = Real.sqrt 2 - 1 := by
  have h0 : (0 : ℝ) ≤ (Real.sqrt 2)⁻¹ := by positivity
  rw [aux_resid_eq, EuclideanSpace.norm_eq, aux_norm_w1]
  norm_num [Fin.sum_univ_two, stallScheme, h0]
  rw [show (1 - Real.sqrt 2 * (1 / 2)) ^ 2 + (1 - Real.sqrt 2 * (1 / 2)) ^ 2
        = (Real.sqrt 2 - 1) ^ 2 by nlinarith [aux_sq2],
    Real.sqrt_sq (by linarith [aux_sq2_lb])]

/-- `w = (1,0)`: `‖w‖ = 1`, `u = (1,0)`, and since `0 ≤ 0` the codebook still returns
`q(u) = (½,½)`, so `r₁ = (½,-½)` has norm `1/√2`.  The one-pass ratio is `1/√2 ≈ 0.707`. -/
private theorem aux_resid_w2 :
    ‖resid stallScheme (WithLp.toLp 2 ![(1 : ℝ), 0]) 1‖ = Real.sqrt 2 / 2 := by
  rw [aux_resid_eq, EuclideanSpace.norm_eq, aux_norm_w2]
  norm_num [Fin.sum_univ_two, stallScheme]
  have h2 : Real.sqrt 2 ≠ 0 := by positivity
  field_simp
  nlinarith [aux_sq2]

/-- **Freshness is necessary.**  With a fixed rotation there is no constant `ρ` for which the
one-pass residual satisfies `‖r_1‖ = ρ ‖w‖` at every input: the ratio depends on the direction
of `w`, which is exactly what a fresh rotation removes.  Hence the hypothesis of Theorem 1 is
not decorative, the conclusion of Proposition 2 fails without it. -/
theorem no_universal_ratio :
    ¬ ∃ ρ : ℝ, ∀ w : EuclideanSpace ℝ (Fin 2), w ≠ 0 →
      ‖resid stallScheme w 1‖ = ρ * ‖w‖ := by
  rintro ⟨ρ, h⟩
  have hw1 : (WithLp.toLp 2 ![(1 : ℝ), 1] : EuclideanSpace ℝ (Fin 2)) ≠ 0 :=
    fun hc => by simpa using congrFun (congrArg WithLp.ofLp hc) 0
  have hw2 : (WithLp.toLp 2 ![(1 : ℝ), 0] : EuclideanSpace ℝ (Fin 2)) ≠ 0 :=
    fun hc => by simpa using congrFun (congrArg WithLp.ofLp hc) 0
  -- `√2 - 1 = ρ·√2` from `w = (1,1)`, and `√2/2 = ρ·1` from `w = (1,0)`;
  -- together these force `√2 = 2`, i.e. `2 = 4`.
  have h1 := h _ hw1
  have h2 := h _ hw2
  rw [aux_resid_w1, aux_norm_w1] at h1
  rw [aux_resid_w2, aux_norm_w2] at h2
  nlinarith [aux_sq2, Real.sqrt_nonneg 2]

end Counterfactual

