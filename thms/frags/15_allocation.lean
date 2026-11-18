/-! ## §15  Per-layer allocation: why the frontier is a separable problem

The paper's calibrated allocation (`fig:frontier`) grants extra residual passes to
sensitive layers and reports a Pareto improvement over uniform allocation.  That
experiment presupposes something never stated: that the aggregate weight-space
error *decomposes* over layers, so choosing a pass count per layer is a knapsack
over independent terms rather than a joint optimisation.

This section states and proves the decomposition.  It is not deep -- linearity of
the integral over a finite family -- but it is the step that makes the allocation
search well-posed, and without it the frontier plot is a heuristic rather than an
optimisation over a known objective.

What this does **not** license, and the paper is careful about elsewhere: the
*choice* of `P` per layer.  `allocation_separable` says the objective is a sum of
per-layer terms `ρ^{2P_i}‖w_i‖²`; it says nothing about how one learns which
layers are sensitive.  The paper's answer there is a single calibration forward
pass, which is the one place the zero-data constraint is spent, and it is flagged
as such wherever it appears.  Note also that `‖w_i‖²` appears with weight: two
layers with equal sensitivity but different norms do not deserve equal passes,
which is visible here and is easy to miss when reading the frontier plot.

The hypothesis is one `IsotropicPasses` bundle per layer *at a shared `ρ`*.  That
is the honest form: `ρ` is a property of the stage quantiser and the direction
law, not of the layer, so layers sharing a codebook and a group size share `ρ`.
Layers with different fan-in `d` would have different `ρ` and want the
`ρ`-indexed variant below. -/

section Allocation

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- **The aggregate error decomposes over layers.**  For a finite family of
weight rows, each quantised by its own scheme for its own number of passes, the
expected total squared residual is the sum of the per-layer envelopes.

This is what makes `fig:frontier`'s allocation a separable problem: the objective
is `∑ᵢ ρ^{2Pᵢ}‖wᵢ‖²` and the constraint is `∑ᵢ Pᵢ·(b + 16/g)`, so the search is a
knapsack, not a joint optimisation over interacting layers. -/
theorem integral_sum_resid_sq [IsProbabilityMeasure μ] {ι : Type*} [Fintype ι]
    {S : ι → Ω → Scheme E} {W : ι → Ω → E} {ρ : ℝ} (P : ι → ℕ)
    (h : ∀ i, IsotropicPasses (S i) (W i) ρ μ) :
    ∫ ω, ∑ i, ‖resid (S i ω) (W i ω) (P i)‖ ^ 2 ∂μ
      = ∑ i, ρ ^ (2 * P i) * ∫ ω, ‖W i ω‖ ^ 2 ∂μ := by
  rw [integral_finset_sum _ (fun i _ => (h i).integrable (P i))]
  exact Finset.sum_congr rfl fun i _ => integral_norm_resid_sq (h i) (P i)

/-- The same with a per-layer decay constant, which is what differing fan-in
gives: `ρ` is a functional of the stage quantiser and the direction law, so
layers of different width do not share one. -/
theorem integral_sum_resid_sq_of_rho [IsProbabilityMeasure μ] {ι : Type*} [Fintype ι]
    {S : ι → Ω → Scheme E} {W : ι → Ω → E} {ρ : ι → ℝ} (P : ι → ℕ)
    (h : ∀ i, IsotropicPasses (S i) (W i) (ρ i) μ) :
    ∫ ω, ∑ i, ‖resid (S i ω) (W i ω) (P i)‖ ^ 2 ∂μ
      = ∑ i, ρ i ^ (2 * P i) * ∫ ω, ‖W i ω‖ ^ 2 ∂μ := by
  rw [integral_finset_sum _ (fun i _ => (h i).integrable (P i))]
  exact Finset.sum_congr rfl fun i _ => integral_norm_resid_sq (h i) (P i)

/-- **Adding a pass to any layer cannot increase the objective**, when `|ρ| ≤ 1`.

The monotonicity the allocation search relies on: the per-layer term is
non-increasing in that layer's pass count, so the knapsack has no incentive to
withhold budget from a layer, and a greedy pass-granting sweep is not fighting a
non-monotone objective.  `|ρ| ≤ 1` is the regime the paper measures
(`ρ = 0.342` at two bits); nothing here assumes it. -/
theorem term_le_of_succ {ρ : ℝ} (hρ : |ρ| ≤ 1) (c : ℝ) (hc : 0 ≤ c) (p : ℕ) :
    ρ ^ (2 * (p + 1)) * c ≤ ρ ^ (2 * p) * c := by
  have hsq : ρ ^ 2 ≤ 1 := by
    have := abs_le.mp hρ
    nlinarith [this.1, this.2]
  have hnn : (0 : ℝ) ≤ ρ ^ (2 * p) := by
    rw [pow_mul]; exact pow_nonneg (sq_nonneg ρ) p
  have hexp : ρ ^ (2 * (p + 1)) = ρ ^ (2 * p) * ρ ^ 2 := by
    rw [Nat.mul_succ, pow_add]
  rw [hexp]
  exact mul_le_mul_of_nonneg_right
    (by nlinarith [hnn, hsq]) hc

end Allocation

