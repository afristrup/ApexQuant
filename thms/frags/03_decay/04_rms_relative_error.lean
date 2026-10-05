
/-- **Proposition 2, eq. (2.5)**, in the form the paper states it: for a *deterministic* weight
row `w` (the expectation is over the rotations alone, as the paper's proof notes), the RMS
relative error after `P` passes is exactly `ρ^P`. -/
theorem rms_relative_error [IsProbabilityMeasure μ] {w : E} (hw : w ≠ 0) (hρ : 0 ≤ ρ)
    (h : IsotropicPasses S (fun _ => w) ρ μ) (P : ℕ) :
    Real.sqrt (∫ ω, ‖resid (S ω) w P‖ ^ 2 ∂μ) / ‖w‖ = ρ ^ P := by
  -- NOTE ON `hρ`: the hypothesis `0 ≤ ρ` is genuinely load-bearing, not cosmetic.
  -- `IsotropicPasses` constrains `ρ` only through `ρ ^ 2` (field `distortion`), so it is
  -- invariant under `ρ ↦ -ρ`; the conclusion is not, for odd `P`.  Without `hρ` the
  -- statement is FALSE — see the report accompanying this fragment for an explicit
  -- counterexample.  The honest unconditional form of eq. (2.5) is `= |ρ| ^ P`.
  have h1 : ∫ ω, ‖resid (S ω) w P‖ ^ 2 ∂μ = ρ ^ (2 * P) * ∫ _ω : Ω, ‖w‖ ^ 2 ∂μ :=
    integral_norm_resid_sq h P
  rw [h1, integral_const, probReal_univ, one_smul, pow_mul',
    Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (pow_nonneg hρ P),
    Real.sqrt_sq (norm_nonneg w), mul_div_assoc, div_self (norm_ne_zero_iff.mpr hw), mul_one]

end Decay

