
/-! ### Auxiliary Beta-integral machinery

Everything below `aux_` is private scaffolding for the four statements of Lemma 1.  The chain is

* `aux_beta_real` : the real-valued Euler Beta integral `∫₀¹ xᵘ⁻¹(1-x)ᵛ⁻¹ = Γu Γv / Γ(u+v)`,
  obtained from `Complex.betaIntegral_eq_Gamma_mul_div` by pushing the (genuinely real)
  integrand through `Complex.ofReal`;
* `aux_ii` : interval integrability of `t ↦ (1 - t²)^a` on `[-1,1]` for `a > -1` (needed
  because for `d = 2` the exponent is `-1/2` and the density is unbounded at `±1`);
* `aux_I` : `∫_{-1}^{1} (1-t²)^a = √π Γ(a+1) / Γ(a+3/2)`, via `t = 2x-1` and Legendre
  duplication;
* `aux_J` : the same with a `t²` weight, via `t²(1-t²)^a = (1-t²)^a - (1-t²)^(a+1)`. -/

private theorem aux_beta_real (u v : ℝ) (hu : 0 < u) (hv : 0 < v) :
    ∫ x in (0:ℝ)..1, x ^ (u - 1) * (1 - x) ^ (v - 1)
      = Real.Gamma u * Real.Gamma v / Real.Gamma (u + v) := by
  have key : (Complex.betaIntegral u v)
      = ((∫ x in (0:ℝ)..1, x ^ (u - 1) * (1 - x) ^ (v - 1) : ℝ) : ℂ) := by
    rw [Complex.betaIntegral, intervalIntegral.integral_of_le zero_le_one,
      intervalIntegral.integral_of_le zero_le_one, ← integral_complex_ofReal]
    refine setIntegral_congr_fun measurableSet_Ioc fun x hx => ?_
    have hx0 : (0:ℝ) ≤ x := le_of_lt hx.1
    have hx1 : (0:ℝ) ≤ 1 - x := by linarith [hx.2]
    rw [Complex.ofReal_mul, Complex.ofReal_cpow hx0, Complex.ofReal_cpow hx1]
    push_cast
    ring
  rw [Complex.betaIntegral_eq_Gamma_mul_div _ _ (by simpa using hu) (by simpa using hv)] at key
  rw [← Complex.ofReal_inj, ← key,
    show ((u:ℂ) + v) = ((u + v : ℝ) : ℂ) by push_cast; ring,
    Complex.Gamma_ofReal, Complex.Gamma_ofReal, Complex.Gamma_ofReal]
  push_cast
  ring
