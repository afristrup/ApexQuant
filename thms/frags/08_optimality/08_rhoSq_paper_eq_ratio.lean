
/-! ### §6b  The paper's `ρ`

`rho d cb (sphericalCoordLaw d)` is the constant of eq. (2.3) as the paper defines it, the one
plotted in the design chart, computable by quadrature before any weight is touched. -/

/-- Eq. (2.3) at the Lemma 1 coordinate law: `ρ² = E[(t-q t)²] / E[t²]` with no side hypothesis,
because §4b supplies `E[t²] = 1/d`. -/
theorem rhoSq_paper_eq_ratio {d : ℕ} (hd : 2 ≤ d) (cb : ℝ → ℝ) :
    rhoSq d cb (sphericalCoordLaw d)
      = (∫ t, (t - cb t) ^ 2 ∂(sphericalCoordLaw d)) / ∫ t, t ^ 2 ∂(sphericalCoordLaw d) := by
  have hd0 : d ≠ 0 := by omega
  exact rhoSq_eq_ratio hd0 (integral_sq_sphericalCoordLaw hd)

end Optimality

