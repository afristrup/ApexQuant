/-! ## §10  Why the scheme needs no calibration data: indifference to the Hessian

The objection this section answers is the strongest one available against
\S3: unweighted per-coordinate MSE is a *proxy*, and the literature is explicit that
minimising a layerwise reconstruction error is not the same as minimising end-to-end
divergence.  Data-aware methods weight the error by an input-side second-moment matrix `H`
and beat the unweighted objective by doing so.

The scheme here has a specific, provable relationship to that objective, and it is neither
"we also minimise it" nor "we ignore it".  Re-isotropisation makes the residual *direction*
isotropic, and for an isotropic direction the weighted objective `E[uᵀ H u]` collapses to
`tr(H)/d`, the **same number for every `H`**.  The scheme is therefore *indifferent* to the
input-side Hessian rather than ignorant of it.

That cuts exactly two ways, and the paper should say both:

* it is why no calibration data is needed, there is nothing about `H` left to learn, because
  `H` has been averaged out of the objective by the rotation; and
* it is why a data-aware method can still win, indifference also means the scheme *cannot*
  exploit `H` when it is available, which is precisely the fixed-budget gap the paper
  concedes rather than hides.

Only the input-side term is addressed here.  The output-side (cross-layer) factor is
untouched by this argument, and nothing below says anything about it. -/

section HessianIndifference

variable {d : ℕ}

open Finset

/-- The isotropy that a fresh Haar rotation delivers, stated in the only form the quadratic
objective consumes: coordinates uncorrelated, with common variance `1/d`.  For a uniform
direction on `S^{d-1}` both facts hold, the variance is `second_moment_eq_inv_card` of §3b,
forced by `‖u‖ = 1`, and the vanishing cross-moments are the reflection symmetry of the
sphere. -/
structure IsotropicDirection (σ : Measure (EuclideanSpace ℝ (Fin d))) (d : ℕ) : Prop where
  /-- `E[uᵢuⱼ] = δᵢⱼ / d`. -/
  moment : ∀ i j, ∫ u, u i * u j ∂σ = if i = j then (1 : ℝ) / d else 0
  /-- each product is integrable (automatic on the sphere, where `|uᵢ| ≤ 1`). -/
  integrable : ∀ i j, Integrable (fun u => u i * u j) σ

/-- **The weighted objective sees only the dimension.**  For an isotropic direction law, the
layerwise proxy loss `E[uᵀ H u]` equals `tr(H)/d` for *every* `H`, no dependence on the
eigenvectors of `H` at all, and dependence on its eigenvalues only through their sum. -/
theorem integral_quadraticForm_isotropic
    {σ : Measure (EuclideanSpace ℝ (Fin d))} (hσ : IsotropicDirection σ d)
    (H : Matrix (Fin d) (Fin d) ℝ) :
    ∫ u, ∑ i, ∑ j, H i j * (u i * u j) ∂σ = (∑ i, H i i) / d := by
  have hrow : ∀ i : Fin d, ∫ u, ∑ j, H i j * (u i * u j) ∂σ = H i i / d := by
    intro i
    rw [integral_finsetSum _ fun j _ => (hσ.integrable i j).const_mul (H i j)]
    simp_rw [integral_const_mul, hσ.moment i]
    simp [mul_ite, Finset.sum_ite_eq, div_eq_mul_inv]
  rw [integral_finsetSum _ fun i _ =>
    integrable_finsetSum _ fun j _ => (hσ.integrable i j).const_mul (H i j)]
  simp_rw [hrow]
  rw [← Finset.sum_div]

/-- **Indifference, as the paper states it.**  Two input-side second-moment matrices with the
same trace induce exactly the same proxy loss under an isotropic residual direction.  There is
therefore nothing about `H` for a calibration set to reveal, which is why the construction is
data-free, and equally why it cannot profit from data when data is available. -/
theorem quadraticForm_indifferent
    {σ : Measure (EuclideanSpace ℝ (Fin d))} (hσ : IsotropicDirection σ d)
    (H K : Matrix (Fin d) (Fin d) ℝ) (htr : ∑ i, H i i = ∑ i, K i i) :
    ∫ u, ∑ i, ∑ j, H i j * (u i * u j) ∂σ
      = ∫ u, ∑ i, ∑ j, K i j * (u i * u j) ∂σ := by
  rw [integral_quadraticForm_isotropic hσ, integral_quadraticForm_isotropic hσ, htr]

end HessianIndifference

