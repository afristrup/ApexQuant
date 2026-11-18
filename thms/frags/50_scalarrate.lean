
/-! ## §48c  The deployed rate of a coordinatewise stage is one scalar quadrature

§36 leaves the deployed per-pass rate `ρ_H²(v)` of a direction `v` sandwiched between the
extreme values of the rate over the sphere, and §36's `hadRate_blockStage` writes it as a sum of
`d` sign averages, one per coordinate.  §48b makes every one of those coordinates carry the same
law `radLaw d v`.  Putting the two together replaces the sandwich, *at a fixed direction*, with
an equality: `hadRate_blockStage_eq_distortion` says the rate **is** `d · D(cb, radLaw d v)`, the
scalar quadrature of `(t − cb t)²` against one law on `ℝ`, with no inequality and no reference to
`H` or to any coordinate.

`hadRate_sub_rhoSq` is the same identity rearranged: the gap between the deployed rate and the
paper's `ρ² = d · D(cb, sphericalCoordLaw d)` is `d` times the difference of the same scalar
functional against two scalar laws.  Whatever is still to be proved about the deployed rate is
therefore a comparison of `radLaw d v` with `sphericalCoordLaw d` under the single test function
`t ↦ (t − cb t)²`.  §37 is the one place that compares those two laws, and it does so under
`t ↦ t⁴`; §41 to §44 bound the distance from the coordinate law to a Gaussian, which is a
different comparison and is not composed with §37 anywhere here.

What is *not* proved here.  Neither statement says the deployed rate equals the Haar rate: the
right-hand side of `hadRate_sub_rhoSq` is not shown to vanish, and it does not vanish in general.
Both statements are restricted to a stage of the form `blockStage cb`, that is a single scalar
codebook applied to every coordinate, and to a flat `H`; for a general quantiser `q` the rate
does not split across coordinates at all.  And the sandwich is not discharged where §36 *uses*
it: `integral_norm_resid_sq_le_had` ranges over every unit direction, so feeding it still needs a
bound on `d · D(cb, radLaw d v)` uniform in `v`, which this section does not supply.  Nothing
here is asymptotic and nothing is assumed about `v`. -/

namespace ScalarGap

open Hadamard CoordLaw

variable {d : ℕ}

/-- The scalar error functional of a measurable codebook is measurable, which is all
`integral_radLaw` needs of it.  The exponent is the one in `distortion`. -/
theorem measurable_sub_cb_sq {cb : ℝ → ℝ} (hcb : Measurable cb) :
    Measurable fun t : ℝ => (t - cb t) ^ 2 :=
  (measurable_id.sub hcb).pow_const 2

/-- `ρ²` is `d` times the per-coordinate distortion, by definition of both. -/
theorem rhoSq_eq_mul_distortion (d : ℕ) (cb : ℝ → ℝ) (ν : Measure ℝ) :
    rhoSq d cb ν = (d : ℝ) * distortion cb ν := rfl

/-- **The deployed rate of a coordinatewise stage is a scalar quadrature.**  For a flat `H` and a
measurable scalar codebook `cb`, the sign-averaged pass distortion of the direction `v` is
exactly `d` times the distortion of `cb` against the single law `radLaw d v`.  The right-hand
side mentions neither `H` nor any coordinate. -/
theorem hadRate_blockStage_eq_distortion {H : Od d} (hH : IsFlat H) {cb : ℝ → ℝ}
    (hcb : Measurable cb) (v : Ed d) :
    hadRate H (blockStage (m := d) (F := ℝ) cb) v = (d : ℝ) * distortion cb (radLaw d v) := by
  have hterm : ∀ i : Fin d,
      (signAvg fun s => ((hadRot H s • v) i - cb ((hadRot H s • v) i)) ^ 2)
        = distortion cb (radLaw d v) := by
    intro i
    rw [distortion, ← coordLaw_eq_radLaw hH v i,
      integral_coordLaw H v i (measurable_sub_cb_sq hcb)]
  rw [hadRate_blockStage, Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- **The gap to the paper's rate is one scalar comparison.**  The difference between the
deployed rate of a coordinatewise stage and `ρ² = d · D(cb, sphericalCoordLaw d)` is `d` times
the difference of the same scalar distortion against the two scalar laws.  This is an identity,
not a bound, and it does not assert that the difference is zero. -/
theorem hadRate_sub_rhoSq {H : Od d} (hH : IsFlat H) {cb : ℝ → ℝ}
    (hcb : Measurable cb) (v : Ed d) :
    hadRate H (blockStage (m := d) (F := ℝ) cb) v - rhoSq d cb (sphericalCoordLaw d)
      = (d : ℝ) * (distortion cb (radLaw d v) - distortion cb (sphericalCoordLaw d)) := by
  rw [hadRate_blockStage_eq_distortion hH hcb v, rhoSq_eq_mul_distortion, mul_sub]

end ScalarGap
