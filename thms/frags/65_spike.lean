
/-! ## §51  A coordinate direction refutes every uniform bound on the deployed rate

§48c makes the deployed per-pass rate of a *fixed* direction exact: for a flat `H` and a scalar
codebook applied coordinatewise, `hadRate H (blockStage cb) v = d·D(cb, radLaw d v)`
(`ScalarGap.hadRate_blockStage_eq_distortion`), one scalar quadrature and no inequality.  §48e
and §62 identify that law at one direction, `v = eⱼ`, twice over and by two independent routes: it is the two-point law on `±1/√d`, every bit of its mass sitting on the
root-mean-square value.  This section multiplies the two together and records what the product
forbids.

The identity is `hadRate_blockStage_single`.  At `v = eⱼ` the quadrature has two nodes, so the
deployed rate is `d·((1/√d − cb(1/√d))² + (−1/√d − cb(−1/√d))²)/2`: a function of the codebook at
exactly two points of the line, and of nothing else.  No nearest-codeword hypothesis is used
anywhere below, only measurability of `cb`, and nothing is assumed about `d` beyond `j : Fin d`
existing.  The right-hand side names neither `H` nor the output coordinate, so every flat `H` and
every coordinate direction give the same number (`hadRate_blockStage_single_congr`); the flat
family cannot be reshuffled out of this.  For an odd codebook the two nodes coincide and the rate
collapses to `d·(1/√d − cb(1/√d))²` (`hadRate_blockStage_single_of_odd`), equivalently
`(1 − √d·cb(1/√d))²` (`hadRate_blockStage_single_of_odd'`), which depends on `d` through the
single number `√d·cb(1/√d)`: a codebook whose codewords are placed at the coordinate scale
`1/√d` therefore has the *same* coordinate-direction rate in every dimension.  Both forms are
stated, since a deployed codebook need not be odd and the two-sided form is the one that holds
unconditionally.

**The refutation.**  `abs_coordGap_le_of_uniform` is the statement in the form that cannot be
misread: *any* `η` that bounds `|hadRate H (blockStage cb) v − ρ²|` over every unit direction
`v` is at least the value of that quantity at `v = eⱼ`, which the identity above makes explicit.
`not_uniform_of_lt_coordGap` is the same fact contrapositively, as the negation of the uniform
statement.  Both are unconditional and codebook-agnostic: they hold for every measurable `cb`,
and they cap `η` by a quantity a given codebook then makes concrete, rather than hardcoding any
one codebook's number.  So a uniform bound on the deployed rate is not open work here.  It is
false as soon as the two-point value at `±1/√d` differs from `ρ² = d·D(cb, f_d)`, and `f_d`
spreads around `±1/√d` rather than sitting on it.

**Where the numbers live, and what is not established.**  No Lloyd-Max level is formalised as a
real number anywhere in this development, and no codebook is constructed: the arithmetic that
turns the cap above into a number, `0.2605` against `ρ² = 0.1171` for the shipped 2-bit codebook,
a factor `2.22` and the same factor at `d = 256`, `512` and `1024`, is a computation in the
paper, not a theorem here.  Nothing below evaluates `rhoSq d cb (sphericalCoordLaw d)`, or proves
it positive, or compares it with anything.

This says nothing about the near-flat regime and does not weaken §59 or §60.  `eⱼ` is the
spikiest unit direction there is, `∑ⱼ vⱼ⁴ = 1` and `maxⱼ|vⱼ|·√d = √d`, so it fails the flat
hypothesis `|vⱼ| ≤ d^{-1/2}` of `Threshold.abs_hadRate_sub_rhoSq_le_of_spread` outright and
satisfies §60's `|vⱼ| ≤ c·d^{-1/2}` only at `c = √d`, where §60's factor is `Θ(√d)` and asserts
nothing.  The refutation and those theorems are consistent, and neither implies the other: what
is refuted is a bound over *all* unit directions, which is the hypothesis §36's
`integral_norm_resid_sq_le_had` would need if the sandwich were to be closed at `ρ²`, and which
§59 and §60 never claimed.  Nor is the deployed stream's decay in question: `Hadamard.hadRate_le`
already bounds the rate at every direction by `(1 + R)²` for a codebook in `B(0, R)`, so §36's
sandwich still has finite ends.  What is ruled out is that those ends can both be taken close to
the Haar rate.  Finally, nothing here is a statement about the pipeline's quality: it bounds what
can be *proved* of the deployed rotation in this shape, and a rate that exceeds `ρ²` at one
direction is not a rate the deployed stream is observed to suffer. -/

namespace CoordSpike

open Hadamard Orthogonal CoordLaw ScalarGap

variable {d : ℕ}

/-- **The deployed rate at a coordinate direction, two-sided.**  For a flat `H` and a measurable
scalar codebook applied coordinatewise, the deployed per-pass rate of `v = eⱼ` is `d` times the
average of the squared codebook error at the two points `±1/√d`, and nothing else enters: the
coordinate law there is the two-point law on `±1/√d` (§48e), so the quadrature of §48c has two
nodes.  Neither `H` nor the output coordinate appears on the right.  The codebook is not assumed
odd, nearest-codeword, ordered or bounded, only measurable. -/
theorem hadRate_blockStage_single {H : Od d} (hH : IsFlat H) {cb : ℝ → ℝ}
    (hcb : Measurable cb) (j : Fin d) :
    hadRate H (blockStage (m := d) (F := ℝ) cb) (EuclideanSpace.single j (1:ℝ))
      = (d : ℝ) * (((1 / Real.sqrt d - cb (1 / Real.sqrt d)) ^ 2
          + (-(1 / Real.sqrt d) - cb (-(1 / Real.sqrt d))) ^ 2) / 2) := by
  rw [hadRate_blockStage_eq_distortion hH hcb, distortion,
    integral_radLaw_single j (measurable_sub_cb_sq hcb)]

/-- **No flat `H` and no choice of coordinate escapes it.**  The rate at a coordinate direction is
the same number for every flat `H` and every coordinate direction, since the right-hand side of
`hadRate_blockStage_single` mentions neither.  This is why the refutation below cannot be evaded
by picking a different member of the flat family or a different basis vector. -/
theorem hadRate_blockStage_single_congr {H H' : Od d} (hH : IsFlat H) (hH' : IsFlat H')
    {cb : ℝ → ℝ} (hcb : Measurable cb) (j j' : Fin d) :
    hadRate H (blockStage (m := d) (F := ℝ) cb) (EuclideanSpace.single j (1:ℝ))
      = hadRate H' (blockStage (m := d) (F := ℝ) cb) (EuclideanSpace.single j' (1:ℝ)) := by
  rw [hadRate_blockStage_single hH hcb j, hadRate_blockStage_single hH' hcb j']

/-- **The odd case, one node.**  If the codebook is odd, `cb(−t) = −cb t`, the two nodes carry
equal error and the rate at a coordinate direction is `d·(1/√d − cb(1/√d))²`.  Oddness is a
hypothesis, not a fact proved of any codebook here; the two-sided
`hadRate_blockStage_single` is the form that needs none. -/
theorem hadRate_blockStage_single_of_odd {H : Od d} (hH : IsFlat H) {cb : ℝ → ℝ}
    (hcb : Measurable cb) (hodd : ∀ t : ℝ, cb (-t) = -cb t) (j : Fin d) :
    hadRate H (blockStage (m := d) (F := ℝ) cb) (EuclideanSpace.single j (1:ℝ))
      = (d : ℝ) * (1 / Real.sqrt d - cb (1 / Real.sqrt d)) ^ 2 := by
  rw [hadRate_blockStage_single hH hcb j, hodd (1 / Real.sqrt d)]
  ring

/-- `d·(1/√d − c)² = (1 − √d·c)²`: the factor `d` is absorbed by squaring `√d`.  Arithmetic, for
`d ≥ 1` and any real `c`. -/
theorem mul_sub_sq_sqrt_inv (hd : 0 < d) (c : ℝ) :
    (d : ℝ) * (1 / Real.sqrt d - c) ^ 2 = (1 - Real.sqrt d * c) ^ 2 := by
  have hd0 : (0 : ℝ) < d := by exact_mod_cast hd
  have hs : Real.sqrt d ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hd0)
  have h1 : Real.sqrt d * (1 / Real.sqrt d - c) = 1 - Real.sqrt d * c := by
    field_simp
  calc (d : ℝ) * (1 / Real.sqrt d - c) ^ 2
      = (Real.sqrt d * (1 / Real.sqrt d - c)) ^ 2 := by
        rw [mul_pow, Real.sq_sqrt hd0.le]
    _ = (1 - Real.sqrt d * c) ^ 2 := by rw [h1]

/-- **The odd case, in the form that shows the dimension cancels.**  The rate at a coordinate
direction is `(1 − √d·cb(1/√d))²`, which depends on `d` only through the single number
`√d·cb(1/√d)`, the codeword nearest the coordinate scale measured in units of that scale.  A
codebook placed at the scale `1/√d` therefore has the same coordinate-direction rate in every
dimension.  Whether any particular codebook is so placed is not addressed here, and the value of
`√d·cb(1/√d)` is not computed for any codebook. -/
theorem hadRate_blockStage_single_of_odd' {H : Od d} (hH : IsFlat H) {cb : ℝ → ℝ}
    (hcb : Measurable cb) (hodd : ∀ t : ℝ, cb (-t) = -cb t) (j : Fin d) :
    hadRate H (blockStage (m := d) (F := ℝ) cb) (EuclideanSpace.single j (1:ℝ))
      = (1 - Real.sqrt d * cb (1 / Real.sqrt d)) ^ 2 := by
  have hd : 0 < d := lt_of_le_of_lt (Nat.zero_le (j : ℕ)) j.isLt
  rw [hadRate_blockStage_single_of_odd hH hcb hodd j, mul_sub_sq_sqrt_inv hd]

/-- **Every uniform bound is capped from below.**  If `η` bounds `|hadRate H (blockStage cb) v −
ρ²|` at *every* unit direction `v`, then `η` is at least the value of that quantity at the
coordinate direction `eⱼ`, which `hadRate_blockStage_single` evaluates explicitly.  `eⱼ` is a
unit vector (`CoordLaw.norm_single_eq_one`), so the hypothesis does apply to it and the
conclusion does bear on the sandwich of §36.

Unconditional and codebook-agnostic: no codebook is named, no number is asserted, and the
right-hand side is not claimed to be large.  It is a cap that a given codebook makes concrete;
for the shipped 2-bit codebook the arithmetic is in the paper, not here. -/
theorem abs_coordGap_le_of_uniform {H : Od d} (hH : IsFlat H) {cb : ℝ → ℝ}
    (hcb : Measurable cb) (j : Fin d) {η : ℝ}
    (hη : ∀ v : Ed d, ‖v‖ = 1 →
      |hadRate H (blockStage (m := d) (F := ℝ) cb) v - rhoSq d cb (sphericalCoordLaw d)| ≤ η) :
    |(d : ℝ) * (((1 / Real.sqrt d - cb (1 / Real.sqrt d)) ^ 2
          + (-(1 / Real.sqrt d) - cb (-(1 / Real.sqrt d))) ^ 2) / 2)
        - rhoSq d cb (sphericalCoordLaw d)| ≤ η := by
  have h := hη (EuclideanSpace.single j (1:ℝ)) (norm_single_eq_one j)
  rwa [hadRate_blockStage_single hH hcb j] at h

/-- The same cap for an odd codebook, in the dimension-free form: `η ≥ |(1 − √d·cb(1/√d))² −
ρ²|`.  Oddness is a hypothesis on `cb`; `abs_coordGap_le_of_uniform` needs none. -/
theorem abs_coordGap_le_of_uniform_of_odd {H : Od d} (hH : IsFlat H) {cb : ℝ → ℝ}
    (hcb : Measurable cb) (hodd : ∀ t : ℝ, cb (-t) = -cb t) (j : Fin d) {η : ℝ}
    (hη : ∀ v : Ed d, ‖v‖ = 1 →
      |hadRate H (blockStage (m := d) (F := ℝ) cb) v - rhoSq d cb (sphericalCoordLaw d)| ≤ η) :
    |(1 - Real.sqrt d * cb (1 / Real.sqrt d)) ^ 2 - rhoSq d cb (sphericalCoordLaw d)| ≤ η := by
  have h := hη (EuclideanSpace.single j (1:ℝ)) (norm_single_eq_one j)
  rwa [hadRate_blockStage_single_of_odd' hH hcb hodd j] at h

/-- **The refutation, stated as a negation.**  Any `η` below the coordinate-direction gap fails
to bound the deployed rate against `ρ²` uniformly over the unit sphere.  This is
`abs_coordGap_le_of_uniform` contrapositively, and it is the statement the development wants on
record: a uniform bound at any level below the explicit quantity is not unproved, it is false.

What is *not* claimed: that the quantity is large, for this or any codebook, and that the
deployed rate is bad anywhere.  The hypothesis `hlt` supplies the comparison, and this
development proves no instance of it. -/
theorem not_uniform_of_lt_coordGap {H : Od d} (hH : IsFlat H) {cb : ℝ → ℝ}
    (hcb : Measurable cb) (j : Fin d) {η : ℝ}
    (hlt : η < |(d : ℝ) * (((1 / Real.sqrt d - cb (1 / Real.sqrt d)) ^ 2
          + (-(1 / Real.sqrt d) - cb (-(1 / Real.sqrt d))) ^ 2) / 2)
        - rhoSq d cb (sphericalCoordLaw d)|) :
    ¬ ∀ v : Ed d, ‖v‖ = 1 →
      |hadRate H (blockStage (m := d) (F := ℝ) cb) v - rhoSq d cb (sphericalCoordLaw d)| ≤ η :=
  fun hη => absurd (abs_coordGap_le_of_uniform hH hcb j hη) (not_le.mpr hlt)

/-- The negation for an odd codebook, against the dimension-free value `(1 − √d·cb(1/√d))²`. -/
theorem not_uniform_of_lt_coordGap_of_odd {H : Od d} (hH : IsFlat H) {cb : ℝ → ℝ}
    (hcb : Measurable cb) (hodd : ∀ t : ℝ, cb (-t) = -cb t) (j : Fin d) {η : ℝ}
    (hlt : η < |(1 - Real.sqrt d * cb (1 / Real.sqrt d)) ^ 2
        - rhoSq d cb (sphericalCoordLaw d)|) :
    ¬ ∀ v : Ed d, ‖v‖ = 1 →
      |hadRate H (blockStage (m := d) (F := ℝ) cb) v - rhoSq d cb (sphericalCoordLaw d)| ≤ η :=
  fun hη => absurd (abs_coordGap_le_of_uniform_of_odd hH hcb hodd j hη) (not_le.mpr hlt)

end CoordSpike
