/-!
# §46  The constant of the block law

§20 and §21 prove that a `B`-block of a uniform direction on the sphere of `ℝ^d` has density
proportional to `(1 − ‖x‖²)^{(d−B−2)/2}` on the unit ball, the constant being the inverse of the
total mass, and §21 identifies that constant with the paper's `f_d` for `B = 1`.  This section
computes it for every `B` without a radial integral: §20's unnormalised identity
`(σ_{U×V})_* fst = σ_V(univ) · blockMeasure U (dim V)` applied to the whole space reads
`(B+m) vol(ball_{B+m}) = m vol(ball_m) · mass`, and Mathlib's `vol(ball_n) = √π^n/Γ(n/2+1)`
with `Γ(x+1) = xΓ(x)` gives `mass = √π^B Γ(m/2)/Γ((B+m)/2)`.

* `blockMeasure_univ`: `∫_{ball} (1 − ‖x‖²)^{(m−2)/2} dx = √π^B Γ(m/2)/Γ((B+m)/2)`.
* `blockLaw_eq_withDensity`: the block law is
  `Γ((B+m)/2)/(√π^B Γ(m/2)) · (1 − ‖x‖²)^{(m−2)/2}` on the unit ball, Lemma 1 in block form
  with its constant, the law the `E8` and trellis stages of §17 integrate against.
-/

open Metric

section BlockConstant

variable (U : Type*) [NormedAddCommGroup U] [InnerProductSpace ℝ U] [MeasurableSpace U]
  [BorelSpace U] [FiniteDimensional ℝ U] [Nontrivial U]
