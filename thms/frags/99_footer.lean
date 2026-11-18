/-! ## §8  Scope: what is *not* claimed

The paper's Remark ("Scope") is careful that per-pass scalar optimality does **not** imply that
`P` passes at `b` bits beat one pass at `Pb` bits.  There is nothing to prove here, the point is
that no theorem above has that shape, and the table at the head of this file is the complete list
of what is claimed.  Three further remarks, one of which has since been discharged:

* Remark ("Haar in theory, Hadamard in practice"): the deployed rotation is a randomised
  Hadamard transform, a *finite family* inside `O(d)`.  `orbitLaw` for a finite family is
  supported on a finite set, so `law_smul_eq_orbitLaw` gives the uniform measure on that finite
  orbit, **not** the uniform measure on the sphere.  Theorem 1 is therefore false as stated for
  the deployed rotation, and that it attains the Haar rate `ρ²` **exactly** is not proved here
  and is not true in general.  What §36–§44 certify instead are the quantitative replacements:
  the coordinate second moment is `‖v‖²/d` for every input, the fourth-moment gap against the
  sphere is an identity in the spikiness `∑ⱼ vⱼ⁴`, the characteristic function is within
  `t⁴ ∑ⱼ vⱼ⁴ / (3d²)` of the Gaussian one and bounded in damped form on the whole range, and
  Esseen's inequality in Fejér-kernel form turns that into the Kolmogorov–Smirnov rate.  §36 also
  proves the per-pass identity for the deployed rotation in conditional form, with the rate
  sandwiched between the extreme values of `ρ²_H` on the sphere.

  §48b and §48c narrow that sandwich for the stages the paper actually ships.  Every coordinate
  of `H D_ε v` carries the *same* law on `ℝ`, `radLaw d v` (`CoordLaw.coordLaw_eq_radLaw`), so
  for a scalar codebook applied coordinatewise and a flat `H` the rate of a fixed direction is
  not bracketed but exactly `d·D(cb, radLaw d v)`
  (`ScalarGap.hadRate_blockStage_eq_distortion`), and the gap to `ρ²` is exactly `d` times a
  difference of two scalar distortions (`ScalarGap.hadRate_sub_rhoSq`).  §49 reduces that
  difference in turn to the second moments of the two laws, their cell masses and their cell
  first moments, and to nothing else (`ScalarGap.distortion_congr_of_cells`).

  §50 to §56 sharpen the cell sum into its threshold form.  For an ordered codebook with
  thresholds at the midpoints, the gap is exactly `−2 ∑ₖ (c_{k+1} − cₖ)` times the
  once-integrated CDF difference at `τₖ` (`Threshold.distortion_sub_distortion_eq_threshold`),
  and a sup-norm bound `η` on that CDF difference therefore bounds the gap by
  `2η·spread·(b − a)` (`Threshold.abs_distortion_sub_distortion_le`), which at the deployed
  rate is `d` times that (`Threshold.abs_hadRate_sub_rhoSq_le`).

  §63 reads that identity in the other direction as well, and the reading costs nothing new: the
  gap is a function of the `K` numbers `∫_{(a,τₖ]}(F_μ − F_ν)` alone, so two pairs of laws that
  agree there have the same gap whatever they do elsewhere
  (`Threshold.distortion_sub_distortion_congr_of_cdfGapInt`), the upper bound is
  `2·(∑ₖ|c_{k+1} − cₖ|)·maxₖ|cdfGapInt|` with the codeword total variation in place of the
  spread, equal to it for an ordered codebook (`Threshold.codewordTV_eq_spread`), and its
  contrapositive `Threshold.exists_abs_cdfGapInt_gt` makes smallness at the `K` thresholds
  necessary and not only sufficient.  Two things that does **not** give.  It gives no two-sided
  estimate by a single quantity: the lower half
  (`Threshold.le_abs_distortion_sub_distortion`, and the sandwich
  `Threshold.abs_distortion_sub_distortion_sandwich`) subtracts the other `K − 1` terms and is
  often negative, because `Threshold.not_exists_pos_lower_bound_of_max` shows the terms can
  cancel exactly, so no lower bound in terms of `maxₖ|cdfGapInt|` alone can hold.  And that
  cancellation is proved about the identity's right-hand side as a function of the `K` numbers,
  **not** about a pair of laws: no measures are constructed for it, and whether every vector of
  `K` numbers is realised by a pair of laws meeting §55's hypotheses is not addressed anywhere
  here.  §63 also supplies no value for any of the `K` numbers, and uses no smoothness of the
  stage error `t ↦ (t − q t)²`: it is the identity plus the triangle inequality and nothing else.

  §61 gives that smoothness a name without spending it here.  The stage error
  `ψ t = (t − q t)²` is continuous everywhere and Lipschitz with constant `2M` wherever
  `\|t − q t\| ≤ M` (`Threshold.continuous_stageErr`, `Threshold.lipschitzOnWith_stageErr`), even
  though the quantiser `q` itself genuinely jumps by the full cell width at every threshold
  (`Threshold.stepQuant_jump`, `Threshold.not_continuousAt_stepQuant`).  §61 proves no bound on a
  distortion gap and no bound on the deployed rate, and is composed with §55's or §63's identity
  nowhere.  The only free inequality between the Lipschitz constant and `ρ²` is `ρ² ≤ d·L²/4`; no
  converse is proved and none is free, since a law can put almost all its mass where the error is
  tiny and a vanishing amount where it is large (`Threshold.le_rhoSq_of_mass` records exactly what
  a converse would cost).

  §64 spends §61's Lipschitz constant on a second route to `\|hadRate − ρ²\|`, through the
  Wasserstein-1 distance rather than the threshold sum, and the comparison is negative.
  `Transport.abs_hadRate_sub_rhoSq_le_wOne` is the transport bound, the easy half of
  Kantorovich–Rubinstein duality with `W₁` built from scratch as an infimum of coupling costs;
  `Transport.abs_cdfGapInt_le_wOne` shows the threshold currency is never the more expensive of
  the two, so the comparison collapses to the prefactors `2·spread cbk` against `L`.  At `d = 512`
  with the shipped 2-bit codebook the threshold route wins by a checkable factor at least `6.95`,
  `Θ(√d)` in general (`Transport.threshold_le_transport_512`), and the transport route is in any
  case not closed here: only one of its two legs, `Transport.wOne_sphericalCoordLaw_gaussian_le`
  (§57's coupling leg repackaged), exists in the transport currency, since §58's leg bounds
  `cdfGapInt`, which does not imply a `W₁` bound.

  Two things that chain does **not** do on its own, and how far each has since been closed; the
  paragraph above stands either way.  It does not make the right-hand side vanish: that the
  deployed rotation attains `ρ²` is measured in the paper rather than proved here, and §37's
  fourth-moment gap is one comparison of the same two laws, under `t ↦ t⁴`, rather than a bound
  on the distortion gap; no theorem here derives the second from the first.  And the bound of §55
  is an **implication**, not a number: `η` is a free variable, and nothing here discharges it for
  `(radLaw d v, sphericalCoordLaw d)`.  §44's Kolmogorov–Smirnov bound is against a *Gaussian*,
  which is a different second law, so no `η` below the trivial `η = 1` is available and nothing
  is bounded uniformly in the direction `v`.  §57 now proves the second leg, `f_d` against that
  Gaussian, but in a **different currency**: it bounds the once-integrated difference
  `cdfGapInt` by `√(1 + 1/(4(d−1)))/d` through a coupling
  (`abs_cdfGapInt_sphericalCoordLaw_gaussian_le`), not the sup-norm `|F − Φ|` that the original
  §55 and §56 bounds consume.  §55 and §56 now also carry a second, parallel route that reads the
  threshold sum against a uniform `cdfGapInt` bound directly, with no `(b − a)` factor
  (`Threshold.abs_distortion_sub_distortion_le_of_cdfGapInt`,
  `Threshold.abs_hadRate_sub_rhoSq_le_of_cdfGapInt`), alongside the sup-norm theorems, which are
  untouched.  §58 supplies the missing second leg in that same currency, `coordLaw H v i` against
  the same Gaussian, by Fourier inversion of the once-integrated gap rather than by a coupling,
  and §59 takes the triangle inequality: `cdfGapInt` telescopes through any third law, so the
  Gaussian cancels and `abs_cdfGapInt_coordLaw_sphericalCoordLaw_le` bounds the pair
  `(coordLaw H v i, sphericalCoordLaw d)` uniformly in `a` and `x`.  Feeding that to the new route
  discharges `hC`, and `Threshold.abs_hadRate_sub_rhoSq_le_unconditional` bounds the deployed rate
  against `ρ²` with no undischarged analytic hypothesis.  Three things that does **not** settle.
  The general bound is `Θ(√d · spread cbk)` at a spiky direction and so does not vanish; only the
  spread instance `Threshold.abs_hadRate_sub_rhoSq_le_of_spread`, `(30/√d + 7)·spread cbk`, loses
  the growing factor, and its primed form `Threshold.abs_hadRate_sub_rhoSq_le_of_spread'`,
  `(21/√d + 28/5)·spread cbk`, which
  uses §58a's exact Dirichlet constant, does the same with a constant `21.6%` smaller.
  `spread cbk` is left abstract there, so the vanishing is conditional on the codebook being
  scaled with `d` and is not proved for any codebook here.  And it remains an
  inequality: that the deployed rotation attains `ρ²` is not proved, and the sup-norm route is
  still open, since no Kolmogorov-Smirnov bound between the two coordinate laws exists in this
  development.  It also does not discharge the sandwich where §36 *uses*
  it: `integral_norm_resid_sq_le_had` ranges over every unit direction, so the stream's
  geometric decay still needs the extreme values of `ρ_H²` over the sphere, and §48c supplies
  an identity per direction, not a bound uniform in the direction.  §51 settles what that gap
  is: it is not open work, because the uniform bound is **false**.  At a coordinate direction
  `v = eⱼ` the deployed coordinate law is the two-point law on `±1/√d` (§48e, §62), all of
  its mass on the root-mean-square value where `f_d` spreads around it, so §48c's quadrature has
  two nodes and the rate is exactly
  `d·((1/√d − cb(1/√d))² + (−1/√d − cb(−1/√d))²)/2` (`CoordSpike.hadRate_blockStage_single`),
  the same number for every flat `H` and every coordinate.  Every `η` that bounds
  `|hadRate − ρ²|` over the whole unit sphere is therefore at least the distance from that number
  to `ρ²` (`CoordSpike.abs_coordGap_le_of_uniform`), and no `η` below it bounds anything
  (`CoordSpike.not_uniform_of_lt_coordGap`).  Both are unconditional and codebook-agnostic; the
  arithmetic that makes the cap a number, `0.2605` against `ρ² = 0.1171` for the shipped 2-bit
  codebook and the same factor `2.22` at `d = 256`, `512` and `1024`, is a computation in the
  paper, since no codebook and no Lloyd-Max level is formalised anywhere here.  Three things §51
  does **not** say.  It does not say the deployed stream fails to decay: `Hadamard.hadRate_le`
  bounds the rate at every direction by `(1 + R)²`, so the sandwich still has finite ends, and
  what is ruled out is only that both ends can be taken close to the Haar rate.  It does not
  weaken §59 or §60, whose hypotheses `eⱼ` fails: `eⱼ` is the spikiest unit direction,
  `∑ⱼ vⱼ⁴ = 1` and `maxⱼ|vⱼ|·√d = √d`, so it meets the near-flat hypothesis only at `c = √d`
  where that bound is `Θ(√d)` and asserts nothing.  And it says nothing about whether the
  deployed pipeline is good, only about what can be proved of it in this shape.
  §60 widens the spread hypothesis from `|vⱼ| ≤ d^{-1/2}`, which at `‖v‖ = 1` is an equality and
  so names `2^d` points, to `|vⱼ| ≤ c·d^{-1/2}` for any `c`, and carries `c` to the conclusion:
  `Threshold.abs_hadRate_sub_rhoSq_le_of_nearFlat` reads
  `2·(3c²/√d + (13/10)c + 1 + 1/(4(d − 1)))·spread cbk`, whose constant at `c = 1` is `4.87` at
  `d = 512` against §59's `8.33`.  That does **not** close the gap just described.  The set is
  now a sup-norm neighbourhood rather than a finite set, but it is not of full measure on the
  sphere: a Haar-random direction has `maxⱼ|vⱼ|·√d` near `√(2 log d)`, about `3.5` at `d = 512`,
  where the bound is already above the free bound `1`.  So `integral_norm_resid_sq_le_had` is
  still not fed, and by §51 no widening of `c` will feed it with a bound that stays near `ρ²`:
  at `c = √d` the near-flat class already contains the coordinate directions, and there
  `CoordSpike.abs_coordGap_le_of_uniform` caps every uniform `η` from below.  The `(13/10)c`
  term, which comes from the range on which the atomic law's characteristic function is
  uncontrolled, is what forbade taking `c` large, and §65 is the section that attacks it.

  §65 replaces that range's trivial estimate and, for the first time here, gets below the free
  bound at a direction that is typical rather than exceptionally flat.  The `13/10` is `(2/π)·2`
  and the `2` is `‖φ_μ − φ_ν‖ ≤ 2`, charged past `u = √d/M` because nothing about `φ_μ` was known
  there.  Something is: `φ_μ(u) = ∏ⱼ cos(aⱼu)`, and the reason the range was surrendered is that
  Mathlib's `Real.cos_bound` gives `|cos x| ≤ e^{−x²/4}` only for `|x| ≤ 1`, which is exactly
  where the split sits.  `abs_cos_mul_le` proves the periodic form
  `|cos(b·s)| ≤ exp(−b²·min(s, π−s)²/2)` for `1 ≤ s ≤ π`, which sees the reflection and survives
  to `π`; `prod_abs_cos_le` damps every coordinate at once with `∑ⱼbⱼ² = 1/M²`; and
  `integral_gapWeight_le_tailsharp` is the five-piece integral, giving
  `K(M) = 1/π + (2√(2π)/π²)·M + 16M⁴` in place of `2`.  At `d = 512` and `c = 3.53` that is
  `0.41`, so `13/10` becomes `13/50`, and
  `Threshold.abs_hadRate_sub_rhoSq_le_of_nearFlat_tail_512` reads `24/25`, true value `0.95`,
  where §60 reads `1.93`.  §60's statements are untouched and
  `FourierGap.integral_gapWeight_le_sharp` is kept beside the new one.

  Four things §65 does **not** do.  It does not reach `ρ² = 0.1171`, which is about eight times
  smaller, and it cannot at this dimension however good leg 1 becomes, because §57's coupling leg
  alone contributes `2·spread cbk = 0.267`.  It does not remove the hypothesis on `v`: `c ≤ 3.53`
  is about `√(2 log d)`, which is where a Haar-random coordinate maximum concentrates, but the
  theorem takes it as a hypothesis and proves nothing about Haar measure.  It does not cover the
  spikiest groups: at `c = 6.09` the bound is `2.00` and asserts nothing, so §36's sandwich over
  every direction is still not fed, and §51 says it never can be.  And the `1/π` is a floor for
  this route rather than remaining slack, since past `π√d/M` the cosine product genuinely returns
  to modulus one and a trivial bound there is necessary.

  §66 does the same job on the other side of the split.  `gapWeight_le_mid_sharp` charges the
  middle range `(40/9)·∑ⱼvⱼ⁴/u²`, and `40/9` is the **supremum** of the damping factor in
  `norm_gap_le_damped`, not its integral; the fix is to integrate it, which that lemma's own
  docstring names and does not do.  What it needs is the Gaussian mass past the split point, and
  Mathlib has no error function, so a tail estimate would give away a factor of two.  The
  complement does not: the half-line integral is exact by `integral_gaussian_Ioi`, and on the
  bounded piece `1 + x ≤ eˣ` is enough, giving
  `∫_{1/2}^∞ e^{−t²} ≤ √π/2 − 11/24 = 0.42789` against a true `0.42495` (`gaussTail_half_le`).
  The middle range integrates to `1.6346·∑ⱼvⱼ⁴/√d` against `4.4444`, within `0.4%` of the best
  this route can give, so `83/18` falls to `A_mid = e^{−1/4} + √π − 3/4` and the leg-1 quadratic
  coefficient from `2.9355`, shipped as `3`, to `1.1468`
  (`integral_gapWeight_le_midsharp`).  At `d = 512` and `c = 3.53` the composed bound falls from
  `0.952` to `0.679` (`abs_hadRate_sub_rhoSq_le_of_nearFlat_mid_512`, `69/100`).

  What §66 leaves.  The gain is confined to the spikiness coefficient; §65's `K(M)` is carried
  through unchanged and the `M`-linear tail term is now the **dominant** one at a near-flat
  direction, `(2/π)·K·c = 0.915` of the `2.547` inner sum at `d = 512`, so that is where any
  further gain has to come from and this development has none.  The rate in `d` is exactly what
  it was, `Θ(spread cbk)` and not `o(1)`.  `ρ² = 0.1171` is not reached and cannot be on this
  route, since §57's coupling leg alone contributes `2·spread cbk = 0.267`; that leg's constant
  is `√(1 + 1/(4(d−1)))`, resting on `Var‖Z‖ ≤ 1` from the Poincaré inequality; §67 takes that
  last piece.

  §67 halves the coupling leg's constant without the Gamma function.  Poincaré cannot be improved
  there, since `1` is sharp for a `1`-Lipschitz function; but `E‖Z‖² = d` exactly, so the whole
  estimate is a lower bound on `E‖Z‖`, and §26's radial integrals supply a better one with no new
  analysis.  Polar factorisation over the total mass cancels the sphere factor, giving
  `E‖Z‖ = R_d/R_{d−1}`; §26's `R_{n+2} = (n+1)R_n` then reads `ρ_d·ρ_{d+1} = d`; and `rho_sq_ge`
  turns that plus the crude `ρ_d² ≥ d − 1` into `ρ_d² ≥ d − 1/2`, because the recurrence makes
  `ρ_d²/(d − 1/2)` decrease in steps of two while the crude bound pins its limit at `1`.  Forward
  induction alone cannot do it, and `Real.Gamma` is never mentioned.  So `Var‖Z‖ ≤ 1/2`
  dimension-free and the leg carries `√(1/2 + 1/(8(d−1)))/d`, a factor `1/√2`.

  Four things §67 does **not** do.  The variance bound is one-sided: no lower bound, nothing about
  `Var(‖Z‖²)`, and `1/2` is approached rather than attained, so nothing is sharp at finite `d`.
  It does not evaluate `E‖Z‖`.  It improves neither §65's `K(M)` nor §66's `A_mid`, which are
  carried through unchanged, and the rate in `d` is exactly what it was.  And it does not bring
  `ρ²` within reach: the floor of the route is `1/√2` rather than `1/2`, since the leg is an `L²`
  cost and the square root enters, so at `d = 512` the coupling leg alone still contributes
  `2·spread cbk/√2 = 0.189` of the composed `0.601`, which is `1.6×ρ²` and is the best this shape
  of proof reaches at that dimension
  `gapWeight_le_mid_sharp`, `integral_gapWeight_le_sharp` and `integral_gapWeight_le_tailsharp`
  are untouched and not weakened
* Remark ("Concentration"): this was once outside the development and is now inside it.
  §25–§35 prove the variance estimate the remark rests on, `variance_stageErr_le_bounded`,
  `Var ‖u − q u‖² ≤ 4ρ²/(d − 2)` for every bounded nearest-codeword stage and every `d ≥ 3`,
  by a Gaussian Poincaré inequality with constant one, the Dunkl–Williams sharp slope and an
  `η`-net from finite codebooks to bounded ones.  §13's downstream compounding is therefore no
  longer conditional: `meas_not_decayWithin_le_bounded` carries the `(1 ± ε)^P` product, the
  union bound `Pδ` and the Chebyshev step through to a rate.  The hypotheses of those theorems,
  not a comment, are now the statement of what they need.
* The dimension hypotheses are not cosmetic.  §47 proves that Lemma 1's `2 ≤ d` cannot be
  weakened: at `d = 1` the unit sphere is two points, so the coordinate law is singular with
  respect to Lebesgue measure and no density exists.  §13's `3 ≤ d` is arithmetic rather than
  measure-theoretic, since the bound divides by `d − 2`.
-/

end Progressive
