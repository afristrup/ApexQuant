
/-! ## §67  The coupling leg's constant halved: `Var(‖Z‖) ≤ 1/2` from the radial recurrence

`GaussianRadius.integral_sq_sqrt_finrank_sub_norm_le` charges `E[(√d − ‖Z‖)²] ≤ 1 + 1/(4(d−1))`,
and that `1` is the Poincaré constant: `variance_norm_gaussianE_le` bounds `Var(‖Z‖)` by `1`
because `‖·‖` is `1`-Lipschitz and `1` is the sharp Gaussian Poincaré constant for a `1`-Lipschitz
function.  The truth is `Var(‖Z‖) ≤ 1/2`, and at `d = 512` it is `0.49976`.  Since
`E‖Z‖² = d` exactly (`integral_normSq_gaussianE`), everything in that estimate is a *lower bound
on `E‖Z‖`*, and Poincaré gives only `E‖Z‖ ≥ √(d−1)` where `√(d − 1/2)` is available.  This section
supplies `√(d − 1/2)`, and with it the `+1` of the composed near-flat bound falls to `1/√2`.

**What is replaced, and by what.**  Not the Poincaré inequality: the route goes through the radial
integrals `R_n = ∫₀^∞ rⁿe^{−r²/2} dr` of §26, and touches no new analysis at all.  Polar
factorisation (`lintegral_gaussRaw_polar` at `ψ = 1`, `φ = id`) and the total mass
(`gaussRaw_univ`) carry the same sphere factor, which cancels in the normalisation, so

  `E‖Z‖ = R_d / R_{d−1} =: ρ d`   (`integral_norm_gaussianE_eq_radGauss_div`)

for every space of finrank `d ≥ 1`.  §26's integration by parts `R_{n+2} = (n+1)R_n`
(`radGauss_succ_succ`) then reads `ρ d · ρ (d+1) = d` with no analysis whatever
(`rhoRad_mul_rhoRad_succ`).  `rho_sq_ge` is the abstract consequence: a positive sequence obeying
that recurrence and the crude bound `d − 1 ≤ ρ d ²` obeys `d − 1/2 ≤ ρ d ²`.  The mechanism is
that the recurrence forces `m·ρ(m+2) = (m+1)·ρ m`, so `ρ_d²/(d − 1/2)` **decreases** in steps of
two because `(m+1)²(m − 1/2) = m²(m + 3/2) − 1/2`, while the crude bound pins its limit at `1`;
a decreasing sequence with limit `1` is at least `1`.  Forward induction alone cannot prove this,
which is why the classical proofs go through an integral representation or an asymptotic
expansion.  **`Real.Gamma` is never mentioned**, and no Gamma-function inequality is used or
needed.

**The constant.**  `integral_sq_sqrt_finrank_sub_norm_le_half` reads

  `E[(√d − ‖Z‖)²] = 2√d(√d − E‖Z‖) ≤ 2√d(√d − √(d − 1/2)) = (√d − √(d−1/2))² + 1/2`
                  `≤ 1/2 + 1/(8(d−1))`,

the last step because `(√d − √(d−1/2))²(√d + √(d−1/2))² = 1/4` and `(√d + √(d−1/2))² ≥ 2(d−1)`.
So the coupling leg `abs_cdfGapInt_sphericalCoordLaw_gaussian_le_half` carries
`√(1/2 + 1/(8(d−1)))/d` in place of `√(1 + 1/(4(d−1)))/d`: a factor `1/√2 = 0.7071` and nothing
more.  The floor of the whole route is `1/√2`, not `1/2`: the leg is an `L²` cost and it is the
square root that enters.  `sqrt_half_add_eighth_le` is the analogue of
`NearFlat.sqrt_one_add_quarter_le_self`, and it goes the other way: `√x ≤ x` is false below `1`,
so the rational cover is `√(1/2 + q) ≤ 5/7 + q`, valid because `(5/7)² = 25/49 ≥ 1/2`.  `5/7` is
`1.0%` above `1/√2` and that `1.0%` is the only slack this section gives away.

**What is *not* done.**  `Var(‖Z‖) ≤ 1/2` is an upper bound only: no lower bound on the variance
is proved, `Var(‖Z‖²)` is not bounded, and the true `1/2 − 1/(8d) + O(d^{−2})` is not attained,
so nothing here is claimed sharp at finite `d`.  `E‖Z‖` is not evaluated; the recurrence would
also give a matching **upper** bound `E‖Z‖ ≤ √d`, which is not stated because the coupling needs
only the lower one.  The rate in `d` is exactly as it was: the coupling leg is still `Θ(1/d)`,
the composed near-flat bound is still `Θ(spread cbk)` and still not `o(1)`, and it still vanishes
only if the codebook is placed at the coordinate scale, which remains a hypothesis about `cbk`
proved for no codebook anywhere in this development.  §65's tail constant `K(M)` and §66's
spikiness constant `A_mid` are carried through **unchanged** and neither is improved.
`variance_norm_gaussianE_le`, `le_integral_norm_gaussianE`,
`integral_sq_sqrt_finrank_sub_norm_le`, `abs_cdfGapInt_sphericalCoordLaw_gaussian_le` and every
§60, §65 and §66 statement are kept untouched, still proved, and not weakened.

**Numerically, at the dimension the paper ships.**  At `d = 512` the leg constant falls from
`√(1 + 1/2044) = 1.00024` to `√(1/2 + 1/4088) = 0.70728`, so the coupling leg's share of the
composed bound, `2·spread cbk = 0.267` at a spread of `3.0164/√512`, falls to `0.189`.  With
§66's `A_mid` and §65's `K(M)` at `c = 3.53 ≈ √(2 log 512)`, the composed factor
`2·((2/π)A_mid·c²/√d + (2/π)K·c + 1/√2 + …)` falls from `5.093` to `4.507` and the bound from
`0.679` to `0.601`; the chain of rational roundings gives `0.6053`, and
`Threshold.abs_hadRate_sub_rhoSq_le_of_nearFlat_half_512` records `61/100`.  That is still far
from `ρ² = 0.1171` and this route cannot reach it: `1/√2` is now the floor of the `+1` slot and
`2·spread cbk/√2 = 0.189` of the remaining `0.601` comes from the coupling leg alone.  The
codebook width is a hypothesis; no codebook is constructed, named or analysed here. -/

noncomputable section CouplingHalf

open Orthogonal Hadamard CoordLaw

/-! ### §67a  The bridge: `E‖Z‖` is a ratio of radial integrals

Two applications of §26's polar factorisation, one for the numerator and one for the total mass.
The sphere measure appears in both and cancels, so nothing here depends on its value, and no
integral over the sphere is evaluated. -/

section RadialBridge

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- `∫ ‖x‖ e^{−‖x‖²/2} dx = σ(S)·R_d` for `d = finrank ℝ E`: `lintegral_gaussRaw_polar` at
`ψ = 1` and `φ = id`, whose radial factor is `r^{d−1}·r = r^d` because `d ≥ 1`.  This is the
unnormalised measure `gaussRaw`; the sphere factor `σ(S) = volume.toSphere univ` is left
unevaluated and is not claimed to be finite or computable here beyond what §26 already proves. -/
theorem lintegral_norm_gaussRaw :
    ∫⁻ x : E, ENNReal.ofReal ‖x‖ ∂(gaussRaw E)
      = volume.toSphere (Set.univ : Set (Metric.sphere (0 : E) 1)) *
          ENNReal.ofReal (radGauss (Module.finrank ℝ E)) := by
  have hn : 1 ≤ Module.finrank ℝ E := Module.finrank_pos
  have h := lintegral_gaussRaw_polar (E := E) (ψ := fun _ => (1:ℝ≥0∞)) measurable_const
    (φ := fun r => ENNReal.ofReal r) (by fun_prop)
  simp only [one_mul] at h
  rw [lintegral_one] at h
  have hrad : (∫⁻ r in Set.Ioi (0:ℝ), ENNReal.ofReal (r ^ (Module.finrank ℝ E - 1)) *
        (ENNReal.ofReal r * ENNReal.ofReal (Real.exp (-(r ^ 2) / 2))))
      = ENNReal.ofReal (radGauss (Module.finrank ℝ E)) := by
    rw [← lintegral_radial (Module.finrank ℝ E)]
    refine setLIntegral_congr_fun measurableSet_Ioi fun r hr => ?_
    have hr0 : (0:ℝ) < r := hr
    have hpow : r ^ (Module.finrank ℝ E - 1) * r = r ^ (Module.finrank ℝ E) := by
      rw [← pow_succ]; congr 1; omega
    rw [← mul_assoc, ← ENNReal.ofReal_mul (pow_nonneg hr0.le _), hpow]
  rw [h, hrad]

/-- **`E‖Z‖ = R_d / R_{d−1}`** for the standard Gaussian on any space of finrank `d ≥ 1`, with
`R_n = radGauss n` the radial integral of §26.  Numerator and denominator carry the same sphere
factor (`lintegral_norm_gaussRaw` and `gaussRaw_univ`) and it cancels in the normalisation
`gaussianE = (gaussRaw univ)⁻¹ • gaussRaw`.  This is an identity, not an estimate: it evaluates
neither `R_d` nor the ratio, and it says nothing about `E‖Z‖` for any other law. -/
theorem integral_norm_gaussianE_eq_radGauss_div :
    ∫ x : E, ‖x‖ ∂(gaussianE E)
      = radGauss (Module.finrank ℝ E) / radGauss (Module.finrank ℝ E - 1) := by
  set S := volume.toSphere (Set.univ : Set (Metric.sphere (0 : E) 1)) with hSdef
  have hS0 : S ≠ 0 := toSphere_univ_ne_zero E
  have hSt : S ≠ ⊤ := measure_ne_top _ _
  have hR : 0 < radGauss (Module.finrank ℝ E) := radGauss_pos _
  have hR' : 0 < radGauss (Module.finrank ℝ E - 1) := radGauss_pos _
  have hraw : gaussRaw E Set.univ = S * ENNReal.ofReal (radGauss (Module.finrank ℝ E - 1)) :=
    gaussRaw_univ
  have hSr : 0 < S.toReal := ENNReal.toReal_pos hS0 hSt
  have hint : ∫ x : E, ‖x‖ ∂(gaussRaw E) = S.toReal * radGauss (Module.finrank ℝ E) := by
    rw [integral_eq_lintegral_of_nonneg_ae
      (Filter.Eventually.of_forall fun x => norm_nonneg x) (by fun_prop)]
    rw [lintegral_norm_gaussRaw, ENNReal.toReal_mul, ENNReal.toReal_ofReal hR.le]
  have hgE : gaussianE E = (gaussRaw E Set.univ)⁻¹ • gaussRaw E := rfl
  rw [hgE, integral_smul_measure, hint, hraw, ENNReal.toReal_inv, ENNReal.toReal_mul,
    ENNReal.toReal_ofReal hR'.le, smul_eq_mul]
  field_simp

end RadialBridge

/-! ### §67b  The recurrence, and what it forces

`rhoRad` is the ratio `R_d/R_{d−1}` as a sequence of reals.  Nothing in this block is a
statement about a measure: `rhoRad_mul_rhoRad_succ` is §26's integration by parts rearranged, and
`rho_sq_ge` is an argument about any sequence satisfying it. -/

section RadialRatio

/-- The radial ratio `ρ d = R_d/R_{d−1}`.  For `d ≥ 1` it is `E‖Z‖` in dimension `d`
(`rhoRad_eq_integral_norm`); `rhoRad 0 = R_0/R_0 = 1` carries no such meaning. -/
def rhoRad (n : ℕ) : ℝ := radGauss n / radGauss (n - 1)

/-- `ρ n > 0`, since every `R_n` is (`radGauss_pos`). -/
theorem rhoRad_pos (n : ℕ) : 0 < rhoRad n := div_pos (radGauss_pos _) (radGauss_pos _)

/-- **`ρ d · ρ (d+1) = d`** for `d ≥ 1`: the product telescopes to `R_{d+1}/R_{d−1}`, which is
`d` by §26's `radGauss_succ_succ` at `n = d − 1`.  No analysis is used; the integration by parts
was done in §26 and nothing is reproved here. -/
theorem rhoRad_mul_rhoRad_succ {d : ℕ} (hd : 1 ≤ d) : rhoRad d * rhoRad (d + 1) = (d : ℝ) := by
  obtain ⟨m, rfl⟩ : ∃ m, d = m + 1 := ⟨d - 1, by omega⟩
  have h1 : radGauss (m + 1) ≠ 0 := (radGauss_pos _).ne'
  have h2 : radGauss m ≠ 0 := (radGauss_pos _).ne'
  have hrec : radGauss (m + 1 + 1) = ((m:ℝ) + 1) * radGauss m := radGauss_succ_succ m
  simp only [rhoRad, Nat.add_sub_cancel]
  rw [div_mul_div_comm, hrec]
  field_simp
  push_cast
  ring

/-- `E‖Z‖ = ρ d` on `Ed d`, which is `integral_norm_gaussianE_eq_radGauss_div` read through
`finrank ℝ (Ed d) = d`.  It is this instantiation, and nothing about `Ed d` specifically, that
lets a statement about the sequence `ρ` be read as a statement about the Gaussian radius. -/
theorem rhoRad_eq_integral_norm {d : ℕ} (hd : 1 ≤ d) :
    ∫ x : Ed d, ‖x‖ ∂(gaussianE (Ed d)) = rhoRad d := by
  haveI : Nontrivial (Ed d) := Module.nontrivial_of_finrank_pos (R := ℝ) (by
    simpa [Ed] using (by omega : 0 < d))
  have hdn : Module.finrank ℝ (Ed d) = d := by simp [Ed]
  rw [integral_norm_gaussianE_eq_radGauss_div, hdn, rhoRad]

/-- The crude bound `d − 1 ≤ ρ d ²`, which is §57b's `le_integral_norm_gaussianE` (Gaussian
Poincaré) read through `rhoRad_eq_integral_norm` for `d ≥ 2`, and trivial at `d = 1`.  This is
the *hypothesis* of `rho_sq_ge`, not its conclusion, and it is the only place where an analytic
input enters the improvement. -/
theorem rhoRad_sq_ge_sub_one {d : ℕ} (hd : 1 ≤ d) : (d : ℝ) - 1 ≤ rhoRad d ^ 2 := by
  rcases lt_or_ge d 2 with h | h
  · have hd1 : d = 1 := by omega
    subst hd1
    have := sq_nonneg (rhoRad 1)
    norm_num
    linarith
  · haveI : Nontrivial (Ed d) := Module.nontrivial_of_finrank_pos (R := ℝ) (by
      simpa [Ed] using (by omega : 0 < d))
    have hdn : Module.finrank ℝ (Ed d) = d := by simp [Ed]
    have hlow := le_integral_norm_gaussianE (E := Ed d) (by rw [hdn]; exact h)
    rw [hdn, rhoRad_eq_integral_norm hd] at hlow
    have hdR : (1:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd.trans' (by omega)
    have hs : Real.sqrt ((d:ℝ) - 1) ^ 2 = (d:ℝ) - 1 := Real.sq_sqrt (by linarith)
    nlinarith [Real.sqrt_nonneg ((d:ℝ) - 1), hlow, hs]

/-- **The abstract core.**  A sequence with `ρ d · ρ (d+1) = d` and the crude bound
`d − 1 ≤ ρ d ²` satisfies `d − 1/2 ≤ ρ d ²`.

The recurrence gives `m·ρ(m+2) = (m+1)·ρ m`, so `A_m = ρ_m²/(m − 1/2)` satisfies
`m²·A_{m+2}(m + 3/2) = (m+1)²·A_m(m − 1/2)` and hence `A_{m+2} ≤ A_m`, because
`(m+1)²(m − 1/2) = m²(m + 3/2) − 1/2`.  Along the arithmetic progression `d, d+2, d+4, …` the
crude bound forces `A_{d+2k} ≥ 1 − (1/2)/(d + 2k − 1/2) → 1`, so `A_d < 1` is impossible.

No positivity, monotonicity or growth hypothesis beyond these two is used, and nothing is claimed
in the other direction: **no upper bound on `ρ d ²` is proved**, and the constant `1/2` is not
shown optimal for an abstract sequence.  Forward induction from `d − 1 ≤ ρ d ²` alone cannot give
this, since the recurrence relates `d` to `d+1` and the crude bound is consistent with equality
at any single `d`. -/
theorem rho_sq_ge {ρ : ℕ → ℝ}
    (hrec : ∀ d : ℕ, 1 ≤ d → ρ d * ρ (d + 1) = (d : ℝ))
    (hlow : ∀ d : ℕ, 1 ≤ d → (d : ℝ) - 1 ≤ ρ d ^ 2) :
    ∀ d : ℕ, 1 ≤ d → (d : ℝ) - 1 / 2 ≤ ρ d ^ 2 := by
  have hstep : ∀ m : ℕ, 1 ≤ m → (m : ℝ) * ρ (m + 2) = ((m : ℝ) + 1) * ρ m := by
    intro m hm
    have h1 := hrec m hm
    have h2 := hrec (m + 1) (by omega)
    have he : m + 1 + 1 = m + 2 := rfl
    rw [he] at h2
    push_cast at h2
    calc (m : ℝ) * ρ (m + 2) = (ρ m * ρ (m + 1)) * ρ (m + 2) := by rw [h1]
      _ = ρ m * (ρ (m + 1) * ρ (m + 2)) := by ring
      _ = ρ m * ((m : ℝ) + 1) := by rw [h2]
      _ = ((m : ℝ) + 1) * ρ m := by ring
  have hdec : ∀ m : ℕ, 1 ≤ m →
      ρ (m + 2) ^ 2 / (((m : ℝ) + 2) - 1 / 2) ≤ ρ m ^ 2 / ((m : ℝ) - 1 / 2) := by
    intro m hm
    have hm1 : (1 : ℝ) ≤ (m : ℝ) := by exact_mod_cast hm
    have hsq : (m : ℝ) ^ 2 * ρ (m + 2) ^ 2 = ((m : ℝ) + 1) ^ 2 * ρ m ^ 2 := by
      have h := hstep m hm
      have h2 := congrArg (fun t : ℝ => t ^ 2) h
      simpa [mul_pow] using h2
    rw [div_le_div_iff₀ (by linarith) (by linarith)]
    nlinarith [hsq, sq_nonneg (ρ m), sq_nonneg (ρ (m + 2)), hm1]
  intro d hd
  have hd1 : (1 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
  have hchain : ∀ k : ℕ,
      ρ (d + 2 * k) ^ 2 / (((d : ℝ) + 2 * k) - 1 / 2) ≤ ρ d ^ 2 / ((d : ℝ) - 1 / 2) := by
    intro k
    induction k with
    | zero => simp
    | succ k ih =>
      have hmk : 1 ≤ d + 2 * k := by omega
      have h := hdec (d + 2 * k) hmk
      have he : d + 2 * k + 2 = d + 2 * (k + 1) := by ring
      rw [he] at h
      have hc : ((d + 2 * k : ℕ) : ℝ) + 2 = ((d : ℝ) + 2 * ((k : ℕ) + 1 : ℕ)) := by push_cast; ring
      rw [hc] at h
      have hc2 : ((d + 2 * k : ℕ) : ℝ) = (d : ℝ) + 2 * k := by push_cast; ring
      rw [hc2] at h
      exact h.trans ih
  by_contra hcon
  rw [not_le] at hcon
  set A : ℝ := ρ d ^ 2 / ((d : ℝ) - 1 / 2) with hA
  have hAlt : A < 1 := by
    rw [hA, div_lt_one (by linarith)]
    linarith
  have hApos : 0 < 1 - A := by linarith
  obtain ⟨k, hk⟩ := exists_nat_gt ((1 / 2) / (1 - A))
  have hk' : (1 : ℝ) / 2 < (k : ℝ) * (1 - A) := by
    rw [div_lt_iff₀ hApos] at hk
    linarith
  have hk0 : (0 : ℝ) ≤ (k : ℝ) := Nat.cast_nonneg k
  have hdk : (0 : ℝ) < ((d : ℝ) + 2 * k) - 1 / 2 := by linarith
  have hkey := hchain k
  rw [div_le_iff₀ hdk] at hkey
  have hlowk : ((d : ℝ) + 2 * k) - 1 ≤ ρ (d + 2 * k) ^ 2 := by
    have h := hlow (d + 2 * k) (by omega)
    push_cast at h
    linarith
  have hmain : ((d : ℝ) + 2 * k) - 1 ≤ A * (((d : ℝ) + 2 * k) - 1 / 2) := le_trans hlowk hkey
  have hXk : (k : ℝ) ≤ ((d : ℝ) + 2 * k) - 1 / 2 := by linarith
  nlinarith [hmain, hk', hXk, hApos]

/-- **`d − 1/2 ≤ ρ d ²`**, the core at the radial ratio.  Both hypotheses are discharged:
`rhoRad_mul_rhoRad_succ` is §26's integration by parts, `rhoRad_sq_ge_sub_one` is §57b's Poincaré
bound.  No upper bound on `ρ d ²` is proved, and the value of `ρ d` is not computed. -/
theorem rhoRad_sq_ge {d : ℕ} (hd : 1 ≤ d) : (d : ℝ) - 1 / 2 ≤ rhoRad d ^ 2 :=
  rho_sq_ge (fun _ hn => rhoRad_mul_rhoRad_succ hn) (fun _ hn => rhoRad_sq_ge_sub_one hn) d hd

/-- The arithmetic behind the halved radius estimate, isolated from any measure theory: if
`√(n − 1/2) ≤ m` then `2n − 2√n·m ≤ 1/2 + 1/(8(n−1))`.  Writing `s = √n` and `r = √(n − 1/2)`,
`2n − 2sr = (s − r)² + 1/2` exactly, and `(s − r)²(s + r)² = (s² − r²)² = 1/4` with
`(s + r)² = 2n − 1/2 + 2sr ≥ 2(n − 1)`, so `(s − r)² ≤ 1/(8(n−1))`.  No lower bound on
`2n − 2√n·m` is claimed and `m` is not assumed to be any particular quantity. -/
theorem two_mul_sub_two_mul_sqrt_mul_le_half {n m : ℝ} (hn2 : 2 ≤ n)
    (hmr : Real.sqrt (n - 1 / 2) ≤ m) :
    2 * n - 2 * Real.sqrt n * m ≤ 1 / 2 + 1 / (8 * (n - 1)) := by
  set s := Real.sqrt n with hsdef
  set r := Real.sqrt (n - 1 / 2) with hrdef
  have hs2 : s ^ 2 = n := Real.sq_sqrt (by linarith)
  have hr2 : r ^ 2 = n - 1 / 2 := Real.sq_sqrt (by linarith)
  have hs0 : 0 ≤ s := Real.sqrt_nonneg _
  have hr0 : 0 ≤ r := Real.sqrt_nonneg _
  have h1 : 2 * n - 2 * s * m ≤ 2 * n - 2 * s * r := by nlinarith
  have h2 : 2 * n - 2 * s * r = (s - r) ^ 2 + 1 / 2 := by nlinarith
  have h3 : 2 * (n - 1) ≤ (s + r) ^ 2 := by nlinarith [mul_nonneg hs0 hr0]
  have hprod : (s - r) * (s + r) = 1 / 2 := by nlinarith
  have h4 : (s - r) ^ 2 * (s + r) ^ 2 = 1 / 4 := by
    rw [← mul_pow, hprod]; norm_num
  have hpos : (0:ℝ) < 8 * (n - 1) := by linarith
  have h5 : (s - r) ^ 2 ≤ 1 / (8 * (n - 1)) := by
    rw [le_div_iff₀ hpos]
    nlinarith [mul_le_mul_of_nonneg_left h3 (sq_nonneg (s - r)), h4]
  linarith

end RadialRatio

/-! ### §67c  The payoff: half the variance, half the second moment

These three restate §57b's estimates with the sharper mean.  They are new declarations beside the
old ones; §57b's `variance_norm_gaussianE_le`, `le_integral_norm_gaussianE` and
`integral_sq_sqrt_finrank_sub_norm_le` are untouched and remain proved. -/

section GaussianRadiusHalf

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- **`E‖Z‖ ≥ √(d − 1/2)`** for the standard Gaussian on any space of finrank `d ≥ 2`, against
`√(d − 1)` from Poincaré (`le_integral_norm_gaussianE`).  `E‖Z‖` is the radial ratio
(`integral_norm_gaussianE_eq_radGauss_div`) and `rhoRad_sq_ge` bounds its square.  No matching
upper bound is proved here, and the exact value is not computed. -/
theorem le_integral_norm_gaussianE_half (hd : 2 ≤ Module.finrank ℝ E) :
    Real.sqrt ((Module.finrank ℝ E : ℝ) - 1 / 2) ≤ ∫ x, ‖x‖ ∂(gaussianE E) := by
  rw [integral_norm_gaussianE_eq_radGauss_div]
  have hr : radGauss (Module.finrank ℝ E) / radGauss (Module.finrank ℝ E - 1)
      = rhoRad (Module.finrank ℝ E) := rfl
  rw [hr]
  have h := rhoRad_sq_ge (d := Module.finrank ℝ E) (by omega)
  calc Real.sqrt ((Module.finrank ℝ E : ℝ) - 1 / 2)
      ≤ Real.sqrt (rhoRad (Module.finrank ℝ E) ^ 2) := Real.sqrt_le_sqrt h
    _ = rhoRad (Module.finrank ℝ E) := Real.sqrt_sq (rhoRad_pos _).le

/-- **`Var(‖Z‖) ≤ 1/2`**, against `1` from the Gaussian Poincaré inequality
(`variance_norm_gaussianE_le`), and dimension-free exactly as that one is.  It is
`Var = E‖Z‖² − (E‖Z‖)² = d − (E‖Z‖)²` with `E‖Z‖² = d` (`integral_normSq_gaussianE`) and the
sharper mean.  The true variance tends to `1/2` from below, so the constant cannot be improved
uniformly in `d`; but **no lower bound on the variance is proved here**, nothing is claimed about
`Var(‖Z‖²)`, and the `1/2 − 1/(8d) + O(d^{−2})` expansion is not established. -/
theorem variance_norm_gaussianE_le_half (hd : 2 ≤ Module.finrank ℝ E) :
    ProbabilityTheory.variance (fun x : E => ‖x‖) (gaussianE E) ≤ 1 / 2 := by
  have hn2 : (2:ℝ) ≤ (Module.finrank ℝ E : ℝ) := by exact_mod_cast hd
  have hf : MemLp (fun x : E => ‖x‖) 2 (gaussianE E) := by
    rw [memLp_two_iff_integrable_sq (by fun_prop)]
    simpa using integrable_normSq_gaussianE (E := E)
  have hveq := ProbabilityTheory.variance_eq_sub hf
  simp only [Pi.pow_apply] at hveq
  rw [integral_normSq_gaussianE] at hveq
  have hmr := le_integral_norm_gaussianE_half (E := E) hd
  have hsq : Real.sqrt ((Module.finrank ℝ E : ℝ) - 1 / 2) ^ 2
      = (Module.finrank ℝ E : ℝ) - 1 / 2 := Real.sq_sqrt (by linarith)
  nlinarith [Real.sqrt_nonneg ((Module.finrank ℝ E : ℝ) - 1 / 2), hmr, hsq, hveq]

/-- **`E[(√d − ‖Z‖)²] ≤ 1/2 + 1/(8(d−1))`**, halving §57b's `1 + 1/(4(d−1))`.  Since
`E‖Z‖² = d` the quantity is exactly `2√d(√d − E‖Z‖)`, so it is `le_integral_norm_gaussianE_half`
plus `two_mul_sub_two_mul_sqrt_mul_le_half`.  The bound tends to `1/2`, not to `0`: the radius
still does not concentrate at a scale better than `O(1)`, and nothing here claims it does.
`integral_sq_sqrt_finrank_sub_norm_le` is untouched. -/
theorem integral_sq_sqrt_finrank_sub_norm_le_half (hd : 2 ≤ Module.finrank ℝ E) :
    ∫ x : E, (Real.sqrt (Module.finrank ℝ E : ℝ) - ‖x‖) ^ 2 ∂(gaussianE E)
      ≤ 1 / 2 + 1 / (8 * ((Module.finrank ℝ E : ℝ) - 1)) := by
  have hn2 : (2:ℝ) ≤ (Module.finrank ℝ E : ℝ) := by exact_mod_cast hd
  rw [integral_sq_const_sub_norm, Real.sq_sqrt (by linarith)]
  have hmr := le_integral_norm_gaussianE_half (E := E) hd
  have h := two_mul_sub_two_mul_sqrt_mul_le_half hn2 hmr
  linarith

end GaussianRadiusHalf

/-! ### §67d  The coupling leg with the halved constant

The same coupling as §57c, `u = Z/‖Z‖` against `Z/√d`, and the same Cauchy-Schwarz split; only
the second factor changes.  Both proofs are §57c's verbatim, with the halved second moment
`integral_sq_sqrt_finrank_sub_norm_le_half` in place of `integral_sq_sqrt_finrank_sub_norm_le`. -/

section CouplingLegHalf

/-- **The `L¹` cost of the coupling, halved.**  `E|u_i − Z_i/√d| ≤ √(1/2 + 1/(8(d−1)))/d`, against
`√(1 + 1/(4(d−1)))/d` in §57c.  The rate is still `Θ(1/d)` and the gain is the constant factor
`1/√2` and nothing else.  This is an `L¹` statement about one coupling; it is not a bound on the
Kolmogorov-Smirnov distance of the two laws, and no such bound follows from it. -/
theorem integral_abs_dir_coord_sub_coord_div_sqrt_le_half {d : ℕ} (hd : 2 ≤ d) (i : Fin d) :
    ∫ z, |(dir z).ofLp i - z.ofLp i / Real.sqrt (d : ℝ)| ∂(gaussianE (Ed d))
      ≤ Real.sqrt (1 / 2 + 1 / (8 * ((d : ℝ) - 1))) / (d : ℝ) := by
  haveI : Nontrivial (Ed d) := Module.nontrivial_of_finrank_pos (R := ℝ) (by
    have : (0 : ℕ) < d := by omega
    simpa [Ed] using this)
  have hdn : Module.finrank ℝ (Ed d) = d := by simp [Ed]
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have hs : (0:ℝ) < Real.sqrt (d:ℝ) := Real.sqrt_pos.mpr hd0
  have hmf : Measurable (fun z : Ed d => (dir z).ofLp i) := measurable_dir_coord i
  have hpt : ∀ z : Ed d, |(dir z).ofLp i - z.ofLp i / Real.sqrt (d:ℝ)|
      = (Real.sqrt (d:ℝ))⁻¹ * (|(dir z).ofLp i| * |Real.sqrt (d:ℝ) - ‖z‖|) := by
    intro z
    have hfac : (dir z).ofLp i - z.ofLp i / Real.sqrt (d:ℝ)
        = (Real.sqrt (d:ℝ))⁻¹ * ((dir z).ofLp i * (Real.sqrt (d:ℝ) - ‖z‖)) := by
      rcases eq_or_ne z 0 with h | h
      · subst h; simp [dir]
      · have hz : ‖z‖ ≠ 0 := norm_ne_zero_iff.mpr h
        show ‖z‖⁻¹ * z.ofLp i - z.ofLp i / Real.sqrt (d:ℝ)
          = (Real.sqrt (d:ℝ))⁻¹ * (‖z‖⁻¹ * z.ofLp i * (Real.sqrt (d:ℝ) - ‖z‖))
        field_simp
    rw [hfac, abs_mul, abs_mul, abs_of_nonneg (by positivity : (0:ℝ) ≤ (Real.sqrt (d:ℝ))⁻¹)]
  have hMf : MemLp (fun z : Ed d => (dir z).ofLp i) 2 (gaussianE (Ed d)) := by
    rw [memLp_two_iff_integrable_sq hmf.aestronglyMeasurable]
    refine (integrable_const (1:ℝ)).mono' (hmf.pow_const 2).aestronglyMeasurable ?_
    filter_upwards with z
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _), ← sq_abs]
    nlinarith [abs_dir_coord_le_one i z, abs_nonneg ((dir z).ofLp i)]
  have hMg : MemLp (fun z : Ed d => Real.sqrt (d:ℝ) - ‖z‖) 2 (gaussianE (Ed d)) :=
    memLp_two_const_sub_norm _
  have hA : ∫ z, ((dir z).ofLp i) ^ 2 ∂(gaussianE (Ed d)) = 1 / (d : ℝ) :=
    integral_sq_dir_coord hd i
  have hB : ∫ z : Ed d, (Real.sqrt (d:ℝ) - ‖z‖) ^ 2 ∂(gaussianE (Ed d))
      ≤ 1 / 2 + 1 / (8 * ((d : ℝ) - 1)) := by
    have h := integral_sq_sqrt_finrank_sub_norm_le_half (E := Ed d) (by rw [hdn]; exact hd)
    rwa [hdn] at h
  have hsqrtA : Real.sqrt (1 / (d:ℝ)) = (Real.sqrt (d:ℝ))⁻¹ := by
    rw [one_div, Real.sqrt_inv]
  calc ∫ z, |(dir z).ofLp i - z.ofLp i / Real.sqrt (d:ℝ)| ∂(gaussianE (Ed d))
      = ∫ z, (Real.sqrt (d:ℝ))⁻¹
          * (|(dir z).ofLp i| * |Real.sqrt (d:ℝ) - ‖z‖|) ∂(gaussianE (Ed d)) :=
        integral_congr_ae (Filter.Eventually.of_forall hpt)
    _ = (Real.sqrt (d:ℝ))⁻¹
          * ∫ z, |(dir z).ofLp i| * |Real.sqrt (d:ℝ) - ‖z‖| ∂(gaussianE (Ed d)) :=
        integral_const_mul _ _
    _ ≤ (Real.sqrt (d:ℝ))⁻¹
          * (Real.sqrt (∫ z, ((dir z).ofLp i) ^ 2 ∂(gaussianE (Ed d)))
            * Real.sqrt (∫ z : Ed d, (Real.sqrt (d:ℝ) - ‖z‖) ^ 2 ∂(gaussianE (Ed d)))) := by
        refine mul_le_mul_of_nonneg_left ?_ (by positivity)
        exact integral_abs_mul_le_sqrt_mul_sqrt _ hMf hMg
    _ ≤ (Real.sqrt (d:ℝ))⁻¹
          * ((Real.sqrt (d:ℝ))⁻¹ * Real.sqrt (1 / 2 + 1 / (8 * ((d : ℝ) - 1)))) := by
        refine mul_le_mul_of_nonneg_left ?_ (by positivity)
        rw [hA, hsqrtA]
        exact mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hB) (by positivity)
    _ = Real.sqrt (1 / 2 + 1 / (8 * ((d : ℝ) - 1))) / (d : ℝ) := by
        rw [← mul_assoc, ← mul_inv, Real.mul_self_sqrt hd0.le, inv_mul_eq_div]

/-- **The spherical-versus-Gaussian leg with the halved constant.**  For every `d ≥ 2` and every
pair `a, x` of reals,

  `|cdfGapInt (sphericalCoordLaw d) (N(0, 1/d)) a x| ≤ √(1/2 + 1/(8(d−1)))/d`,

against `√(1 + 1/(4(d−1)))/d` in §57c.  Uniform in `a` and `x`, `Θ(1/d)` in the dimension, no
support hypothesis on either law.

What it is **not**.  It is still not a Kolmogorov-Smirnov bound: `|cdf μ − cdf ν|` is not bounded,
no `η` for §55 or §56 is produced, and it is one leg only, the other being §44's and §65's
business.  `abs_cdfGapInt_sphericalCoordLaw_gaussian_le` is untouched and remains proved. -/
theorem abs_cdfGapInt_sphericalCoordLaw_gaussian_le_half {d : ℕ} (hd : 2 ≤ d) (a x : ℝ) :
    |Threshold.cdfGapInt (sphericalCoordLaw d)
        (gaussianReal 0 (Real.toNNReal (1 / (d : ℝ)))) a x|
      ≤ Real.sqrt (1 / 2 + 1 / (8 * ((d : ℝ) - 1))) / (d : ℝ) := by
  haveI := isProbabilityMeasure_sphericalCoordLaw hd
  haveI : Nontrivial (Ed d) := Module.nontrivial_of_finrank_pos (R := ℝ) (by
    have : (0 : ℕ) < d := by omega
    simpa [Ed] using this)
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hrhs : (0:ℝ) ≤ Real.sqrt (1 / 2 + 1 / (8 * ((d : ℝ) - 1))) / (d : ℝ) := by positivity
  rcases le_or_gt a x with hax | hax
  · have hi : (0:ℕ) < d := by omega
    set i : Fin d := ⟨0, hi⟩ with hidef
    have hmX : Measurable (fun z : Ed d => (dir z).ofLp i) := measurable_dir_coord i
    have hcoord : Measurable (fun z : Ed d => z.ofLp i) :=
      (measurable_pi_apply i).comp (WithLp.measurable_ofLp 2 _)
    have hmY : Measurable (fun z : Ed d => z.ofLp i / Real.sqrt (d:ℝ)) := hcoord.div_const _
    have hXlaw : Measure.map (fun z : Ed d => (dir z).ofLp i) (gaussianE (Ed d))
        = sphericalCoordLaw d := gaussianE_map_dir_coord hd i
    have hYlaw : Measure.map (fun z : Ed d => z.ofLp i / Real.sqrt (d:ℝ)) (gaussianE (Ed d))
        = gaussianReal 0 (Real.toNNReal (1 / (d : ℝ))) := by
      have h1 : (fun z : Ed d => z.ofLp i / Real.sqrt (d:ℝ))
          = (fun t : ℝ => t / Real.sqrt (d:ℝ)) ∘ (fun z : Ed d => z.ofLp i) := rfl
      have hdiv : Measurable (fun t : ℝ => t / Real.sqrt (d:ℝ)) := measurable_id.div_const _
      rw [h1, ← Measure.map_map hdiv hcoord, gaussianE_map_ofLp i,
        gaussianReal_map_div_sqrt hd]
    have hIX : Integrable (fun z : Ed d => (dir z).ofLp i) (gaussianE (Ed d)) := by
      refine (integrable_const (1:ℝ)).mono' hmX.aestronglyMeasurable ?_
      filter_upwards with z
      rw [Real.norm_eq_abs]
      exact abs_dir_coord_le_one i z
    have hIY : Integrable (fun z : Ed d => z.ofLp i / Real.sqrt (d:ℝ)) (gaussianE (Ed d)) := by
      refine ((integrable_norm_gaussianE (E := Ed d)).div_const
        (Real.sqrt (d:ℝ))).mono' hmY.aestronglyMeasurable ?_
      filter_upwards with z
      rw [Real.norm_eq_abs, abs_div, abs_of_nonneg (Real.sqrt_nonneg _)]
      gcongr
      simpa using PiLp.norm_apply_le z i
    calc |Threshold.cdfGapInt (sphericalCoordLaw d)
            (gaussianReal 0 (Real.toNNReal (1 / (d : ℝ)))) a x|
        ≤ ∫ z, |(dir z).ofLp i - z.ofLp i / Real.sqrt (d:ℝ)| ∂(gaussianE (Ed d)) :=
          Threshold.abs_cdfGapInt_le_integral_abs_sub_of_coupling _ hmX.aemeasurable
            hmY.aemeasurable hIX hIY hXlaw hYlaw hax
      _ ≤ Real.sqrt (1 / 2 + 1 / (8 * ((d : ℝ) - 1))) / (d : ℝ) :=
          integral_abs_dir_coord_sub_coord_div_sqrt_le_half hd i
  · have hempty : Set.Ioc a x = (∅ : Set ℝ) := Set.Ioc_eq_empty (by linarith)
    rw [Threshold.cdfGapInt, hempty, Measure.restrict_empty, integral_zero_measure, abs_zero]
    exact hrhs

end CouplingLegHalf

/-! ### §67e  Rewiring: the composed bound with the halved leg

Each declaration below is a new one beside the §60, §65 and §66 statement it copies; none of
those is edited, weakened or removed.  The only change is the `+1` slot. -/

namespace NearFlat

variable {d : ℕ}

/-- The analogue of `sqrt_one_add_quarter_le_self`, and it runs the other way.  `√x ≤ x` fails
below `1`, so the rational cover of `√(1/2 + q)` is `5/7 + q`, valid because `(5/7)² = 25/49` is
above `1/2` and `2·(5/7) ≥ 1`.  `5/7 = 0.71429` is `1.0%` above `1/√2 = 0.70711`, which is the
true limit of the left side and the floor of this route; the `1.0%` is given away here and
nowhere else. -/
theorem sqrt_half_add_eighth_le (hd : 2 ≤ d) :
    Real.sqrt (1 / 2 + 1 / (8 * ((d : ℝ) - 1))) ≤ 5 / 7 + 1 / (8 * ((d : ℝ) - 1)) := by
  have hd1 : (1:ℝ) ≤ (d:ℝ) - 1 := by
    have : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
    linarith
  have hq : (0:ℝ) ≤ 1 / (8 * ((d:ℝ) - 1)) := by positivity
  have hle : 1 / 2 + 1 / (8 * ((d:ℝ) - 1)) ≤ (5 / 7 + 1 / (8 * ((d:ℝ) - 1))) ^ 2 := by nlinarith
  have := Real.sqrt_le_sqrt hle
  rwa [Real.sqrt_sq (by linarith)] at this

/-- **The composed leg at a near-flat direction, with the halved coupling constant.**  Given any
leg-1 bound of the shape `(A·∑ⱼvⱼ⁴ + B·(c/√d))/√d` against `N(0, 1/d)`, uniform in `x`,

  `|cdfGapInt (coordLaw H v i) (f_d) a x| ≤ (A·S·√d + B·c + 5/7 + 1/(8(d−1)))/d`,

against `(A·S·√d + B·c + 1 + 1/(4(d−1)))/d` in `abs_cdfGapInt_le_of_legBound`.  The near-flat
hypothesis is not used; this step is algebra plus `sqrt_half_add_eighth_le`.  It is not a
Kolmogorov-Smirnov bound and produces no `η`, and `abs_cdfGapInt_le_of_legBound` is untouched. -/
theorem abs_cdfGapInt_le_of_legBound_half (hd : 2 ≤ d) {H : Od d} {v : Ed d} (i : Fin d)
    {c A B a : ℝ}
    (hleg : ∀ x, |Threshold.cdfGapInt (coordLaw H v i)
        (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (A * (∑ j, v j ^ 4) + B * (c / Real.sqrt d)) / Real.sqrt d) (x : ℝ) :
    |Threshold.cdfGapInt (coordLaw H v i) (sphericalCoordLaw d) a x|
      ≤ (A * (∑ j, v j ^ 4) * Real.sqrt d + B * c + 5 / 7 + 1 / (8 * ((d : ℝ) - 1)))
          / (d : ℝ) := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hcomp := Threshold.abs_cdfGapInt_le_add_of_common (hleg x)
    (abs_cdfGapInt_sphericalCoordLaw_gaussian_le_half hd a x)
  have step2 : Real.sqrt (1 / 2 + 1 / (8 * ((d : ℝ) - 1))) / (d : ℝ)
      ≤ (5 / 7 + 1 / (8 * ((d:ℝ) - 1))) / (d:ℝ) := by
    gcongr
    exact sqrt_half_add_eighth_le (d := d) hd
  have heq : (A * (∑ j, v j ^ 4) + B * (c / Real.sqrt d)) / Real.sqrt d
        + (5 / 7 + 1 / (8 * ((d:ℝ) - 1))) / (d:ℝ)
      = (A * (∑ j, v j ^ 4) * Real.sqrt d + B * c + 5 / 7 + 1 / (8 * ((d : ℝ) - 1)))
          / (d : ℝ) := by
    have hs2 : Real.sqrt (d:ℝ) * Real.sqrt (d:ℝ) = (d:ℝ) := Real.mul_self_sqrt hd0.le
    set s := Real.sqrt (d:ℝ) with hs
    rw [← hs2]
    have hs0 : s ≠ 0 := ne_of_gt hsd
    have hs1 : s * s - 1 ≠ 0 := by
      rw [hs2]
      have : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
      intro h; linarith
    field_simp
    ring
  linarith

end NearFlat

namespace Threshold

open Hadamard Orthogonal CoordLaw NearFlat

variable {d : ℕ} {K : ℕ}

/-- **The deployed rate from any leg-1 bound, with the halved coupling constant.**  For `d ≥ 2`,
a flat `H`, a unit direction, an ordered codebook whose thresholds lie in a window `[a, b]` around
the support, and any measurable nearest-codeword map,

  `|hadRate H (blockStage q) v − ρ²| ≤ 2·(A·S·√d + B·c + 5/7 + 1/(8(d−1)))·spread cbk`

whenever leg 1 holds in the form `(A·S + B·(c/√d))/√d`, with `S = ∑ⱼvⱼ⁴`.  This is §56's
`abs_hadRate_sub_rhoSq_le_of_cdfGapInt` with `hC` discharged by
`NearFlat.abs_cdfGapInt_le_of_legBound_half`.  No `η`, no `C` and no `(b − a)` appears in the
conclusion, the near-flat hypothesis is not used, nothing is claimed sharp, and the deployed
rotation is not shown to attain `ρ²`.  `abs_hadRate_sub_rhoSq_le_of_legBound` is untouched. -/
theorem abs_hadRate_sub_rhoSq_le_of_legBound_half (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) {c A B : ℝ} (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hleg : ∀ x, |cdfGapInt (coordLaw H v i)
        (gaussianReal 0 (Real.toNNReal (1 / (d:ℝ)))) a x|
      ≤ (A * (∑ j, v j ^ 4) + B * (c / Real.sqrt d)) / Real.sqrt d) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ 2 * (A * (∑ j, v j ^ 4) * Real.sqrt d + B * c + 5 / 7 + 1 / (8 * ((d:ℝ) - 1)))
          * spread cbk := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  refine (abs_hadRate_sub_rhoSq_le_of_cdfGapInt hd hH v hv i cbk hq hqm ha hb hthr
    (fun x => abs_cdfGapInt_le_of_legBound_half hd i hleg x)).trans_eq ?_
  field_simp

/-- **The deployed rate at a near-flat direction, damping integrated and coupling halved.**  Same
hypotheses as `abs_hadRate_sub_rhoSq_le_of_nearFlat_mid`, whose `+1 + 1/(4(d−1))` is replaced:

  `|hadRate H (blockStage q) v − ρ²|`
    `≤ 2·((2/π)A_mid·c²/√d + (2/π)K(c/√d)·c + 5/7 + 1/(8(d−1)))·spread cbk`

with `A_mid = e^{−1/4} + √π − 3/4` of §66 and the tail constant `K` of §65, both **unchanged**.
At `d = 512` and `c = 3.53` the factor falls from `5.093` to `4.507`.

The caveats of `abs_hadRate_sub_rhoSq_le_of_nearFlat_mid` apply verbatim and none of them is
addressed here.  This is `Θ(spread cbk)` and not `o(1)`; `c` still bounds the **largest**
coordinate, so the near-flat class is still not of full measure on the sphere; `spread cbk` is
left abstract and no codebook is constructed.  `abs_hadRate_sub_rhoSq_le_of_nearFlat_mid` is
untouched and not weakened. -/
theorem abs_hadRate_sub_rhoSq_le_of_nearFlat_mid_half (hd : 2 ≤ d) {H : Od d} (hH : IsFlat H)
    (v : Ed d) (hv : ‖v‖ = 1) (i : Fin d) {c : ℝ} (hc0 : 0 < c) (hcd : c ≤ Real.sqrt d)
    (hnf : ∀ j, |v j| ≤ c / Real.sqrt d) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b) :
    |hadRate H (blockStage (m := d) (F := ℝ) q) v - rhoSq d q (sphericalCoordLaw d)|
      ≤ 2 * (2 / π * FourierGap.midConst * c ^ 2 / Real.sqrt d
          + 2 / π * FourierGap.tailConst (c / Real.sqrt d) * c + 5 / 7 + 1 / (8 * ((d:ℝ) - 1)))
          * spread cbk := by
  have hd0 : (0:ℝ) < (d:ℝ) := by exact_mod_cast (by omega : (0:ℕ) < d)
  have hsd : 0 < Real.sqrt d := Real.sqrt_pos.mpr hd0
  have hpi0 : (0:ℝ) < π := Real.pi_pos
  have hM0 : 0 < c / Real.sqrt d := div_pos hc0 hsd
  have hM1 : c / Real.sqrt d ≤ 1 := by rw [div_le_one hsd]; exact hcd
  have hA0 : (0:ℝ) ≤ 2 / π * FourierGap.midConst := by
    have := FourierGap.midConst_pos
    positivity
  have hmain := abs_hadRate_sub_rhoSq_le_of_legBound_half hd hH v hv i cbk hq hqm ha hb hthr
    (A := 2 / π * FourierGap.midConst)
    (B := 2 / π * FourierGap.tailConst (c / Real.sqrt d)) (c := c)
    (fun x => NearFlat.abs_cdfGapInt_coordLaw_gaussian_le_mid hd hH hv i hnf hM0 hM1 a x)
  refine hmain.trans ?_
  have hS := sum_pow_four_le_sq_div_of_nearFlat hd hv hnf
  have hval : (c ^ 2 / (d:ℝ)) * Real.sqrt d = c ^ 2 / Real.sqrt d := by
    have hs2 : Real.sqrt (d:ℝ) * Real.sqrt (d:ℝ) = (d:ℝ) := Real.mul_self_sqrt hd0.le
    set s := Real.sqrt (d:ℝ) with hs
    have hs0 : s ≠ 0 := ne_of_gt hsd
    rw [← hs2]
    field_simp
  have hkey : 2 / π * FourierGap.midConst * (∑ j, v j ^ 4) * Real.sqrt d
      ≤ 2 / π * FourierGap.midConst * c ^ 2 / Real.sqrt d := by
    have h1 : (∑ j, v j ^ 4) * Real.sqrt d ≤ c ^ 2 / Real.sqrt d := by
      rw [← hval]; exact mul_le_mul_of_nonneg_right hS hsd.le
    calc 2 / π * FourierGap.midConst * (∑ j, v j ^ 4) * Real.sqrt d
        = 2 / π * FourierGap.midConst * ((∑ j, v j ^ 4) * Real.sqrt d) := by ring
      _ ≤ 2 / π * FourierGap.midConst * (c ^ 2 / Real.sqrt d) :=
          mul_le_mul_of_nonneg_left h1 hA0
      _ = 2 / π * FourierGap.midConst * c ^ 2 / Real.sqrt d := by ring
  exact mul_le_mul_of_nonneg_right (by linarith) (spread_nonneg cbk)

/-- **The composed bound at the near-flat radius a random direction has, with the halved leg.**
At `d = 512`, for a flat `H`, a unit direction near-flat with `c ≤ 3.53`, any ordered codebook
whose thresholds lie in a window around the support and whose **spread is at most `3.0164/√512`**,
and any measurable nearest-codeword map,

  `|hadRate H (blockStage q) v − ρ²| ≤ 61/100`.

`1` is the bound that holds for free whenever `0` is a codeword; §66's
`abs_hadRate_sub_rhoSq_le_of_nearFlat_mid_512` reaches `69/100` at the same `c` and the same
spread, and §65's `abs_hadRate_sub_rhoSq_le_of_nearFlat_tail_512` reaches `24/25`.  The true value
of this bound is `0.601` and the chain of rational roundings gives `0.6053`; `61/100` is what
`√512 ≥ 22.627`, `√(2π) ≤ 2.5067`, `√π ≤ 1.77246`, `e^{−1/4} ≤ 250/321` and `√(1/2 + q) ≤ 5/7 + q`
support.

**The codebook width is a hypothesis, not a fact proved here.**  No codebook is constructed,
named or analysed, and nothing is claimed about any codebook the paper ships.  Nor is `1` the
target: `ρ² = 0.1171` is roughly five times smaller, and this route still cannot reach it, since
the coupling leg alone now contributes `2·spread cbk/√2 = 0.189` of the `0.601`.  The near-flat
hypothesis is a hypothesis about `v`, and the statement says nothing about directions that fail
it.  `abs_hadRate_sub_rhoSq_le_of_nearFlat_mid_512` is untouched and remains proved. -/
theorem abs_hadRate_sub_rhoSq_le_of_nearFlat_half_512 {H : Od 512} (hH : IsFlat H)
    (v : Ed 512) (hv : ‖v‖ = 1) (i : Fin 512) {c : ℝ} (hc0 : 0 < c) (hc : c ≤ 3.53)
    (hnf : ∀ j, |v j| ≤ c / Real.sqrt 512) (cbk : Codebook K) {q : ℝ → ℝ}
    (hq : IsNearest (Set.range cbk.c) q) (hqm : Measurable q) {a b : ℝ} (ha : a < -‖v‖)
    (hb : ‖v‖ ≤ b) (hthr : ∀ k : Fin K, thr cbk k ∈ Set.Icc a b)
    (hw : spread cbk ≤ 3.0164 / Real.sqrt 512) :
    |hadRate H (blockStage (m := 512) (F := ℝ) q) v
        - rhoSq 512 q (sphericalCoordLaw 512)| ≤ 61 / 100 := by
  have hs := FourierGap.sqrt_512_ge'
  have hsp : (0:ℝ) < Real.sqrt 512 := by linarith
  have hpi0 : (0:ℝ) < π := Real.pi_pos
  have hpi : (3.141592:ℝ) < π := Real.pi_gt_d6
  have hcast : Real.sqrt ((512:ℕ):ℝ) = Real.sqrt 512 := by norm_num
  have hnf' : ∀ j, |v j| ≤ c / Real.sqrt ((512:ℕ):ℝ) := by simpa [hcast] using hnf
  have hmain := abs_hadRate_sub_rhoSq_le_of_nearFlat_mid_half (d := 512) (by norm_num) hH v hv i
    hc0 (by rw [hcast]; linarith) hnf' cbk hq hqm ha hb hthr
  rw [hcast] at hmain
  push_cast at hmain
  have hsp0 : 0 ≤ spread cbk := spread_nonneg cbk
  have hM0 : 0 ≤ c / Real.sqrt 512 := by positivity
  have hMb : c / Real.sqrt 512 ≤ 1561 / 10000 := by
    rw [div_le_iff₀ hsp]
    nlinarith
  have hKnn : (0:ℝ) ≤ FourierGap.tailConst (c / Real.sqrt 512) := by
    rw [FourierGap.tailConst]; positivity
  have hK : FourierGap.tailConst (c / Real.sqrt 512) ≤ 409 / 1000 := by
    refine (FourierGap.tailConst_le_explicit hM0).trans ?_
    have h4 : (c / Real.sqrt 512) ^ 4 ≤ 1 / 1650 := by
      have h := pow_le_pow_left₀ hM0 hMb 4
      have hnum : ((1561:ℝ) / 10000) ^ 4 ≤ 1 / 1650 := by norm_num
      linarith
    linarith
  have hB : 2 / π * FourierGap.tailConst (c / Real.sqrt 512) ≤ 2604 / 10000 := by
    have h1 : 2 / π ≤ 2 / 3.141592 := by
      rw [div_le_div_iff₀ hpi0 (by norm_num)]; linarith
    calc 2 / π * FourierGap.tailConst (c / Real.sqrt 512)
        ≤ 2 / 3.141592 * FourierGap.tailConst (c / Real.sqrt 512) :=
          mul_le_mul_of_nonneg_right h1 hKnn
      _ ≤ 2 / 3.141592 * (409 / 1000) := mul_le_mul_of_nonneg_left hK (by norm_num)
      _ ≤ 2604 / 10000 := by norm_num
  have hAnn : (0:ℝ) ≤ FourierGap.midConst := le_of_lt FourierGap.midConst_pos
  have hA : 2 / π * FourierGap.midConst ≤ 11468 / 10000 := by
    have h1 : 2 / π ≤ 2 / 3.141592 := by
      rw [div_le_div_iff₀ hpi0 (by norm_num)]; linarith
    calc 2 / π * FourierGap.midConst
        ≤ 2 / 3.141592 * FourierGap.midConst := mul_le_mul_of_nonneg_right h1 hAnn
      _ ≤ 2 / 3.141592 * (18013 / 10000) :=
          mul_le_mul_of_nonneg_left FourierGap.midConst_le (by norm_num)
      _ ≤ 11468 / 10000 := by norm_num
  have hfac : 2 * (2 / π * FourierGap.midConst * c ^ 2 / Real.sqrt 512
      + 2 / π * FourierGap.tailConst (c / Real.sqrt 512) * c + 5 / 7 + 1 / (8 * ((512:ℝ) - 1)))
      ≤ 454 / 100 := by
    have hc2 : c ^ 2 ≤ 3.53 ^ 2 := by nlinarith
    have hA0 : (0:ℝ) ≤ 2 / π * FourierGap.midConst := by positivity
    have h1 : 2 / π * FourierGap.midConst * c ^ 2 / Real.sqrt 512
        ≤ 11468 / 10000 * (3.53 ^ 2) / 22.627 := by
      rw [div_le_div_iff₀ hsp (by norm_num)]
      nlinarith [hs, hc2, hA0, hA]
    have hBnn : (0:ℝ) ≤ 2 / π * FourierGap.tailConst (c / Real.sqrt 512) := by positivity
    have h2 : 2 / π * FourierGap.tailConst (c / Real.sqrt 512) * c ≤ 2604 / 10000 * 3.53 := by
      nlinarith
    norm_num at h1 h2 ⊢
    linarith
  have hstep : |hadRate H (blockStage (m := 512) (F := ℝ) q) v
      - rhoSq 512 q (sphericalCoordLaw 512)| ≤ 454 / 100 * (3.0164 / Real.sqrt 512) := by
    refine hmain.trans ((mul_le_mul_of_nonneg_right hfac hsp0).trans ?_)
    exact mul_le_mul_of_nonneg_left hw (by norm_num)
  refine hstep.trans ?_
  have heq : (454:ℝ) / 100 * (3.0164 / Real.sqrt 512)
      = (454 * 3.0164) / (100 * Real.sqrt 512) := by ring
  rw [heq, div_le_iff₀ (by positivity)]
  nlinarith [hs]

end Threshold

end CouplingHalf
