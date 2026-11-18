
/-! ## §48e  The coordinate law at a basis direction is the two-point law, via `radLaw`

§48b identifies the law of every coordinate of the deployed rotation `H D_ε v` with `radLaw d v`,
the law of the Rademacher sum `∑ₖ vₖ εₖ / √d` (`coordLaw_eq_radLaw`).  At a basis direction
`v = eⱼ` that sum has only one nonzero term, since every other coordinate of `eⱼ` vanishes, so it
collapses to `εⱼ / √d`: the deployed coordinate is `+1/√d` or `-1/√d`, each with probability
`1/2`, and nothing else.  This is exactly the two-point law that the development's header claims
a uniform bound on `hadRate` cannot be reconciled with: the spherical marginal `f_d` spreads
around `±1/√d` rather than sitting on it.  Nothing here proves that inequality; this section only
identifies the coordinate law itself, which is what the inequality is about.

Every law in this development is recorded by its integral against an arbitrary measurable test
function, exactly as `integral_radLaw` and `integral_coordLaw` already are (§48b, §41); that is
the idiom followed here, since it is the form every downstream user of a coordinate law actually
consumes (`distortion`, `cdfGapInt`, moments), and it is what pins a finitely-supported measure
down completely.  `integral_radLaw_single` is that identity at `v = eⱼ`; it needs no hypothesis
on `H`, since `radLaw` does not mention `H` at all.  `integral_coordLaw_single` transports it
through `coordLaw_eq_radLaw`, so it needs flatness of `H`; that is the only extra hypothesis
anywhere in this section, and no bound on `d` beyond the coordinate `j : Fin d` existing is used.

This is the same target as `CoordLaw.coordLaw_basis_eq_twoPoint`, proved independently by a
different route: that one pushes forward `uniformSigns d` by hand, against an explicit
`Measure.dirac` combination, and never mentions `radLaw`; this one goes through the `radLaw`
pushforward `coordLaw_eq_radLaw` already on record and states the result as an integral identity
rather than as an equation between measures.  Neither weakens or duplicates the other; both are
kept. -/

namespace CoordLaw

open Hadamard Orthogonal

variable {d : ℕ}

/-- **A fair bit averages its two values.**  For any `h : Bool → ℝ` and any coordinate `j`, the
sign average of `h` composed with reading bit `j` off the sign pattern is the mean of `h true`
and `h false`.  Proved by the same bit-flip symmetry as `signAvg_radSum_congr_abs` (§48):
flipping bit `j` alone is a bijection of the sign cube that swaps `s j` with `!(s j)`, so the
sign average is unchanged under `h ↦ h ∘ (!·)`, and adding the two equal averages collapses the
sum without ever counting how many sign patterns have `s j = true`. -/
theorem signAvg_eval_eq_avg_two (j : Fin d) (h : Bool → ℝ) :
    signAvg (fun s : Signs d => h (s j)) = (h true + h false) / 2 := by
  set τ : Fin d → Bool := fun k => decide (k = j) with hτ
  have hflipj : ∀ s : Signs d, (flipEquiv τ s) j = !(s j) := by
    intro s
    show xor (τ j) (s j) = !(s j)
    simp [hτ]
  have hsum : (∑ s : Signs d, h (s j)) = ∑ s : Signs d, h (!(s j)) := by
    rw [← Equiv.sum_comp (flipEquiv τ) (fun s : Signs d => h (s j))]
    exact Finset.sum_congr rfl fun s _ => by rw [hflipj s]
  have hadd : ∀ s : Signs d, h (s j) + h (!(s j)) = h true + h false := by
    intro s; cases hs : s j <;> simp [add_comm]
  have hsum2 : (∑ s : Signs d, h (s j)) + (∑ s : Signs d, h (s j))
      = ∑ _s : Signs d, (h true + h false) := by
    calc (∑ s : Signs d, h (s j)) + (∑ s : Signs d, h (s j))
        = (∑ s : Signs d, h (s j)) + (∑ s : Signs d, h (!(s j))) := by rw [← hsum]
      _ = ∑ s : Signs d, (h (s j) + h (!(s j))) := (Finset.sum_add_distrib).symm
      _ = ∑ _s : Signs d, (h true + h false) := Finset.sum_congr rfl fun s _ => hadd s
  rw [Finset.sum_const, Finset.card_univ, card_signs, nsmul_eq_mul] at hsum2
  push_cast at hsum2
  unfold signAvg
  field_simp
  linarith [hsum2]

/-- **Collapsing the Rademacher sum at a basis vector.**  At `v = eⱼ`, `radSum (v/√d) s` has only
one nonzero term, since every other coordinate of `eⱼ` vanishes. -/
theorem radSum_single (j : Fin d) (s : Signs d) :
    radSum (fun k => (EuclideanSpace.single j (1:ℝ) : Ed d) k / Real.sqrt d) s
      = sgn (s j) / Real.sqrt d := by
  unfold radSum
  rw [Finset.sum_eq_single j (fun k _ hk => by simp [hk])
    (fun h => absurd (Finset.mem_univ j) h)]
  simp; ring

/-- **`radLaw d eⱼ` is the two-point law on `±1/√d`.**  For every measurable test function `g`,
its integral against `radLaw d eⱼ` is the average of `g` at `1/√d` and at `-1/√d`: by
`radSum_single` the Rademacher sum collapses to `sgn(εⱼ)/√d`, a fair coin scaled by `1/√d`, and
`signAvg_eval_eq_avg_two` averages it.  No hypothesis on `H` or on `d` is needed, since `radLaw`
mentions neither. -/
theorem integral_radLaw_single (j : Fin d) {g : ℝ → ℝ} (hg : Measurable g) :
    ∫ x, g x ∂(radLaw d (EuclideanSpace.single j (1:ℝ)))
      = (g (1 / Real.sqrt d) + g (-(1 / Real.sqrt d))) / 2 := by
  rw [integral_radLaw d _ hg]
  simp_rw [radSum_single]
  have hkey := signAvg_eval_eq_avg_two j (fun b => g (sgn b / Real.sqrt d))
  rw [sgn_true, sgn_false] at hkey
  rwa [show (-1:ℝ) / Real.sqrt d = -(1 / Real.sqrt d) by ring] at hkey

/-- **The deployed coordinate law at a basis direction is the two-point law on `±1/√d`.**  For a
flat `H`, every coordinate `i` of `H D_ε eⱼ` has, against every measurable `g`, expectation the
average of `g` at `1/√d` and at `-1/√d`: the coordinate itself takes only those two values, each
with probability `1/2`.  This is `integral_radLaw_single` transported through
`coordLaw_eq_radLaw` (§48b); it needs flatness of `H` and nothing else, and unlike
`integral_radLaw_single` it mentions `H` and `i`, but the right-hand side still does not.

**What this does and does not say.**  It identifies the law exactly, at one direction (`eⱼ`) and
for the deployed rotation only.  It says nothing about any other direction and nothing about a
codebook or a rate; the development's header records why no bound uniform over every unit
direction can be exact here too, and no such claim is made in this theorem. -/
theorem integral_coordLaw_single {H : Od d} (hH : IsFlat H) (i j : Fin d) {g : ℝ → ℝ}
    (hg : Measurable g) :
    ∫ x, g x ∂(coordLaw H (EuclideanSpace.single j (1:ℝ)) i)
      = (g (1 / Real.sqrt d) + g (-(1 / Real.sqrt d))) / 2 := by
  rw [coordLaw_eq_radLaw hH (EuclideanSpace.single j (1:ℝ)) i]
  exact integral_radLaw_single j hg

end CoordLaw
