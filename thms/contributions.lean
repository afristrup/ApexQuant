/-
# Autoformalisation of the theory of *Progressive Density-Matched Quantisation*

Source: `texts/pagers/paper/sections/{background,theory}.tex`
(ApexQuant Team, 3 July 2026).  Lean 4.33.0-rc1 + Mathlib v4.33.0-rc1.

Build: `lake build` in `thms_compressed/`.  Every theorem below is either proved outright or
carries its assumptions in its signature; **nothing is asserted by `axiom`**, and `#print axioms`
on every public result returns only `propext, Classical.choice, Quot.sound`.

This development accompanies the paper.  It contains every result referenced in the paper's
theory section and proofs appendix.

---
## What is proved

| paper                                        | here                                        |
| -------------------------------------------- | ------------------------------------------- |
| eq. (2.2), the pass recursion                | `resid`, `unitVec`, `code`                  |
| `‖r_{p+1}‖ = ‖r_p‖·‖u_p − q(u_p)‖`           | `norm_resid_succ` (§1) — *identity, no hypotheses* |
| `w − ŵ_P = r_P`                              | `sub_recon` (§1)                            |
| Prop. 2, one pass                            | `integral_norm_resid_sq_succ` (§3)          |
| Prop. 2, `E‖r_P‖² = ρ^{2P}·E‖w‖²`            | `integral_norm_resid_sq` (§3)               |
| Prop. 2, eq. (2.5)                           | `rms_relative_error` (§3)                   |
| `ρ² = d·E[(t−q t)²]`, exchangeable coords    | `integral_norm_sub_cb_sq`, `rhoSq_eq_ratio` (§3b) |
| `E_{f_d}[t²] = 1/d`                          | `second_moment_eq_inv_card` (§3b)           |
| Lemma 1, `f_d` is a density, moments `0`,`1/d` | `sphericalPdf_*`, `sphericalCoordLaw` (§4, §4b) |
| Thm. 1, the right-invariance argument        | `orbitLaw_smul`, `law_smul_eq_orbitLaw` (§5) |
| Thm. 1, the independence `u ⟂ scale`         | `indepFun_smul_of_indepPair` (§5)           |
| `O(d)` is transitive on `S^{d−1}`            | `exists_isometry_map_eq`, `orbit_eq_sphere` (§5b) |
| Cor. 1, and the envelope `ρ^P`               | `optimal_codebook_optimal_at_every_pass` (§6) |
| design rule `P = ⌈ln ε / ln ρ⌉`              | `rho_pow_ceil_le`, `le_ceil_of_rho_pow_le` (§6) |
| Cor. 1's scope is *scalar* quantisers        | `sInf_stageDistortion_scalarStages_le` (§9) |
| the decay law is quantiser-agnostic          | `integral_norm_resid_sq_vectorStage` (§9)   |
| `r_p` depends only on `(w, R_0,…,R_{p−1})`   | `resid_rotScheme_eq_residPast` (§11)        |
| the fresh `R_p` is independent of the past   | `indepFun_rot_past_of_iIndepFun` (§11)      |
| `u_p` has the *same* law at every pass       | `law_unitVec_eq_orbitLaw` (§11)             |
| Prop. 2's hypotheses, *derived*              | `isotropicPasses_of_rotationStream` (§11)   |
| Prop. 2 from stream hypotheses alone         | `integral_norm_resid_sq_rotationStream` (§11) |
| Cor. 1 for *every* codebook class            | `stageDistortion_le_of_codebook_optimal` (§16) |
| a nearest-codeword stage exists, `C` finite  | `exists_isNearest` (§16)                    |
| `ρ²` of a block (E8, trellis) stage, eq. (2.3) | `stageDistortion_blockStage`, `stageDistortion_blockStage_eq_ratio` (§17) |
| the scalar quadrature is the `F = ℝ` case    | `stageDistortion_blockStage_real`, `scalarStages_eq_blockStage` (§17) |
| per-block nearest is nearest in the product  | `isNearest_blockStage` (§17)                |
| blocks are identically distributed           | `blockMarginal_eq_of_swap` (§17b)           |
| Cor. 1 at the residual level, every pass     | `integral_norm_resid_sq_le_of_codebook_optimal` (§18) |
| Prop. 2 for a block stage, `ρ²` by quadrature | `integral_norm_resid_sq_blockStage` (§18)   |
| the polar formula, and its 2-D quadrant form | `lintegral_eq_polar`, `lintegral_quadFun` (§19) |
| Lemma 1 for a block of `B` coordinates       | `toSphere_map_fst`, `sphereUniform_map_block` (§20, §21) |
| Lemma 1, the *distributional* statement      | `sphereUniform_map_inner`, `sphereUniformAmbient_map_inner` (§21) |
| the sphere measure is isometry-invariant     | `toSphere_map_sphereMap` (§21)              |
| Prop. 1, the identification with the uniform law | `orbitLaw_eq_sphereUniformAmbient` (§22) |
| `u_p` uniform on the sphere at every pass    | `law_unitVec_eq_sphereUniform` (§22)        |
| Haar on a compact group is right-invariant   | `isMulRightInvariant_of_compactSpace` (§23) |
| `O(d)` compact, Haar probability, transitive | `Orthogonal.haarOd`, `Orthogonal.transitive` (§23) |
| Prop. 1 at `O(d)`, in full                   | `Orthogonal.orbitLaw_haar` (§23)            |
| Prop. 2 exactly as the paper states it       | `integral_norm_resid_sq_paper` (§24)        |
| the hypotheses hold at `O(d)` for every `d`  | `exists_rotationStream_Od`, `exists_witness_decay_Od` (§24) |
| **the uniform law is the only rotation-invariant probability measure on the sphere** | `eq_sphereUniformAmbient_of_invariant` (§45) |
| **Lemma 1, block form, with its constant** | `blockLaw_eq_withDensity` (§46) |

## Scope

The development covers the core theory (re-isotropisation, geometric decay, optimality,
block marginals, the bridge from abstract hypotheses to $O(d)$) and two auxiliary results
(uniqueness of the rotation-invariant measure and the block-form constant).  Results not
referenced in the paper — including the per-pass concentration bound, the deployed-rotation
formalisation, and the detailed Haar-versus-Hadamard gap analysis — are not included.
-/

import Mathlib

open MeasureTheory ProbabilityTheory Finset
open scoped ENNReal NNReal Real

namespace Progressive

/-! ## §1  The pass recursion (eq. 2.2), deterministically

The paper's eq. (2.2) is

  `s_p = ‖r_p‖,  u_p = R_p r_p / s_p,  r_{p+1} = r_p - s_p R_pᵀ q(u_p)`,  `r_0 = w`.

Everything in this section is exact linear algebra in a real inner product space: no
probability, no coordinates, no assumption on `q` whatsoever. -/
section Deterministic

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The data of a progressive (multistage) quantiser: a stream of rotations `R_p`, and a
**single fixed** quantiser `q` reused at every pass.  That `q` does not depend on `p` is the
paper's central design claim; it is expressed here by the type. -/
structure Scheme (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E] where
  /-- `rot p` is the rotation `R_p` applied at pass `p`. -/
  rot : ℕ → (E ≃ₗᵢ[ℝ] E)
  /-- the codebook rounding map; in the paper, nearest-neighbour rounding applied
  coordinate-wise to a `2^b`-level Lloyd–Max codebook for `f_d`. -/
  q : E → E

variable (S : Scheme E) (w : E)

/-- `resid S w p` is `r_p` of eq. (2.2). -/
noncomputable def resid : ℕ → E
  | 0 => w
  | p + 1 =>
      resid p - ‖resid p‖ • (S.rot p).symm (S.q (‖resid p‖⁻¹ • S.rot p (resid p)))

/-- `unitVec S w p` is `u_p = R_p r_p / ‖r_p‖` of eq. (2.2).

At `r_p = 0` this evaluates to `0` by Lean's `(0 : ℝ)⁻¹ = 0` convention; every statement below
that mentions `u_p` is either guarded by `r_p ≠ 0` or is trivially true at `r_p = 0`. -/
noncomputable def unitVec (p : ℕ) : E := ‖resid S w p‖⁻¹ • S.rot p (resid S w p)

/-- The *code* emitted at pass `p`: the scale `s_p` (stored as one fp16 number per group) and
the codeword `q(u_p)` (stored as `d` `b`-bit indices).  This pair is the entire artifact of
pass `p`. -/
noncomputable def code (p : ℕ) : ℝ × E := (‖resid S w p‖, S.q (unitVec S w p))

@[simp] theorem resid_zero : resid S w 0 = w := rfl

theorem resid_succ (p : ℕ) :
    resid S w (p + 1) = resid S w p - ‖resid S w p‖ • (S.rot p).symm (S.q (unitVec S w p)) :=
  rfl

/-- `‖u_p‖ = 1` whenever the pass is not already exact. -/
theorem norm_unitVec (p : ℕ) (h : resid S w p ≠ 0) : ‖unitVec S w p‖ = 1 := by
  rw [unitVec, norm_smul, LinearIsometryEquiv.norm_map, norm_inv, norm_norm,
    inv_mul_cancel₀ (norm_ne_zero_iff.mpr h)]

/-- **The per-pass identity, exactly.**  The residual of pass `p+1` is the pass-`p` scale times
the pulled-back quantisation error of the unit direction.  This is the step the paper writes as
"because `R_p` is orthogonal, `‖r_{p+1}‖² = s_p² ‖u_p - q(u_p)‖²`", and it holds *identically* —
no probability, no hypothesis on `q`, and including the degenerate case `r_p = 0`. -/
theorem resid_succ_eq_smul (p : ℕ) :
    resid S w (p + 1)
      = ‖resid S w p‖ • (S.rot p).symm (unitVec S w p - S.q (unitVec S w p)) := by
  rw [resid_succ, map_sub, smul_sub]
  congr 1
  rw [unitVec, map_smul, LinearIsometryEquiv.symm_apply_apply, smul_smul]
  rcases eq_or_ne (resid S w p) 0 with h | h
  · simp [h]
  · rw [mul_inv_cancel₀ (norm_ne_zero_iff.mpr h), one_smul]

/-- The norm form of `resid_succ_eq_smul`: `‖r_{p+1}‖ = ‖r_p‖ · ‖u_p - q(u_p)‖`. -/
theorem norm_resid_succ (p : ℕ) :
    ‖resid S w (p + 1)‖ = ‖resid S w p‖ * ‖unitVec S w p - S.q (unitVec S w p)‖ := by
  rw [resid_succ_eq_smul, norm_smul, LinearIsometryEquiv.norm_map, norm_norm]

/-- The squared form, which is what Proposition 2 iterates. -/
theorem norm_resid_succ_sq (p : ℕ) :
    ‖resid S w (p + 1)‖ ^ 2
      = ‖resid S w p‖ ^ 2 * ‖unitVec S w p - S.q (unitVec S w p)‖ ^ 2 := by
  rw [norm_resid_succ, mul_pow]

/-- The residual norms are non-increasing along passes **iff** each pass does not increase the
normalised error; the paper's `ρ < 1` regime.  Stated as a hypothesis-carrying lemma rather
than assumed. -/
theorem norm_resid_le_of_contract {c : ℝ} (hc : 0 ≤ c)
    (h : ∀ p, ‖unitVec S w p - S.q (unitVec S w p)‖ ≤ c) (P : ℕ) :
    ‖resid S w P‖ ≤ c ^ P * ‖w‖ := by
  induction P with
  | zero => simp
  | succ P ih =>
      rw [norm_resid_succ]
      calc ‖resid S w P‖ * ‖unitVec S w P - S.q (unitVec S w P)‖
          ≤ (c ^ P * ‖w‖) * c :=
            mul_le_mul ih (h P) (norm_nonneg _) (mul_nonneg (pow_nonneg hc P) (norm_nonneg _))
        _ = c ^ (P + 1) * ‖w‖ := by ring

/-! ### Reconstruction and the telescoping identity -/

/-- `recon S w P = ŵ_P = ∑_{p<P} s_p R_pᵀ q(u_p)`, the `P`-pass reconstruction. -/
noncomputable def recon (P : ℕ) : E :=
  ∑ p ∈ range P, ‖resid S w p‖ • (S.rot p).symm (S.q (unitVec S w p))

@[simp] theorem recon_zero : recon S w 0 = 0 := by simp [recon]

/-- Auxiliary: the telescoping identity `w - ŵ_P = r_P`, from which `sub_recon` (§2) follows. -/
private theorem aux_sub_recon (P : ℕ) : w - recon S w P = resid S w P := by
  induction P with
  | zero => simp
  | succ P ih =>
      have hr : recon S w (P + 1)
          = recon S w P + ‖resid S w P‖ • (S.rot P).symm (S.q (unitVec S w P)) := by
        simp [recon, Finset.sum_range_succ]
      rw [hr, ← sub_sub, ih, ← resid_succ]
/-! ## §3  Proposition 2 (geometric decay)

The paper's Proposition 2 asserts `E‖r_{p+1}‖² = ρ² E‖r_p‖²`, hence `√(E‖r_P‖²)/‖w‖ = ρ^P`.

Its proof uses exactly two probabilistic inputs, isolated here as `IsotropicPasses`:

* `indep p`: the normalised direction's squared quantisation error is independent of the
  squared scale.  Theorem 1 (§5) supplies this — `u_p ⟂ r_p` gives `u_p ⟂ s_p`.
* `distortion p`: every pass sees the *same* expected normalised distortion `ρ²`.
  Corollary 1 (§6) plus Lemma 1 (§4) supply this — the coordinate marginals are `f_d` at
  every pass, so the expectation is the same number `ρ²` at every pass.

Note what is *absent*: the closed form of `f_d`, the optimality of the codebook, and any
concentration statement.  Proposition 2 is an identity in expectation and nothing more. -/

section Decay

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- The two hypotheses Proposition 2 actually consumes, for a random weight row `W`. -/
structure IsotropicPasses (S : Ω → Scheme E) (W : Ω → E) (ρ : ℝ) (μ : Measure Ω) : Prop where
  /-- integrability of the squared residual norms (implicit in the paper). -/
  integrable : ∀ p, Integrable (fun ω => ‖resid (S ω) (W ω) p‖ ^ 2) μ
  /-- the pass-`p` squared normalised distortion is integrable. -/
  integrable_err : ∀ p,
    Integrable (fun ω => ‖unitVec (S ω) (W ω) p - (S ω).q (unitVec (S ω) (W ω) p)‖ ^ 2) μ
  /-- **direction ⟂ scale** — supplied by Theorem 1. -/
  indep : ∀ p, IndepFun (fun ω => ‖resid (S ω) (W ω) p‖ ^ 2)
    (fun ω => ‖unitVec (S ω) (W ω) p - (S ω).q (unitVec (S ω) (W ω) p)‖ ^ 2) μ
  /-- **every pass has the same normalised distortion `ρ²`** — supplied by Theorem 1 +
  Lemma 1: the coordinate marginals are `f_d` at every pass, so this expectation does not
  depend on `p`. -/
  distortion : ∀ p,
    ∫ ω, ‖unitVec (S ω) (W ω) p - (S ω).q (unitVec (S ω) (W ω) p)‖ ^ 2 ∂μ = ρ ^ 2

variable {S : Ω → Scheme E} {W : Ω → E} {ρ : ℝ}

/-- **Proposition 2, one pass.**  `E‖r_{p+1}‖² = ρ² · E‖r_p‖²`. -/
theorem integral_norm_resid_sq_succ [IsProbabilityMeasure μ]
    (h : IsotropicPasses S W ρ μ) (p : ℕ) :
    ∫ ω, ‖resid (S ω) (W ω) (p + 1)‖ ^ 2 ∂μ = ρ ^ 2 * ∫ ω, ‖resid (S ω) (W ω) p‖ ^ 2 ∂μ := by
  -- The deterministic per-pass identity `‖r_{p+1}‖² = ‖r_p‖² · ‖u_p - q u_p‖²` (§1),
  -- then independence to split the expectation of the product, then `distortion p`.
  have hpt : ∀ ω, ‖resid (S ω) (W ω) (p + 1)‖ ^ 2
      = ‖resid (S ω) (W ω) p‖ ^ 2 * ‖unitVec (S ω) (W ω) p - (S ω).q (unitVec (S ω) (W ω) p)‖ ^ 2 :=
    fun ω => norm_resid_succ_sq (S ω) (W ω) p
  simp_rw [hpt]
  rw [(h.indep p).integral_fun_mul_eq_mul_integral
      (h.integrable p).aestronglyMeasurable (h.integrable_err p).aestronglyMeasurable,
    h.distortion p, mul_comm]

/-- **Proposition 2.**  `E‖r_P‖² = ρ^{2P} · E‖w‖²` for every `P`. -/
theorem integral_norm_resid_sq [IsProbabilityMeasure μ]
    (h : IsotropicPasses S W ρ μ) (P : ℕ) :
    ∫ ω, ‖resid (S ω) (W ω) P‖ ^ 2 ∂μ = ρ ^ (2 * P) * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  induction P with
  | zero => simp
  | succ P ih =>
      rw [integral_norm_resid_sq_succ h P, ih, Nat.mul_succ, pow_add]
      ring

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

/-! ### §3b  Where `ρ²` comes from: the coordinate layer

`distortion` above is an assumption about vectors.  Here it is discharged from the paper's
actual hypothesis — that the `d` coordinates of `u_p` share a common marginal law `ν` — by
linearity of expectation.  The paper is careful on this point and the formalisation confirms
it: *the coordinates need not be independent, only identically distributed*. -/

section Coordinates

variable {d : ℕ} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- `ρ²` of eq. (2.3): `d · E_{t∼ν}[(t - cb t)²]`. -/
noncomputable def rhoSq (d : ℕ) (cb : ℝ → ℝ) (ν : Measure ℝ) : ℝ :=
  (d : ℝ) * ∫ t, (t - cb t) ^ 2 ∂ν

/-- The second equality of eq. (2.3): `d · E[(t-q t)²] = E[(t-q t)²] / E[t²]` once `E[t²] = 1/d`.
The paper derives `E_{f_d}[t²] = 1/d` from `∑ᵢ E[u_i²] = 1`, i.e. from `‖u‖ = 1` and
exchangeability — not from the closed form of `f_d`. -/
theorem rhoSq_eq_ratio {cb : ℝ → ℝ} {ν : Measure ℝ} (hd : d ≠ 0)
    (hν : ∫ t, t ^ 2 ∂ν = 1 / (d : ℝ)) :
    rhoSq d cb ν = (∫ t, (t - cb t) ^ 2 ∂ν) / (∫ t, t ^ 2 ∂ν) := by
  have hd' : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hd
  rw [rhoSq, hν]
  field_simp

/-- **Linearity of expectation over exchangeable coordinates.**  If the `d` coordinates of the
random unit vector `U` each have law `ν`, then `E‖U - q U‖² = d · E_ν[(t - cb t)²] = ρ²`,
where `q` is `cb` applied coordinate-wise.  No independence between coordinates is used. -/
theorem integral_norm_sub_cb_sq
    {U : Ω → EuclideanSpace ℝ (Fin d)} {cb : ℝ → ℝ} {ν : Measure ℝ}
    (hmeas : ∀ i, Measurable fun ω => U ω i)
    (hmarg : ∀ i, μ.map (fun ω => U ω i) = ν)
    (hint : Integrable (fun t => (t - cb t) ^ 2) ν) :
    ∫ ω, ‖U ω - WithLp.toLp 2 (fun i => cb (U ω i))‖ ^ 2 ∂μ = rhoSq d cb ν := by
  -- Each coordinate map pushes `μ` forward to `ν`, so `(t - cb t)^2` is `ν`-a.e. strongly
  -- measurable when read through any one of them.
  have hasm : ∀ i, AEStronglyMeasurable (fun t => (t - cb t) ^ 2)
      (μ.map (fun ω => U ω i)) := by
    intro i; rw [hmarg i]; exact hint.aestronglyMeasurable
  -- Integrability transported back along each coordinate.  This is the *only* place a
  -- side condition beyond the marginal law is needed, and it is per-coordinate: nothing
  -- about the joint law of the `d` coordinates is used.
  have hmapint : ∀ i, Integrable (fun ω => (U ω i - cb (U ω i)) ^ 2) μ := fun i =>
    (integrable_map_measure (hasm i) (hmeas i).aemeasurable).mp (by rw [hmarg i]; exact hint)
  -- Change of variables, one coordinate at a time.
  have hcoord : ∀ i : Fin d, ∫ ω, (U ω i - cb (U ω i)) ^ 2 ∂μ = ∫ t, (t - cb t) ^ 2 ∂ν := by
    intro i; rw [← hmarg i, integral_map (hmeas i).aemeasurable (hasm i)]
  -- `‖·‖²` on `EuclideanSpace` is the coordinate sum; `q` acts coordinate-wise.
  have hexp : ∀ ω : Ω, ‖U ω - WithLp.toLp 2 (fun i => cb (U ω i))‖ ^ 2
      = ∑ i, (U ω i - cb (U ω i)) ^ 2 := by
    intro ω; rw [EuclideanSpace.real_norm_sq_eq]; simp
  -- Linearity of expectation: swap `∫` and the finite `∑`.  No independence enters.
  simp_rw [hexp]
  rw [integral_finsetSum _ (fun i _ => hmapint i)]
  simp_rw [hcoord]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, rhoSq]

/-- `E_ν[t²] = 1/d` is *forced*: it is `‖U‖ = 1` plus the coordinates sharing a law, not an
extra assumption about `f_d`.  (This is the step the paper's proof records parenthetically.) -/
theorem second_moment_eq_inv_card
    {U : Ω → EuclideanSpace ℝ (Fin d)} {ν : Measure ℝ} [IsProbabilityMeasure μ] (hd : d ≠ 0)
    (hmeas : ∀ i, Measurable fun ω => U ω i)
    (hmarg : ∀ i, μ.map (fun ω => U ω i) = ν)
    (hint : Integrable (fun t => t ^ 2) ν)
    (hunit : ∀ ω, ‖U ω‖ = 1) :
    ∫ t, t ^ 2 ∂ν = 1 / (d : ℝ) := by
  have hd' : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hd
  have hasm : ∀ i, AEStronglyMeasurable (fun t : ℝ => t ^ 2) (μ.map (fun ω => U ω i)) := by
    intro i; rw [hmarg i]; exact hint.aestronglyMeasurable
  have hmapint : ∀ i, Integrable (fun ω => (U ω i) ^ 2) μ := fun i =>
    (integrable_map_measure (hasm i) (hmeas i).aemeasurable).mp (by rw [hmarg i]; exact hint)
  have hcoord : ∀ i : Fin d, ∫ ω, (U ω i) ^ 2 ∂μ = ∫ t, t ^ 2 ∂ν := by
    intro i; rw [← hmarg i, integral_map (hmeas i).aemeasurable (hasm i)]
  -- `‖U ω‖ = 1` pointwise, i.e. the coordinate squares sum to `1` on the nose.
  have hone : ∀ ω : Ω, ∑ i, (U ω i) ^ 2 = 1 := by
    intro ω
    rw [← EuclideanSpace.real_norm_sq_eq, hunit ω, one_pow]
  -- Integrate: `d · E_ν[t²] = E[‖U‖²] = 1`.  Again only linearity of expectation.
  have key : (d : ℝ) * ∫ t, t ^ 2 ∂ν = 1 := by
    have h1 : ∫ ω, ∑ i, (U ω i) ^ 2 ∂μ = 1 := by
      simp_rw [hone]; simp
    rw [integral_finsetSum _ (fun i _ => hmapint i)] at h1
    simp_rw [hcoord] at h1
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h1
    exact h1
  field_simp at key ⊢
  linarith [key]

end Coordinates

/-! ## §4  Lemma 1 (spherical marginal) and the data-free evaluation of `ρ`

Lemma 1 states that a coordinate of a uniform point on `S^{d-1}` has density

  `f_d(t) = Γ(d/2) / (√π Γ((d-1)/2)) · (1 - t²)^((d-3)/2)`  on `[-1,1]`,

equivalently `(u_i + 1)/2 ∼ Beta((d-1)/2, (d-1)/2)`, with mean `0` and variance `1/d`.

What §3 needs from this is only `∫ t² f_d = 1/d`, which §3b already derives for free.  What
Lemma 1 is genuinely *for* is the paper's data-free design chart: it makes `ρ` a
one-dimensional quadrature that can be evaluated before any weight is touched.  The statements
below certify that `f_d` really is a probability density with the advertised moments. -/

section Marginal

variable {d : ℕ}

/-- The claimed marginal density `f_d`. -/
noncomputable def sphericalPdf (d : ℕ) (t : ℝ) : ℝ :=
  Real.Gamma ((d : ℝ) / 2) / (Real.sqrt π * Real.Gamma (((d : ℝ) - 1) / 2))
    * (1 - t ^ 2) ^ (((d : ℝ) - 3) / 2)

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

private theorem aux_ii_right (a : ℝ) (ha : -1 < a) :
    IntervalIntegrable (fun t : ℝ => (1 - t ^ 2) ^ a) volume 0 1 := by
  have h1 : IntervalIntegrable (fun t : ℝ => (1 - t) ^ a) volume 0 1 := by
    have := (intervalIntegral.intervalIntegrable_rpow' (a := 0) (b := 1) ha).comp_sub_left 1
    simpa using this.symm
  have h2 : ContinuousOn (fun t : ℝ => (1 + t) ^ a) (Set.uIcc (0:ℝ) 1) := by
    apply ContinuousOn.rpow_const (by fun_prop)
    intro x hx
    rw [Set.uIcc_of_le (by norm_num : (0:ℝ) ≤ 1)] at hx
    left; nlinarith [hx.1]
  refine (h1.mul_continuousOn h2).congr ?_
  intro x hx
  rw [Set.uIoc_of_le (by norm_num : (0:ℝ) ≤ 1)] at hx
  have hx1 : (0:ℝ) ≤ 1 - x := by linarith [hx.2]
  have hx2 : (0:ℝ) ≤ 1 + x := by linarith [hx.1]
  simp only
  rw [← Real.mul_rpow hx1 hx2]
  ring_nf

private theorem aux_ii (a : ℝ) (ha : -1 < a) :
    IntervalIntegrable (fun t : ℝ => (1 - t ^ 2) ^ a) volume (-1) 1 := by
  have hR := aux_ii_right a ha
  have hL : IntervalIntegrable (fun t : ℝ => (1 - t ^ 2) ^ a) volume (-1) 0 := by
    have := (IntervalIntegrable.iff_comp_neg (f := fun t : ℝ => (1 - t ^ 2) ^ a)
      (a := 0) (b := 1) (by finiteness)).mp hR
    simpa using this.symm
  exact hL.trans hR

private theorem aux_I (a : ℝ) (ha : -1 < a) :
    ∫ t in (-1:ℝ)..1, (1 - t ^ 2) ^ a
      = Real.sqrt π * Real.Gamma (a + 1) / Real.Gamma (a + 3/2) := by
  have hsub := intervalIntegral.smul_integral_comp_mul_add (a := (0:ℝ)) (b := 1)
    (f := fun t => (1 - t^2) ^ a) 2 (-1)
  simp only [smul_eq_mul] at hsub
  norm_num at hsub
  have hB : ∫ x in (0:ℝ)..1, x ^ a * (1 - x) ^ a
      = Real.Gamma (a+1) * Real.Gamma (a+1) / Real.Gamma (2*a+2) := by
    have := aux_beta_real (a+1) (a+1) (by linarith) (by linarith)
    simpa [add_sub_cancel_right, show a+1+(a+1) = 2*a+2 by ring] using this
  have hcong : ∫ x in (0:ℝ)..1, (1 - (2*x + -1)^2) ^ a
      = (4:ℝ)^a * ∫ x in (0:ℝ)..1, x^a * (1-x)^a := by
    rw [← intervalIntegral.integral_const_mul]
    refine intervalIntegral.integral_congr fun x hx => ?_
    rw [Set.uIcc_of_le zero_le_one] at hx
    have h1 : (0:ℝ) ≤ x := hx.1
    have h2 : (0:ℝ) ≤ 1 - x := by linarith [hx.2]
    rw [show 1 - (2*x + -1)^2 = 4 * (x * (1-x)) by ring,
      Real.mul_rpow (by norm_num) (by positivity), Real.mul_rpow h1 h2]
    try ring
  rw [← hsub, hcong, hB]
  have hg1 : 0 < Real.Gamma (a+1) := Real.Gamma_pos_of_pos (by linarith)
  have hg2 : 0 < Real.Gamma (a+3/2) := Real.Gamma_pos_of_pos (by linarith)
  have hg3 : 0 < Real.Gamma (2*a+2) := Real.Gamma_pos_of_pos (by linarith)
  have hdup : Real.Gamma (a+1) * Real.Gamma (a+3/2)
      = Real.Gamma (2*a+2) * (2:ℝ)^(-1-2*a) * Real.sqrt π := by
    have h := Real.Gamma_mul_Gamma_add_half (a+1)
    rw [show a+1+1/2 = a+3/2 by ring, show (1:ℝ)-2*(a+1) = -1-2*a by ring,
      show 2*(a+1) = 2*a+2 by ring] at h
    exact h
  have h4 : 2 * (4:ℝ)^a * (2:ℝ)^(-1-2*a) = 1 := by
    rw [show (4:ℝ) = 2 * 2 by norm_num, Real.mul_rpow (by norm_num) (by norm_num),
      mul_assoc, mul_assoc, ← Real.rpow_add (by norm_num : (0:ℝ) < 2),
      ← Real.rpow_add (by norm_num : (0:ℝ) < 2),
      show a + (a + (-1 - 2*a)) = -1 by ring, Real.rpow_neg_one]
    norm_num
  have hkey : 2 * (4:ℝ)^a * (Real.Gamma (a+1) * Real.Gamma (a+1)) * Real.Gamma (a+3/2)
      = Real.sqrt π * Real.Gamma (a+1) * Real.Gamma (2*a+2) := by
    calc 2 * (4:ℝ)^a * (Real.Gamma (a+1) * Real.Gamma (a+1)) * Real.Gamma (a+3/2)
        = 2 * (4:ℝ)^a * Real.Gamma (a+1) * (Real.Gamma (a+1) * Real.Gamma (a+3/2)) := by ring
      _ = 2 * (4:ℝ)^a * Real.Gamma (a+1)
            * (Real.Gamma (2*a+2) * (2:ℝ)^(-1-2*a) * Real.sqrt π) := by rw [hdup]
      _ = (2 * (4:ℝ)^a * (2:ℝ)^(-1-2*a))
            * (Real.sqrt π * Real.Gamma (a+1) * Real.Gamma (2*a+2)) := by ring
      _ = _ := by rw [h4, one_mul]
  rw [show 2 * ((4:ℝ)^a * (Real.Gamma (a+1) * Real.Gamma (a+1) / Real.Gamma (2*a+2)))
      = (2 * (4:ℝ)^a * (Real.Gamma (a+1) * Real.Gamma (a+1))) / Real.Gamma (2*a+2) by ring,
    div_eq_div_iff hg3.ne' hg2.ne']
  linear_combination hkey

private theorem aux_J (a : ℝ) (ha : -1 < a) :
    ∫ t in (-1:ℝ)..1, t ^ 2 * (1 - t ^ 2) ^ a
      = Real.sqrt π * Real.Gamma (a + 1) / Real.Gamma (a + 3/2)
        - Real.sqrt π * Real.Gamma (a + 2) / Real.Gamma (a + 5/2) := by
  have hcongr : ∫ t in (-1:ℝ)..1, t ^ 2 * (1 - t ^ 2) ^ a
      = ∫ t in (-1:ℝ)..1, ((1 - t ^ 2) ^ a - (1 - t ^ 2) ^ (a + 1)) := by
    refine intervalIntegral.integral_congr fun t ht => ?_
    rw [Set.uIcc_of_le (by norm_num : (-1:ℝ) ≤ 1)] at ht
    have hs : (0:ℝ) ≤ 1 - t ^ 2 := by nlinarith [ht.1, ht.2]
    rcases eq_or_lt_of_le hs with h | h
    · have ht2 : t ^ 2 = 1 := by linarith
      rw [← h, Real.zero_rpow (by linarith : a + 1 ≠ 0), ht2]
      ring
    · rw [Real.rpow_add h a 1, Real.rpow_one]
      ring
  have hI2 := aux_I (a + 1) (by linarith)
  rw [show a + 1 + 1 = a + 2 by ring, show a + 1 + 3/2 = a + 5/2 by ring] at hI2
  rw [hcongr, intervalIntegral.integral_sub (aux_ii a ha) (aux_ii (a+1) (by linarith)),
    aux_I a ha, hI2]

theorem sphericalPdf_nonneg {d : ℕ} {t : ℝ} (ht : t ∈ Set.Icc (-1 : ℝ) 1) (hd : 2 ≤ d) :
    0 ≤ sphericalPdf d t := by
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have h1 : 0 < Real.Gamma ((d:ℝ)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h2 : 0 < Real.Gamma (((d:ℝ)-1)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h3 : 0 < Real.sqrt π := Real.sqrt_pos.mpr Real.pi_pos
  obtain ⟨h5, h6⟩ := ht
  have h4 : (0:ℝ) ≤ 1 - t ^ 2 := by nlinarith
  exact mul_nonneg (by positivity) (Real.rpow_nonneg h4 _)

/-- `f_d` is a probability density: a Beta integral. -/
theorem sphericalPdf_integral (hd : 2 ≤ d) :
    ∫ t in (-1 : ℝ)..1, sphericalPdf d t = 1 := by
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have ha : (-1:ℝ) < ((d:ℝ) - 3) / 2 := by linarith
  have h1 : 0 < Real.Gamma ((d:ℝ)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h2 : 0 < Real.Gamma (((d:ℝ)-1)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h3 : 0 < Real.sqrt π := Real.sqrt_pos.mpr Real.pi_pos
  have hI := aux_I (((d:ℝ) - 3) / 2) ha
  rw [show ((d:ℝ) - 3) / 2 + 1 = ((d:ℝ) - 1) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 3/2 = (d:ℝ) / 2 by ring] at hI
  simp only [sphericalPdf]
  rw [intervalIntegral.integral_const_mul, hI]
  field_simp

/-- `f_d` has mean zero (it is even). -/
theorem sphericalPdf_first_moment (hd : 2 ≤ d) :
    ∫ t in (-1 : ℝ)..1, t * sphericalPdf d t = 0 := by
  have hodd : ∀ t : ℝ, (fun s : ℝ => s * sphericalPdf d s) (-t)
      = -((fun s : ℝ => s * sphericalPdf d s) t) := by
    intro t
    simp only [sphericalPdf, neg_mul, neg_inj]
    rw [show (-t) ^ 2 = t ^ 2 by ring]
  have h := intervalIntegral.integral_comp_neg (a := (-1:ℝ)) (b := 1)
    (fun s : ℝ => s * sphericalPdf d s)
  simp only [hodd, intervalIntegral.integral_neg] at h
  norm_num at h
  linarith

/-- `f_d` has variance `1/d`, matching `§3b`'s `second_moment_eq_inv_card`.  That these two
routes agree is the consistency check between Lemma 1 and Proposition 2. -/
theorem sphericalPdf_second_moment (hd : 2 ≤ d) :
    ∫ t in (-1 : ℝ)..1, t ^ 2 * sphericalPdf d t = 1 / (d : ℝ) := by
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have ha : (-1:ℝ) < ((d:ℝ) - 3) / 2 := by linarith
  have h1 : 0 < Real.Gamma ((d:ℝ)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h2 : 0 < Real.Gamma (((d:ℝ)-1)/2) := Real.Gamma_pos_of_pos (by linarith)
  have h3 : 0 < Real.sqrt π := Real.sqrt_pos.mpr Real.pi_pos
  have h5 : 0 < Real.Gamma (((d:ℝ)+2)/2) := Real.Gamma_pos_of_pos (by linarith)
  -- the two Gamma recurrences that make the answer collapse to `1/d`
  have e1 : Real.Gamma (((d:ℝ)+1)/2) = (((d:ℝ)-1)/2) * Real.Gamma (((d:ℝ)-1)/2) := by
    have := Real.Gamma_add_one (s := ((d:ℝ)-1)/2) (by intro h; linarith [h])
    rw [show ((d:ℝ)-1)/2 + 1 = ((d:ℝ)+1)/2 by ring] at this
    exact this
  have e2 : Real.Gamma (((d:ℝ)+2)/2) = ((d:ℝ)/2) * Real.Gamma ((d:ℝ)/2) := by
    have := Real.Gamma_add_one (s := (d:ℝ)/2) (by intro h; linarith [h])
    rw [show (d:ℝ)/2 + 1 = ((d:ℝ)+2)/2 by ring] at this
    exact this
  have hJ := aux_J (((d:ℝ) - 3) / 2) ha
  rw [show ((d:ℝ) - 3) / 2 + 1 = ((d:ℝ) - 1) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 3/2 = (d:ℝ) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 2 = ((d:ℝ) + 1) / 2 by ring,
    show ((d:ℝ) - 3) / 2 + 5/2 = ((d:ℝ) + 2) / 2 by ring] at hJ
  have hmul : ∀ t : ℝ, t ^ 2 * sphericalPdf d t
      = (Real.Gamma ((d : ℝ) / 2) / (Real.sqrt π * Real.Gamma (((d : ℝ) - 1) / 2)))
        * (t ^ 2 * (1 - t ^ 2) ^ (((d : ℝ) - 3) / 2)) := by
    intro t; simp only [sphericalPdf]; ring
  simp only [hmul]
  rw [intervalIntegral.integral_const_mul, hJ, e1, e2]
  have hdne : (d:ℝ) ≠ 0 := by linarith
  field_simp
  ring

/-! ### §4b  The coordinate law, and the link to `ρ`

The audit of the first draft observed, correctly, that `ν` in §3b/§6 and `sphericalPdf` in §4
were formally unrelated symbols that the prose silently identifies.  `sphericalCoordLaw d` is
that identification made explicit: the probability measure on `ℝ` whose density is `f_d`.  With
it, `rhoSq_eq_ratio` applies to the paper's actual `ρ` with no side hypothesis. -/

/-- The coordinate law claimed by Lemma 1: the measure on `ℝ` with density `f_d` on `[-1,1]`. -/
noncomputable def sphericalCoordLaw (d : ℕ) : Measure ℝ :=
  (volume.restrict (Set.Icc (-1 : ℝ) 1)).withDensity fun t => ENNReal.ofReal (sphericalPdf d t)

/-! The bridge from the interval integrals above to the measure `sphericalCoordLaw` is pure
bookkeeping: `Set.Icc (-1) 1` and `Set.Ioc (-1) 1` differ by a Lebesgue-null set, and
`∫ x in a..b` *is* the integral against `volume.restrict (Set.Ioc a b)` when `a ≤ b`. -/

private theorem aux_restrict_Icc_eq (a b : ℝ) :
    volume.restrict (Set.Icc a b) = volume.restrict (Set.Ioc a b) :=
  (Measure.restrict_congr_set Ioc_ae_eq_Icc).symm

private theorem aux_meas_pdf (d : ℕ) : Measurable (sphericalPdf d) := by
  unfold sphericalPdf
  fun_prop

private theorem aux_ii_pdf (hd : 2 ≤ d) :
    IntervalIntegrable (sphericalPdf d) volume (-1) 1 := by
  have hdR : (2:ℝ) ≤ (d:ℝ) := by exact_mod_cast hd
  have ha : (-1:ℝ) < ((d:ℝ) - 3) / 2 := by linarith
  unfold sphericalPdf
  exact (aux_ii (((d:ℝ) - 3) / 2) ha).const_mul
      (Real.Gamma ((d : ℝ) / 2) / (Real.sqrt π * Real.Gamma (((d : ℝ) - 1) / 2)))

private theorem aux_integrable_pdf (hd : 2 ≤ d) :
    Integrable (sphericalPdf d) (volume.restrict (Set.Icc (-1:ℝ) 1)) := by
  rw [aux_restrict_Icc_eq]
  exact (intervalIntegrable_iff_integrableOn_Ioc_of_le (by norm_num)).mp (aux_ii_pdf hd)

private theorem aux_integral_pdf (hd : 2 ≤ d) :
    ∫ t, sphericalPdf d t ∂(volume.restrict (Set.Icc (-1:ℝ) 1)) = 1 := by
  rw [aux_restrict_Icc_eq, ← intervalIntegral.integral_of_le (by norm_num : (-1:ℝ) ≤ 1)]
  exact sphericalPdf_integral hd

private theorem aux_integral_sq_pdf (hd : 2 ≤ d) :
    ∫ t, t ^ 2 * sphericalPdf d t ∂(volume.restrict (Set.Icc (-1:ℝ) 1)) = 1 / (d : ℝ) := by
  rw [aux_restrict_Icc_eq, ← intervalIntegral.integral_of_le (by norm_num : (-1:ℝ) ≤ 1)]
  exact sphericalPdf_second_moment hd

/-- `f_d` really is a probability density. -/
theorem isProbabilityMeasure_sphericalCoordLaw (hd : 2 ≤ d) :
    IsProbabilityMeasure (sphericalCoordLaw d) := by
  constructor
  have hnn : 0 ≤ᵐ[volume.restrict (Set.Icc (-1:ℝ) 1)] sphericalPdf d := by
    rw [Filter.EventuallyLE, ae_restrict_iff' measurableSet_Icc]
    exact Filter.Eventually.of_forall fun t ht => sphericalPdf_nonneg ht hd
  rw [sphericalCoordLaw, withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ,
    ← ofReal_integral_eq_lintegral_ofReal (aux_integrable_pdf hd) hnn,
    aux_integral_pdf hd, ENNReal.ofReal_one]

/-- The second moment of the coordinate law is `1/d`, matching what `second_moment_eq_inv_card`
of §3b derives from `‖u‖ = 1` alone.  That these two routes agree is the only place where
Lemma 1 and Proposition 2 make contact. -/
theorem integral_sq_sphericalCoordLaw (hd : 2 ≤ d) :
    ∫ t, t ^ 2 ∂(sphericalCoordLaw d) = 1 / (d : ℝ) := by
  rw [sphericalCoordLaw,
    integral_withDensity_eq_integral_toReal_smul
      (f := fun t => ENNReal.ofReal (sphericalPdf d t))
      (ENNReal.measurable_ofReal.comp (aux_meas_pdf d))
      (Filter.Eventually.of_forall fun t => ENNReal.ofReal_lt_top)]
  rw [show (∫ t, (ENNReal.ofReal (sphericalPdf d t)).toReal • t ^ 2
        ∂(volume.restrict (Set.Icc (-1:ℝ) 1)))
      = ∫ t, t ^ 2 * sphericalPdf d t ∂(volume.restrict (Set.Icc (-1:ℝ) 1)) from
    setIntegral_congr_fun measurableSet_Icc fun t ht => by
      rw [ENNReal.toReal_ofReal (sphericalPdf_nonneg ht hd), smul_eq_mul, mul_comm]]
  exact aux_integral_sq_pdf hd

end Marginal

/-! ## §5  Theorem 1 (re-isotropisation)

The paper: *"Let `r ∈ ℝ^d \ {0}` be a random vector and `R` Haar on `O(d)`, independent of `r`.
Then `u := Rr/‖Rr‖` is uniform on `S^{d-1}` and independent of `‖r‖`, regardless of the
distribution of `r`."*

The proof given in the paper is a right-invariance argument, and right-invariance is genuinely
all it needs.  Formalising it abstractly makes that visible: the statement is about a compact
group acting transitively on a set, and has nothing to do with spheres, rotations, or `ℝ^d`.

`orbitLaw κ x₀` is the law of `g • x₀` for `g ∼ κ`.

**Read the scope of this section carefully.**  `orbitLaw κ x₀` is *defined* as a pushforward.
Nothing below identifies it with the normalised surface measure on `S^{d-1}`, so the paper's
phrase "`u` is uniform on the sphere" is **not** established here: that identification needs
(a) Haar measure on the orthogonal group, (b) uniqueness of the rotation-invariant probability
measure on the sphere, and (c) a Mathlib handle on the uniform measure on `Metric.sphere`.
Mathlib currently supplies none of the three — there is no `MeasurableSpace` instance on
`E ≃ₗᵢ[ℝ] E` and no Haar measure on the orthogonal group — so the abstract results below cannot
presently even be *instantiated* at the paper's setting.  What is proved is the
distributional-invariance half of Theorem 1: the law of `R • x` is the same for every `x` in the
orbit, whatever the law of `x`.  That is the paper's right-invariance argument, and it is the
part Proposition 2 consumes; the "uniform" wording is stronger than the proof. -/

section Reisotropisation

variable {G X : Type*} [Group G] [MeasurableSpace G] [MeasurableSpace X]
variable [MulAction G X] [MeasurableSMul G X]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- The law of `g • x` when `g ∼ κ`. -/
noncomputable def orbitLaw (κ : Measure G) (x : X) : Measure X := κ.map (fun g => g • x)

/-- **The whole content of Theorem 1.**  Right-invariance of Haar measure makes the orbit law
*constant along the orbit*: it is the same measure whichever base point of the orbit you push
forward from.  (Paper: "by right invariance of the Haar measure, `R T_a =ᵈ R`".) -/
theorem orbitLaw_smul (κ : Measure G) [κ.IsMulRightInvariant] [MeasurableMul G] (h : G) (x : X) :
    orbitLaw κ (h • x) = orbitLaw κ x := by
  unfold orbitLaw
  have hcomp : (fun g : G => g • (h • x)) = (fun g : G => g • x) ∘ (fun g : G => g * h) := by
    funext g; simp [mul_smul]
  rw [hcomp, ← Measure.map_map (measurable_smul_const x) (measurable_mul_const h),
    map_mul_right_eq_self]

/-- Consequently the orbit law does not depend on the base point chosen within an orbit. -/
theorem orbitLaw_eq_of_mem_orbit (κ : Measure G) [κ.IsMulRightInvariant] [MeasurableMul G]
    {x₀ x : X} (hx : x ∈ MulAction.orbit G x₀) : orbitLaw κ x = orbitLaw κ x₀ := by
  obtain ⟨g, rfl⟩ := hx
  exact orbitLaw_smul κ g x₀

/-! ### Auxiliary machinery for the three probabilistic statements

Everything past the two lemmas just proved needs the action to be **jointly** measurable,
`MeasurableSMul₂ G X` (i.e. `(g, x) ↦ g • x` measurable on `G × X`), and not merely separately
measurable in each variable, which is all that `MeasurableSMul G X` provides.  The `aux_`
lemmas below record the complete proofs under that extra instance; see the `FIXME`s for why the
three `theorem`s *as stated* cannot be discharged from `MeasurableSMul` alone. -/

private theorem aux_orbitLaw_apply (κ : Measure G) (x : X) {s : Set X} (hs : MeasurableSet s) :
    orbitLaw κ x s = κ {g | g • x ∈ s} :=
  Measure.map_apply (measurable_smul_const x) hs

private theorem aux_isProbabilityMeasure_orbitLaw (κ : Measure G) [IsProbabilityMeasure κ] (x : X) :
    IsProbabilityMeasure (orbitLaw κ x) :=
  Measure.isProbabilityMeasure_map (measurable_smul_const x).aemeasurable

/-- The joint law of `(R • r, r)` is the product of the orbit law and the law of `r`.  This is
the single computation behind all three of `law_smul_eq_orbitLaw`, `indepFun_smul_left` and
`indepFun_smul_comp`; it is proved here in full, but only under `[MeasurableSMul₂ G X]`. -/
private theorem aux_map_pair [MeasurableSMul₂ G X] [IsProbabilityMeasure μ]
    {κ : Measure G} [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] [MeasurableMul G]
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    μ.map (fun ω => (R ω • r ω, r ω)) = (orbitLaw κ x₀).prod (μ.map r) := by
  haveI := aux_isProbabilityMeasure_orbitLaw κ x₀
  haveI : IsProbabilityMeasure (μ.map r) :=
    Measure.isProbabilityMeasure_map hrmeas.aemeasurable
  have hΦ : Measurable (fun p : G × X => (p.1 • p.2, p.2)) := by fun_prop
  have hpair : μ.map (fun ω => (R ω, r ω)) = κ.prod (μ.map r) := by
    rw [hindep.map_prod_eq_prod_map_map hRmeas.aemeasurable hrmeas.aemeasurable, hR]
  have h1 : μ.map (fun ω => (R ω • r ω, r ω))
      = (κ.prod (μ.map r)).map (fun p : G × X => (p.1 • p.2, p.2)) := by
    rw [← hpair, Measure.map_map hΦ (hRmeas.prodMk hrmeas)]; rfl
  rw [h1]
  refine (Measure.prod_eq ?_).symm
  intro s t hs ht
  have hst : MeasurableSet ((fun p : G × X => (p.1 • p.2, p.2)) ⁻¹' (s ×ˢ t)) := hΦ (hs.prod ht)
  rw [Measure.map_apply hΦ (hs.prod ht), Measure.prod_apply_symm hst]
  set F : X → ℝ≥0∞ :=
    fun y => κ ((fun x : G => (x, y)) ⁻¹' ((fun p : G × X => (p.1 • p.2, p.2)) ⁻¹' (s ×ˢ t)))
    with hF
  have hFmeas : Measurable F := measurable_measure_prodMk_right hst
  -- the orbit hypothesis turns the inner measure into the *constant* `orbitLaw κ x₀ s`
  have hstep : ∀ᵐ ω ∂μ, F (r ω) = t.indicator (fun _ => orbitLaw κ x₀ s) (r ω) := by
    filter_upwards [horb] with ω horbω
    by_cases hyt : r ω ∈ t
    · have he : (fun x : G => (x, r ω)) ⁻¹'
          ((fun p : G × X => (p.1 • p.2, p.2)) ⁻¹' (s ×ˢ t)) = {g : G | g • r ω ∈ s} := by
        ext g; simp [hyt]
      simp only [hF, he, Set.indicator_of_mem hyt]
      rw [← aux_orbitLaw_apply κ (r ω) hs, orbitLaw_eq_of_mem_orbit κ horbω]
    · have he : (fun x : G => (x, r ω)) ⁻¹'
          ((fun p : G × X => (p.1 • p.2, p.2)) ⁻¹' (s ×ˢ t)) = ∅ := by
        ext g; simp [hyt]
      simp only [hF, he, Set.indicator_of_notMem hyt, measure_empty]
  rw [lintegral_map hFmeas hrmeas]
  calc ∫⁻ ω, F (r ω) ∂μ
      = ∫⁻ ω, (r ⁻¹' t).indicator (fun _ => orbitLaw κ x₀ s) ω ∂μ := by
        refine lintegral_congr_ae ?_
        filter_upwards [hstep] with ω hstepω
        rw [hstepω]
        by_cases hyt : r ω ∈ t <;> simp [hyt, Set.indicator_of_mem, Set.indicator_of_notMem]
    _ = orbitLaw κ x₀ s * (μ.map r) t := by
        rw [lintegral_indicator (hrmeas ht), setLIntegral_const, Measure.map_apply hrmeas ht]

/-- `law_smul_eq_orbitLaw`, proved — but with the joint-measurability instance added. -/
private theorem aux_law_smul_eq_orbitLaw [MeasurableSMul₂ G X] [IsProbabilityMeasure μ]
    {κ : Measure G} [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] [MeasurableMul G]
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    μ.map (fun ω => R ω • r ω) = orbitLaw κ x₀ := by
  haveI : IsProbabilityMeasure (μ.map r) :=
    Measure.isProbabilityMeasure_map hrmeas.aemeasurable
  have hpm : Measurable (fun ω => (R ω • r ω, r ω)) := (hRmeas.smul hrmeas).prodMk hrmeas
  have h := congrArg (Measure.map Prod.fst) (aux_map_pair hRmeas hrmeas hR hindep horb)
  rwa [Measure.map_map measurable_fst hpm, Measure.map_fst_prod, measure_univ, one_smul] at h

/-- `indepFun_smul_left`, proved — but with the joint-measurability instance added. -/
private theorem aux_indepFun_smul_left [MeasurableSMul₂ G X] [IsProbabilityMeasure μ]
    {κ : Measure G} [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] [MeasurableMul G]
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    IndepFun (fun ω => R ω • r ω) r μ := by
  refine (indepFun_iff_map_prod_eq_prod_map_map
    (hRmeas.smul hrmeas).aemeasurable hrmeas.aemeasurable).2 ?_
  show μ.map (fun ω => (R ω • r ω, r ω)) = (μ.map (fun ω => R ω • r ω)).prod (μ.map r)
  rw [aux_map_pair hRmeas hrmeas hR hindep horb,
    aux_law_smul_eq_orbitLaw hRmeas hrmeas hR hindep horb]

/-- `indepFun_smul_comp`, proved — but with the joint-measurability instance added. -/
private theorem aux_indepFun_smul_comp [MeasurableSMul₂ G X] [IsProbabilityMeasure μ]
    {κ : Measure G} [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] [MeasurableMul G]
    {β : Type*} [MeasurableSpace β] {f : X → β}
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hfmeas : Measurable f) (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    IndepFun (fun ω => R ω • r ω) (fun ω => f (r ω)) μ :=
  (aux_indepFun_smul_left hRmeas hrmeas hR hindep horb).comp measurable_id hfmeas

/-- **Theorem 1, abstract form, part 1 (the law).**  If `R ∼ κ` is Haar-distributed and
independent of `r`, and `r` lives in a single orbit, then `R • r` has the orbit law — *whatever
the law of `r`*.  This is the "`r` may be anything" clause that licenses reuse of one codebook
across residual passes. -/
theorem law_smul_eq_orbitLaw [IsProbabilityMeasure μ] {κ : Measure G} [IsProbabilityMeasure κ]
    [κ.IsMulRightInvariant] [MeasurableMul G] [MeasurableSMul₂ G X]
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    μ.map (fun ω => R ω • r ω) = orbitLaw κ x₀ :=
  aux_law_smul_eq_orbitLaw hRmeas hrmeas hR hindep horb

/-- **Theorem 1, abstract form, part 2 (the independence).**  `R • r` is independent of `r`.
This — not the uniformity — is what Proposition 2 consumes, via `u_p ⟂ s_p`. -/
theorem indepFun_smul_left [IsProbabilityMeasure μ] {κ : Measure G} [IsProbabilityMeasure κ]
    [κ.IsMulRightInvariant] [MeasurableMul G] [MeasurableSMul₂ G X]
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    IndepFun (fun ω => R ω • r ω) r μ :=
  aux_indepFun_smul_left hRmeas hrmeas hR hindep horb

/-- Composing: any function of `r` — in particular `‖r‖` — is independent of `R • r`. -/
theorem indepFun_smul_comp [IsProbabilityMeasure μ] {κ : Measure G} [IsProbabilityMeasure κ]
    [κ.IsMulRightInvariant] [MeasurableMul G] [MeasurableSMul₂ G X] {β : Type*} [MeasurableSpace β] {f : X → β}
    {R : Ω → G} {r : Ω → X} (hRmeas : Measurable R) (hrmeas : Measurable r)
    (hfmeas : Measurable f) (hR : μ.map R = κ) (hindep : IndepFun R r μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    IndepFun (fun ω => R ω • r ω) (fun ω => f (r ω)) μ :=
  aux_indepFun_smul_comp hRmeas hrmeas hfmeas hR hindep horb

/-- The joint law of `(R • r, Z)` for an auxiliary variable `Z` carried alongside `r`.  This is
`aux_map_pair` with the action applied to the *first* component of `X × β` while `β` is dragged
through untouched: the orbit hypothesis still makes the inner `κ`-measure the constant
`orbitLaw κ x₀ s`, so Fubini factorises exactly as before.  Note that the second factor is the
law of `Z` alone — all information about `r` beyond its orbit has been washed out. -/
private theorem aux_map_pair_snd [MeasurableSMul₂ G X] [IsProbabilityMeasure μ]
    {κ : Measure G} [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] [MeasurableMul G]
    {β : Type*} [MeasurableSpace β]
    {R : Ω → G} {r : Ω → X} {Z : Ω → β}
    (hRmeas : Measurable R) (hrmeas : Measurable r) (hZmeas : Measurable Z)
    (hR : μ.map R = κ) (hindep : IndepFun R (fun ω => (r ω, Z ω)) μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    μ.map (fun ω => (R ω • r ω, Z ω)) = (orbitLaw κ x₀).prod (μ.map Z) := by
  haveI := aux_isProbabilityMeasure_orbitLaw κ x₀
  haveI : IsProbabilityMeasure (μ.map Z) :=
    Measure.isProbabilityMeasure_map hZmeas.aemeasurable
  have hYmeas : Measurable (fun ω => (r ω, Z ω)) := hrmeas.prodMk hZmeas
  haveI : IsProbabilityMeasure (μ.map (fun ω => (r ω, Z ω))) :=
    Measure.isProbabilityMeasure_map hYmeas.aemeasurable
  have hΦ : Measurable (fun p : G × (X × β) => (p.1 • p.2.1, p.2.2)) := by fun_prop
  have hpair : μ.map (fun ω => (R ω, (r ω, Z ω)))
      = κ.prod (μ.map fun ω => (r ω, Z ω)) := by
    rw [hindep.map_prod_eq_prod_map_map hRmeas.aemeasurable hYmeas.aemeasurable, hR]
  have h1 : μ.map (fun ω => (R ω • r ω, Z ω))
      = (κ.prod (μ.map fun ω => (r ω, Z ω))).map
          (fun p : G × (X × β) => (p.1 • p.2.1, p.2.2)) := by
    rw [← hpair, Measure.map_map hΦ (hRmeas.prodMk hYmeas)]; rfl
  rw [h1]
  refine (Measure.prod_eq ?_).symm
  intro s t hs ht
  have hst : MeasurableSet
      ((fun p : G × (X × β) => (p.1 • p.2.1, p.2.2)) ⁻¹' (s ×ˢ t)) := hΦ (hs.prod ht)
  rw [Measure.map_apply hΦ (hs.prod ht), Measure.prod_apply_symm hst]
  set F : X × β → ℝ≥0∞ :=
    fun y => κ ((fun x : G => (x, y)) ⁻¹'
      ((fun p : G × (X × β) => (p.1 • p.2.1, p.2.2)) ⁻¹' (s ×ˢ t))) with hF
  have hFmeas : Measurable F := measurable_measure_prodMk_right hst
  -- the orbit hypothesis turns the inner measure into the *constant* `orbitLaw κ x₀ s`,
  -- independently of `ω`; only the `Z`-slice survives
  have hstep : ∀ᵐ ω ∂μ, F (r ω, Z ω) = (Z ⁻¹' t).indicator (fun _ => orbitLaw κ x₀ s) ω := by
    filter_upwards [horb] with ω horbω
    by_cases hyt : Z ω ∈ t
    · have he : (fun x : G => (x, (r ω, Z ω))) ⁻¹'
          ((fun p : G × (X × β) => (p.1 • p.2.1, p.2.2)) ⁻¹' (s ×ˢ t))
            = {g : G | g • r ω ∈ s} := by
        ext g; simp [hyt]
      simp only [hF, he, Set.indicator_of_mem (show ω ∈ Z ⁻¹' t from hyt)]
      rw [← aux_orbitLaw_apply κ (r ω) hs, orbitLaw_eq_of_mem_orbit κ horbω]
    · have he : (fun x : G => (x, (r ω, Z ω))) ⁻¹'
          ((fun p : G × (X × β) => (p.1 • p.2.1, p.2.2)) ⁻¹' (s ×ˢ t)) = ∅ := by
        ext g; simp [hyt]
      simp only [hF, he, Set.indicator_of_notMem (show ω ∉ Z ⁻¹' t from hyt), measure_empty]
  rw [lintegral_map hFmeas hYmeas]
  calc ∫⁻ ω, F (r ω, Z ω) ∂μ
      = ∫⁻ ω, (Z ⁻¹' t).indicator (fun _ => orbitLaw κ x₀ s) ω ∂μ := lintegral_congr_ae hstep
    _ = orbitLaw κ x₀ s * (μ.map Z) t := by
        rw [lintegral_indicator (hZmeas ht), setLIntegral_const, Measure.map_apply hZmeas ht]

/-- **The statement Proposition 2 actually consumes.**

`indepFun_smul_comp` above is, at `G = O(d)` acting on `ℝ^d`, *vacuous* for the paper's purpose:
the hypothesis `∀ ω, r ω ∈ orbit G x₀` forces `‖r ω‖` to be a.s. constant, so these lemmas can
only be instantiated with the *normalised* residual `r̂ = r/‖r‖` — and then `f(r̂) = ‖r̂‖ ≡ 1`.
The paper's claim is `u ⟂ ‖r‖`, and `‖r‖` is not a function of `r̂`.

The correct statement carries the scale along as a separate variable `Z`: `R • r̂` is independent
of *any* auxiliary variable `Z` that is jointly independent of the fresh rotation.  Taking
`Z = ‖r‖` gives the paper's "`u` is independent of `‖r‖`", and taking `Z = (‖r‖, R_0, …, R_{p-1})`
gives the inductive step across passes. -/
theorem indepFun_smul_of_indepPair [IsProbabilityMeasure μ] {κ : Measure G}
    [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] [MeasurableMul G] [MeasurableSMul₂ G X]
    {β : Type*} [MeasurableSpace β]
    {R : Ω → G} {r : Ω → X} {Z : Ω → β}
    (hRmeas : Measurable R) (hrmeas : Measurable r) (hZmeas : Measurable Z)
    (hR : μ.map R = κ) (hindep : IndepFun R (fun ω => (r ω, Z ω)) μ)
    {x₀ : X} (horb : ∀ᵐ ω ∂μ, r ω ∈ MulAction.orbit G x₀) :
    IndepFun (fun ω => R ω • r ω) Z μ := by
  haveI := aux_isProbabilityMeasure_orbitLaw κ x₀
  haveI : IsProbabilityMeasure (μ.map Z) :=
    Measure.isProbabilityMeasure_map hZmeas.aemeasurable
  have hmp := aux_map_pair_snd hRmeas hrmeas hZmeas hR hindep horb
  -- the first marginal of that joint law is the orbit law: the scale variable integrates out
  have hfst : μ.map (fun ω => R ω • r ω) = orbitLaw κ x₀ := by
    have hpm : Measurable (fun ω => (R ω • r ω, Z ω)) := (hRmeas.smul hrmeas).prodMk hZmeas
    have h := congrArg (Measure.map Prod.fst) hmp
    rwa [Measure.map_map measurable_fst hpm, Measure.map_fst_prod, measure_univ, one_smul] at h
  refine (indepFun_iff_map_prod_eq_prod_map_map
    (hRmeas.smul hrmeas).aemeasurable hZmeas.aemeasurable).2 ?_
  show μ.map (fun ω => (R ω • r ω, Z ω)) = (μ.map fun ω => R ω • r ω).prod (μ.map Z)
  rw [hmp, hfst]

end Reisotropisation
/-! ### §5b  Instantiating at the sphere

The orbit of a unit vector under the linear isometry group is the whole unit sphere: this is the
transitivity the paper invokes when it writes "complete `â` to an orthonormal basis". -/

section Sphere

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- The tautological action of the isometry group on the space.  (Mathlib has
`LinearEquiv.applyDistribMulAction` but no isometric analogue, so we supply it.) -/
scoped instance isoApplyMulAction : MulAction (E ≃ₗᵢ[ℝ] E) E where
  smul L x := L x
  one_smul _ := rfl
  mul_smul _ _ _ := rfl

@[simp] theorem isoApplyMulAction_smul (L : E ≃ₗᵢ[ℝ] E) (x : E) : L • x = L x := rfl

/-- A unit vector occurs as a prescribed member of some orthonormal basis indexed by
`Fin (finrank ℝ E)`. -/
private theorem aux_exists_orthonormalBasis_apply_eq {u : E} (hu : ‖u‖ = 1)
    (i₀ : Fin (Module.finrank ℝ E)) :
    ∃ b : OrthonormalBasis (Fin (Module.finrank ℝ E)) ℝ E, b i₀ = u := by
  have ho : Orthonormal ℝ (Set.restrict ({i₀} : Set (Fin (Module.finrank ℝ E))) (fun _ => u)) := by
    rw [orthonormal_subsingleton_iff]
    intro i; simpa using hu
  obtain ⟨b, hb⟩ := ho.exists_orthonormalBasis_extension_of_card_eq (by simp)
  exact ⟨b, hb i₀ rfl⟩

/-- Transitivity: the isometry group acts transitively on the unit sphere. -/
theorem exists_isometry_map_eq {u v : E} (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) :
    ∃ L : E ≃ₗᵢ[ℝ] E, L u = v := by
  -- a unit vector is nonzero, so the space has positive rank and `Fin (finrank ℝ E)` is inhabited
  have hune : u ≠ 0 := by
    intro h; rw [h, norm_zero] at hu; norm_num at hu
  have hpos : 0 < Module.finrank ℝ E := Module.finrank_pos_iff_exists_ne_zero.2 ⟨u, hune⟩
  obtain ⟨bu, hbu⟩ := aux_exists_orthonormalBasis_apply_eq hu ⟨0, hpos⟩
  obtain ⟨bv, hbv⟩ := aux_exists_orthonormalBasis_apply_eq hv ⟨0, hpos⟩
  refine ⟨bu.equiv bv (Equiv.refl _), ?_⟩
  rw [← hbu, bu.equiv_apply_basis, Equiv.refl_apply, hbv]

/-- Hence the unit sphere is a single orbit. -/
theorem orbit_eq_sphere {u : E} (hu : ‖u‖ = 1) :
    MulAction.orbit (E ≃ₗᵢ[ℝ] E) u = {v : E | ‖v‖ = 1} := by
  ext v
  constructor
  · rintro ⟨L, rfl⟩
    simpa using hu
  · intro hv
    obtain ⟨L, hL⟩ := exists_isometry_map_eq hu hv
    exact ⟨L, by simpa using hL⟩

end Sphere

/-! ## §6  Corollary 1 and the design rule

Corollary 1: since every pass sees the same marginal, the codebook minimising the per-coordinate
MSE minimises it at *every* pass, hence minimises `ρ`, hence minimises the whole envelope `ρ^P`.

The formal content is a monotonicity statement — trivial, and worth stating precisely for that
reason: the corollary is a consequence of Theorem 1, not an independent optimality theorem, and
its scope is *scalar quantisers applied coordinate-wise*. -/

section Optimality

variable {d : ℕ} {ν : Measure ℝ}

/-- Per-coordinate distortion of a scalar quantiser under `ν`. -/
noncomputable def distortion (cb : ℝ → ℝ) (ν : Measure ℝ) : ℝ := ∫ t, (t - cb t) ^ 2 ∂ν

/-- `ρ = √(d · D(cb))`. -/
noncomputable def rho (d : ℕ) (cb : ℝ → ℝ) (ν : Measure ℝ) : ℝ :=
  Real.sqrt (rhoSq d cb ν)

/-- **Corollary 1, per pass.**  A codebook with smaller per-coordinate distortion has a smaller
decay constant. -/
theorem rho_le_of_distortion_le {cb₁ cb₂ : ℝ → ℝ} (h : distortion cb₁ ν ≤ distortion cb₂ ν) :
    rho d cb₁ ν ≤ rho d cb₂ ν := by
  unfold rho rhoSq
  exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left h (Nat.cast_nonneg d))

/-- **Corollary 1, whole envelope.**  …and therefore a smaller error at *every* pass count:
optimality per pass upgrades to optimality of the geometric envelope `ρ^P`.  This is the exact
sense in which "the fixed density-matched codebook is optimal not only per pass but for the
entire geometric envelope". -/
theorem rho_pow_le_of_distortion_le {cb₁ cb₂ : ℝ → ℝ}
    (h : distortion cb₁ ν ≤ distortion cb₂ ν) (P : ℕ) :
    rho d cb₁ ν ^ P ≤ rho d cb₂ ν ^ P := by
  exact pow_le_pow_left₀ (Real.sqrt_nonneg _) (rho_le_of_distortion_le h) P

/-- **Corollary 1 as stated**: if `cb★` minimises the per-coordinate distortion over a class `Q`
of `2^b`-level scalar quantisers, it minimises the `P`-pass error for every `P`, so no per-pass
re-fitting can improve on it. -/
theorem optimal_codebook_optimal_at_every_pass {Q : Set (ℝ → ℝ)} {cb : ℝ → ℝ}
    (hmem : cb ∈ Q) (hmin : ∀ c ∈ Q, distortion cb ν ≤ distortion c ν) (P : ℕ) :
    ∀ c ∈ Q, rho d cb ν ^ P ≤ rho d c ν ^ P :=
  fun c hc => rho_pow_le_of_distortion_le (hmin c hc) P

/-! ### The pass-count design rule -/

/-- **`P = ⌈ln ε / ln ρ⌉` passes suffice** to reach relative error `ε`.  With `ρ` computable by
quadrature (§4), this is the paper's data-free design rule: the number of passes is known before
any weight is touched. -/
theorem rho_pow_ceil_le {ρ ε : ℝ} (hρ0 : 0 < ρ) (hρ1 : ρ < 1) (hε0 : 0 < ε) (hε1 : ε ≤ 1) :
    ρ ^ (⌈Real.log ε / Real.log ρ⌉₊) ≤ ε := by
  have hL : Real.log ρ < 0 := Real.log_neg hρ0 hρ1
  set n : ℕ := ⌈Real.log ε / Real.log ρ⌉₊ with hn
  -- `n * log ρ ≤ log ε`, the inequality flipping because `log ρ < 0`.
  have hstep : (n : ℝ) * Real.log ρ ≤ Real.log ε := by
    have := mul_le_mul_of_nonpos_right (Nat.le_ceil (Real.log ε / Real.log ρ)) hL.le
    rwa [div_mul_cancel₀ _ hL.ne] at this
  rw [← Real.log_le_log_iff (pow_pos hρ0 n) hε0, Real.log_pow]
  exact hstep

/-- …and it is the least such `P`. -/
theorem le_ceil_of_rho_pow_le {ρ ε : ℝ} {P : ℕ} (hρ0 : 0 < ρ) (hρ1 : ρ < 1) (hε0 : 0 < ε)
    (h : ρ ^ P ≤ ε) : Real.log ε / Real.log ρ ≤ P := by
  have hL : Real.log ρ < 0 := Real.log_neg hρ0 hρ1
  rw [div_le_iff_of_neg hL, ← Real.log_pow]
  exact Real.log_le_log (pow_pos hρ0 P) h

/-! ### §6b  The paper's `ρ`

`rho d cb (sphericalCoordLaw d)` is the constant of eq. (2.3) as the paper defines it — the one
plotted in the design chart, computable by quadrature before any weight is touched. -/

/-- Eq. (2.3) at the Lemma 1 coordinate law: `ρ² = E[(t-q t)²] / E[t²]` with no side hypothesis,
because §4b supplies `E[t²] = 1/d`. -/
theorem rhoSq_paper_eq_ratio {d : ℕ} (hd : 2 ≤ d) (cb : ℝ → ℝ) :
    rhoSq d cb (sphericalCoordLaw d)
      = (∫ t, (t - cb t) ^ 2 ∂(sphericalCoordLaw d)) / ∫ t, t ^ 2 ∂(sphericalCoordLaw d) := by
  have hd0 : d ≠ 0 := by omega
  exact rhoSq_eq_ratio hd0 (integral_sq_sphericalCoordLaw hd)

end Optimality

/-! ## §9  Beyond scalar codebooks: where Corollary 1's optimality actually stops

Corollary 1 (§6) says the density-matched Lloyd–Max codebook is MSE-optimal *among `2^b`-level
scalar quantisers applied coordinate-wise*.  That qualifier is not a footnote.  Trellis-coded
quantisation (QTIP) and lattice codebooks (QuIP#'s E8P) exist precisely to leave that class, and
on a Gaussian source they recover a shaping gain that no scalar quantiser can reach at any rate.
A reader who takes "one codebook is optimal at every pass" at face value will be misled about
exactly the comparison the low-bit literature cares about.

Two facts, both proved below, place the boundary precisely:

* **The decay law never used the coordinate structure.**  `Scheme.q` is an arbitrary map
  `E → E`, and `IsotropicPasses.distortion` is stated as `∫‖u_p − q u_p‖² = ρ²` with no
  reference to coordinates.  So §3's geometric envelope applies verbatim to a trellis or lattice
  stage quantiser (`integral_norm_resid_sq_vectorStage`).  The progressive construction is *not*
  tied to scalar codebooks; only the evaluation of `ρ` by 1-D quadrature is.

* **Enlarging the class can only help** (`sInf_stageDistortion_le_of_subset`).  So "optimal at
  every pass" is optimality within a strictly smaller class whenever the containment is strict,
  and the attainable decay constant over a richer class is no larger.

Nothing here quantifies the gap — that is an information-theoretic statement about the source,
not a consequence of this recursion.  What is established is that the gap is on the right side of
the inequality, and that the framework does not have to pay it. -/

section VectorStage

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]

/-- The stage distortion of an **arbitrary** quantiser `Q` against the direction law `σ` — the
measure §5 calls `orbitLaw`, i.e. the law of `u_p` on the unit sphere.  This is `ρ²` of
eq. (2.3) with the coordinate structure removed. -/
noncomputable def stageDistortion (Q : E → E) (σ : Measure E) : ℝ := ∫ u, ‖u - Q u‖ ^ 2 ∂σ

theorem stageDistortion_nonneg (Q : E → E) (σ : Measure E) : 0 ≤ stageDistortion Q σ :=
  integral_nonneg fun _ => by positivity

/-- **Corollary 1's optimality is optimality within a class.**  If `Q₁ ⊆ Q₂` — for instance `Q₁`
the coordinate-wise scalar quantisers and `Q₂` those together with the lattice or trellis
quantisers of the same rate — then the best attainable stage distortion over `Q₂` is no worse
than over `Q₁`.  Corollary 1 is therefore silent about, not evidence against, the richer class. -/
theorem sInf_stageDistortion_le_of_subset {Q₁ Q₂ : Set (E → E)} (hsub : Q₁ ⊆ Q₂)
    (hne : Q₁.Nonempty) (σ : Measure E) :
    sInf ((fun Q => stageDistortion Q σ) '' Q₂) ≤ sInf ((fun Q => stageDistortion Q σ) '' Q₁) :=
  csInf_le_csInf ⟨0, fun _ hx => by obtain ⟨Q, _, rfl⟩ := hx; exact stageDistortion_nonneg Q σ⟩
    (hne.image _) (Set.image_mono hsub)

/-- …and a smaller stage distortion gives a smaller `P`-pass envelope, for every `P` — the
vector-valued form of `rho_pow_le_of_distortion_le`. -/
theorem stage_rho_pow_le_of_le {Q₁ Q₂ : E → E} {σ : Measure E}
    (h : stageDistortion Q₂ σ ≤ stageDistortion Q₁ σ) (P : ℕ) :
    Real.sqrt (stageDistortion Q₂ σ) ^ P ≤ Real.sqrt (stageDistortion Q₁ σ) ^ P :=
  pow_le_pow_left₀ (Real.sqrt_nonneg _) (Real.sqrt_le_sqrt h) P

/-- The coordinate-wise scalar quantisers built from a set `C` of scalar codebooks.  This is the
class Corollary 1 optimises over; `sInf_stageDistortion_le_of_subset` applies to any class
containing it. -/
def scalarStages (d : ℕ) (C : Set (ℝ → ℝ)) :
    Set (EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d)) :=
  {Q | ∃ cb ∈ C, Q = fun v => WithLp.toLp 2 fun i => cb (v i)}

/-- Instantiating the previous lemma: whatever richer class `Qs` a trellis or lattice stage
quantiser lives in, as long as it also contains the scalar stages, its optimum is at least as
good.  This is the formal content of "Corollary 1's scope is scalar quantisers". -/
theorem sInf_stageDistortion_scalarStages_le {d : ℕ} {C : Set (ℝ → ℝ)}
    {Qs : Set (EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))}
    (hsub : scalarStages d C ⊆ Qs) (hne : (scalarStages d C).Nonempty)
    (σ : Measure (EuclideanSpace ℝ (Fin d))) :
    sInf ((fun Q => stageDistortion Q σ) '' Qs)
      ≤ sInf ((fun Q => stageDistortion Q σ) '' scalarStages d C) :=
  sInf_stageDistortion_le_of_subset hsub hne σ

end VectorStage

/-- **The decay law is quantiser-agnostic.**  A restatement of `integral_norm_resid_sq` whose only
purpose is to be citable at an arbitrary stage quantiser: `Scheme.q` is an unrestricted map, and
neither `IsotropicPasses` nor the proof of §3 mentions coordinates, so a trellis-coded or lattice
stage quantiser inherits the same `ρ^{2P}` envelope with no change.  (This is definitionally the
same theorem; it is named separately because the paper's prose ties `ρ` to a scalar codebook and
the independence is worth making explicit.) -/
theorem integral_norm_resid_sq_vectorStage {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {S : Ω → Scheme E} {W : Ω → E} {ρ : ℝ}
    (h : IsotropicPasses S W ρ μ) (P : ℕ) :
    ∫ ω, ‖resid (S ω) (W ω) P‖ ^ 2 ∂μ = ρ ^ (2 * P) * ∫ ω, ‖W ω‖ ^ 2 ∂μ :=
  integral_norm_resid_sq h P

/-! ## §11  The bridge: a rotation stream *derives* `IsotropicPasses`

Header item 3 records that §3's Proposition 2 is conditional on `IsotropicPasses`, and that
nothing connected §5's re-isotropisation machinery to it.  This section builds that connection.

The setting is the one §5 can actually be instantiated in.  Mathlib has no `MeasurableSpace` on
`E ≃ₗᵢ[ℝ] E`, so the rotations are drawn from an abstract measurable group `G` acting on `E`,
together with a map `ι : G → (E ≃ₗᵢ[ℝ] E)` that *implements* the action (`ι g x = g • x`).  For
the paper's `O(d)` one takes `G = O(d)` with its Borel structure; for the deployed randomised
Hadamard transform, `G` is the finite sign-flip group — the results below are indifferent to
which, exactly as §5 is.

The chain is:

* `resid_rotScheme_eq_residPast` — `r_p` is a fixed function of `(w, g_0, …, g_{p-1})` only;
* `measurable_residPast` — that function is measurable;
* `unitVec_rotScheme_eq_smul` — `u_p = g_p • r̂_p` with `r̂_p` a function of the past;
* `indepFun_rot_past` — mutual independence of `(W, R_0, R_1, …)` makes the fresh `R_p`
  independent of the whole past, hence of `(r̂_p, ‖r_p‖²)`;
* `law_unitVec_eq_orbitLaw` — therefore `u_p` has the *same* law `σ = orbitLaw κ x₀` at every
  pass.  This is the statement re-isotropisation is for, and it is what makes the per-pass
  distortion the same number at every pass;
* `isotropicPasses_of_rotationStream` — assembly.

What is thereby removed from the hypothesis list of Proposition 2: `IsotropicPasses.indep` and
`IsotropicPasses.distortion`, the two substantive fields, and both integrability fields. -/

section BridgeAlgebra

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {G : Type*} [Group G] [MulAction G E]

/-- The scheme whose pass-`p` rotation is `ι (g p)` and whose (single, fixed) quantiser is `q`. -/
def rotScheme (ι : G → (E ≃ₗᵢ[ℝ] E)) (q : E → E) (g : ℕ → G) : Scheme E where
  rot p := ι (g p)
  q := q

/-- `residPast q p` computes `r_p` from the pair `(w, (g_0,…,g_{p-1}))` — the *past* alone. -/
noncomputable def residPast (q : E → E) : ∀ p : ℕ, E × (Fin p → G) → E
  | 0, x => x.1
  | (p + 1), x =>
      let r := residPast q p (x.1, fun i => x.2 i.castSucc)
      r - ‖r‖ • ((x.2 (Fin.last p))⁻¹ • q (‖r‖⁻¹ • ((x.2 (Fin.last p)) • r)))

/-- The normalised residual `r̂_p = r_p/‖r_p‖`, as a function of the past. -/
noncomputable def residDirPast (q : E → E) (p : ℕ) (x : E × (Fin p → G)) : E :=
  ‖residPast q p x‖⁻¹ • residPast q p x

variable {ι : G → (E ≃ₗᵢ[ℝ] E)}

/-- If `ι` implements the `G`-action, the inverse isometry implements the inverse action. -/
theorem symm_apply_eq_inv_smul (hι : ∀ (g : G) (x : E), ι g x = g • x) (g : G) (y : E) :
    (ι g).symm y = g⁻¹ • y := by
  have h : ι g (g⁻¹ • y) = y := by rw [hι, smul_inv_smul]
  conv_lhs => rw [← h]
  rw [LinearIsometryEquiv.symm_apply_apply]

/-- If `ι` implements the `G`-action, that action is `ℝ`-linear. -/
theorem smul_comm_real (hι : ∀ (g : G) (x : E), ι g x = g • x) (g : G) (c : ℝ) (x : E) :
    g • (c • x) = c • (g • x) := by
  rw [← hι, ← hι, map_smul]

/-- **The residual factors through the past.**  `r_p` is the value at `(w, g_0,…,g_{p-1})` of a
fixed map: it does not see `g_p, g_{p+1}, …`.  This is the inductive fact the paper argues in
prose after eq. (2.2), and it is what makes the pass-`p` rotation *fresh*. -/
theorem resid_rotScheme_eq_residPast (hι : ∀ (g : G) (x : E), ι g x = g • x) (q : E → E) (w : E)
    (g : ℕ → G) (p : ℕ) :
    resid (rotScheme ι q g) w p = residPast q p (w, fun i : Fin p => g i) := by
  induction p with
  | zero => rfl
  | succ p ih =>
      have hcast : (fun i : Fin p => (fun j : Fin (p + 1) => g j) i.castSucc)
          = fun i : Fin p => g i := by
        funext i; simp
      rw [resid_succ, unitVec, ih, residPast]
      simp only [hcast, Fin.val_last]
      rw [show (rotScheme ι q g).rot p = ι (g p) from rfl, show (rotScheme ι q g).q = q from rfl,
        symm_apply_eq_inv_smul hι, hι]

/-- **The unit direction is the fresh rotation applied to a function of the past.**  This is the
exact shape `indepFun_smul_of_indepPair` (§5) consumes. -/
theorem unitVec_rotScheme_eq_smul (hι : ∀ (g : G) (x : E), ι g x = g • x) (q : E → E) (w : E)
    (g : ℕ → G) (p : ℕ) :
    unitVec (rotScheme ι q g) w p = g p • residDirPast q p (w, fun i : Fin p => g i) := by
  rw [unitVec, residDirPast, smul_comm_real hι, resid_rotScheme_eq_residPast hι,
    show (rotScheme ι q g).rot p = ι (g p) from rfl, hι]

/-- The normalised residual is a unit vector whenever the pass is not already exact. -/
theorem norm_residDirPast (hι : ∀ (g : G) (x : E), ι g x = g • x) {q : E → E} {w : E}
    {g : ℕ → G} {p : ℕ} (h : resid (rotScheme ι q g) w p ≠ 0) :
    ‖residDirPast q p (w, fun i : Fin p => g i)‖ = 1 := by
  rw [resid_rotScheme_eq_residPast hι] at h
  rw [residDirPast, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr h)]

end BridgeAlgebra

section BridgeMeasurable

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G]

theorem measurable_residPast {q : E → E} (hq : Measurable q) (p : ℕ) :
    Measurable (residPast (G := G) q p) := by
  induction p with
  | zero => exact measurable_fst
  | succ p ih =>
      have h1 : Measurable (fun x : E × (Fin (p + 1) → G) =>
          (x.1, fun i : Fin p => x.2 i.castSucc)) :=
        measurable_fst.prodMk (measurable_pi_lambda _ fun i =>
          (measurable_pi_apply (i.castSucc)).comp measurable_snd)
      have hr : Measurable (fun x : E × (Fin (p + 1) → G) =>
          residPast (G := G) q p (x.1, fun i : Fin p => x.2 i.castSucc)) := ih.comp h1
      have hg : Measurable (fun x : E × (Fin (p + 1) → G) => x.2 (Fin.last p)) :=
        (measurable_pi_apply _).comp measurable_snd
      exact hr.sub ((hr.norm).smul
        ((hg.inv).smul (hq.comp (((hr.norm).inv).smul (hg.smul hr)))))

theorem measurable_residDirPast {q : E → E} (hq : Measurable q) (p : ℕ) :
    Measurable (residDirPast (G := G) q p) :=
  ((measurable_residPast hq p).norm.inv).smul (measurable_residPast hq p)

end BridgeMeasurable

/-! ### §11b  Mutual independence of the stream makes each rotation fresh

The remaining structural fact is that the pass-`p` rotation is independent of *everything the
recursion has already used*.  Stated for the family `(W, R_0, R_1, …)` indexed by `Option ℕ`
(`none` = the weight row), this is a disjoint-index-sets consequence of mutual independence.

The `Option ℕ`-indexed family is heterogeneous (`E` at `none`, `G` elsewhere), which forces `E`
and `G` into a common universe here; nothing downstream of this subsection needs that, because
`indepFun_rot_past_of_iIndepFun`'s *conclusion* is universe-polymorphic and is the only thing
the assembly consumes. -/

section BridgeStream

universe u

variable {E : Type u} [MeasurableSpace E] {G : Type u} [MeasurableSpace G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- The state space of the joint randomness: the weight row at `none`, a rotation at `some p`. -/
def streamSpace (E G : Type u) : Option ℕ → Type u
  | none => E
  | some _ => G

instance streamSpaceMeasurableSpace (E G : Type u) [MeasurableSpace E] [MeasurableSpace G] :
    ∀ i : Option ℕ, MeasurableSpace (streamSpace E G i)
  | none => inferInstanceAs (MeasurableSpace E)
  | some _ => inferInstanceAs (MeasurableSpace G)

/-- The joint randomness of the scheme, as one family of random variables. -/
def streamVar (W : Ω → E) (R : Ω → ℕ → G) : ∀ i : Option ℕ, Ω → streamSpace E G i
  | none => W
  | some p => fun ω => R ω p

/-- **The fresh-rotation lemma.**  If the weight row and the whole rotation stream are mutually
independent, then the pass-`p` rotation is independent of the pair *(weight row, all earlier
rotations)* — which by `resid_rotScheme_eq_residPast` is everything `r_p` depends on. -/
theorem indepFun_rot_past_of_iIndepFun {W : Ω → E} {R : Ω → ℕ → G}
    (hW : Measurable W) (hR : ∀ p, Measurable (fun ω => R ω p))
    (h : iIndepFun (streamVar W R) μ) (p : ℕ) :
    IndepFun (fun ω => R ω p) (fun ω => (W ω, fun i : Fin p => R ω i)) μ := by
  have hmeas : ∀ i, Measurable (streamVar W R i) := by
    rintro (_ | q)
    · exact hW
    · exact hR q
  have hdisj : Disjoint ({some p} : Finset (Option ℕ))
      (insert none ((Finset.range p).image some)) := by
    simp [Finset.disjoint_singleton_left]
  have key := h.indepFun_finset {some p} (insert none ((Finset.range p).image some)) hdisj hmeas
  have hmemp : some p ∈ ({some p} : Finset (Option ℕ)) := Finset.mem_singleton_self _
  have hmemn : none ∈ insert none ((Finset.range p).image some) := Finset.mem_insert_self _ _
  have hmemi : ∀ i : Fin p, some (i : ℕ) ∈ insert none ((Finset.range p).image some) := fun i =>
    Finset.mem_insert_of_mem (Finset.mem_image_of_mem some (Finset.mem_range.mpr i.isLt))
  have hφ : Measurable (fun v : ((i : ({some p} : Finset (Option ℕ))) → streamSpace E G i) =>
      v ⟨some p, hmemp⟩) := measurable_pi_apply _
  have hψ : Measurable
      (fun v : ((i : (insert none ((Finset.range p).image some) : Finset (Option ℕ))) →
          streamSpace E G i) =>
        ((v ⟨none, hmemn⟩ : E), fun i : Fin p => (v ⟨some (i : ℕ), hmemi i⟩ : G))) :=
    (measurable_pi_apply _).prodMk
      (measurable_pi_lambda _ fun _ => measurable_pi_apply _)
  exact key.comp hφ hψ

end BridgeStream

/-! ### §11c  Assembling `IsotropicPasses` -/

section BridgeAssembly

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- **The hypotheses of the re-isotropised scheme.**  A rotation stream `R` drawn from a
measurable group `G` that acts on `E` through isometries `ι`, together with the regularity the
measure-theoretic argument needs.

Every field is a statement about the *inputs* of the algorithm — never about a residual law, and
never about a per-pass expectation.  That is the whole point: `IsotropicPasses.indep` and
`IsotropicPasses.distortion`, which are statements about residuals, become theorems. -/
structure RotationStream (ι : G → (E ≃ₗᵢ[ℝ] E)) (q : E → E) (W : Ω → E) (R : Ω → ℕ → G)
    (κ : Measure G) (x₀ : E) (μ : Measure Ω) : Prop where
  /-- `ι` implements the `G`-action: this is what makes `G` "a group of rotations". -/
  implements : ∀ (g : G) (x : E), ι g x = g • x
  /-- the quantiser is measurable. -/
  meas_q : Measurable q
  /-- the weight row is measurable. -/
  meas_W : Measurable W
  /-- every rotation is measurable. -/
  meas_R : ∀ p, Measurable (fun ω => R ω p)
  /-- **every pass draws from the same law `κ`** — one Haar draw per pass. -/
  law_R : ∀ p, μ.map (fun ω => R ω p) = κ
  /-- **the pass-`p` rotation is fresh**: independent of the weight row and of all earlier
  rotations, which by `resid_rotScheme_eq_residPast` is everything `r_p` can depend on.
  `indepFun_rot_past_of_iIndepFun` derives this from mutual independence of `(W, R_0, R_1, …)`. -/
  fresh : ∀ p, IndepFun (fun ω => R ω p) (fun ω => (W ω, fun i : Fin p => R ω i)) μ
  /-- no pass is exact, almost surely.  §5's orbit hypothesis is stated `∀ᵐ ω ∂μ`, which is the
  honest measure-theoretic form and lets this field be a.e.\ rather than pointwise. -/
  exact_never : ∀ p, ∀ᵐ ω ∂μ, resid (rotScheme ι q (R ω)) (W ω) p ≠ 0
  /-- `G` is transitive on the unit sphere, with base point `x₀`.  For `G = O(d)` this is
  `orbit_eq_sphere` of §5b; for a finite subgroup it is false, and the conclusions below then
  genuinely do not apply (cf. header item 4). -/
  transitive : ∀ x : E, ‖x‖ = 1 → x ∈ MulAction.orbit G x₀

variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {q : E → E} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G} {x₀ : E}

omit [BorelSpace E] [SecondCountableTopology E] [MeasurableSMul₂ G E] [MeasurableInv G] in
private theorem aux_meas_past (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    Measurable (fun ω => (W ω, fun i : Fin p => R ω i)) :=
  h.meas_W.prodMk (measurable_pi_lambda _ fun i => h.meas_R i)

private theorem aux_meas_dir (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    Measurable (fun ω => residDirPast q p (W ω, fun i : Fin p => R ω i)) :=
  (measurable_residDirPast h.meas_q p).comp (aux_meas_past h p)

private theorem aux_meas_unitVec (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    Measurable (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p) := by
  simp only [unitVec_rotScheme_eq_smul h.implements]
  exact (h.meas_R p).smul (aux_meas_dir h p)

omit [BorelSpace E] [SecondCountableTopology E] [MeasurableSMul₂ G E] [MeasurableInv G] in
private theorem aux_orbit (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    ∀ᵐ ω ∂μ, residDirPast q p (W ω, fun i : Fin p => R ω i) ∈ MulAction.orbit G x₀ := by
  filter_upwards [h.exact_never p] with ω hω
  exact h.transitive _ (norm_residDirPast h.implements hω)

omit [BorelSpace E] [SecondCountableTopology E] [MeasurableSMul₂ G E] [MeasurableInv G] in
private theorem aux_isProb (h : RotationStream ι q W R κ x₀ μ) [IsProbabilityMeasure μ] :
    IsProbabilityMeasure κ := by
  rw [← h.law_R 0]
  exact Measure.isProbabilityMeasure_map (h.meas_R 0).aemeasurable

/-- The fresh rotation is independent of the pair *(normalised residual, squared scale)* — the
exact hypothesis of `indepFun_smul_of_indepPair`.  This is `RotationStream.fresh` pushed through
the measurable map `resid_rotScheme_eq_residPast` supplies. -/
theorem indepFun_rot_dirScale (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    IndepFun (fun ω => R ω p)
      (fun ω => (residDirPast q p (W ω, fun i : Fin p => R ω i),
        ‖resid (rotScheme ι q (R ω)) (W ω) p‖ ^ 2)) μ := by
  have hΨ : Measurable (fun x : E × (Fin p → G) =>
      (residDirPast (G := G) q p x, ‖residPast (G := G) q p x‖ ^ 2)) :=
    (measurable_residDirPast h.meas_q p).prodMk
      ((measurable_residPast h.meas_q p).norm.pow_const 2)
  have hc := (h.fresh p).comp (measurable_id (α := G)) hΨ
  simpa [Function.comp_def, resid_rotScheme_eq_residPast h.implements] using hc

private theorem aux_meas_scale (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    Measurable (fun ω => ‖resid (rotScheme ι q (R ω)) (W ω) p‖ ^ 2) := by
  simp only [resid_rotScheme_eq_residPast h.implements]
  exact ((measurable_residPast h.meas_q p).comp (aux_meas_past h p)).norm.pow_const 2

variable [MeasurableMul G] [IsProbabilityMeasure μ] [κ.IsMulRightInvariant]

/-- **Re-isotropisation, the statement the paper is really making.**  The unit direction `u_p`
has *the same law* `orbitLaw κ x₀` at every pass `p` — no matter what the residual `r_p` looks
like, and no matter what the quantiser `q` has done to it in the previous `p` passes.

This is the fact `IsotropicPasses.distortion` was assuming.  It is the payoff of §5's
`law_smul_eq_orbitLaw` combined with `resid_rotScheme_eq_residPast`: the residual direction is a
function of the past, the fresh rotation is independent of the past, and the orbit law is
constant along the orbit. -/
theorem law_unitVec_eq_orbitLaw (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    μ.map (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p) = orbitLaw κ x₀ := by
  haveI := aux_isProb h
  have hindep : IndepFun (fun ω => R ω p)
      (fun ω => residDirPast q p (W ω, fun i : Fin p => R ω i)) μ := by
    simpa [Function.comp_def] using
      (indepFun_rot_dirScale h p).comp (measurable_id (α := G)) measurable_fst
  have hkey := law_smul_eq_orbitLaw (R := fun ω => R ω p)
      (r := fun ω => residDirPast q p (W ω, fun i : Fin p => R ω i))
      (h.meas_R p) (aux_meas_dir h p) (h.law_R p) hindep (aux_orbit h p)
  simpa only [unitVec_rotScheme_eq_smul h.implements] using hkey

/-- **`IsotropicPasses.distortion`, derived.**  The expected squared normalised quantisation
error is the *same number* at every pass, namely the stage distortion of `q` against the
direction law.  The paper asserts this; here it is a consequence of `law_unitVec_eq_orbitLaw`. -/
theorem integral_stageErr_eq_stageDistortion (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    ∫ ω, ‖unitVec (rotScheme ι q (R ω)) (W ω) p
        - q (unitVec (rotScheme ι q (R ω)) (W ω) p)‖ ^ 2 ∂μ
      = stageDistortion q (orbitLaw κ x₀) := by
  have hasm : AEStronglyMeasurable (fun u : E => ‖u - q u‖ ^ 2)
      (μ.map (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p)) :=
    ((measurable_id.sub h.meas_q).norm.pow_const 2).aestronglyMeasurable
  rw [stageDistortion, ← law_unitVec_eq_orbitLaw h p,
    integral_map (aux_meas_unitVec h p).aemeasurable hasm]

/-- **`IsotropicPasses.integrable_err`, derived** from finiteness of the stage distortion. -/
theorem integrable_stageErr (h : RotationStream ι q W R κ x₀ μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀)) (p : ℕ) :
    Integrable (fun ω => ‖unitVec (rotScheme ι q (R ω)) (W ω) p
        - q (unitVec (rotScheme ι q (R ω)) (W ω) p)‖ ^ 2) μ := by
  rw [← law_unitVec_eq_orbitLaw h p] at hint
  have hasm : AEStronglyMeasurable (fun u : E => ‖u - q u‖ ^ 2)
      (μ.map (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p)) :=
    ((measurable_id.sub h.meas_q).norm.pow_const 2).aestronglyMeasurable
  exact (integrable_map_measure hasm (aux_meas_unitVec h p).aemeasurable).mp hint

/-- **`IsotropicPasses.indep`, derived.**  The squared scale `‖r_p‖²` — a function of the past —
is independent of the squared normalised quantisation error at pass `p`, because the latter is a
function of `R_p • r̂_p` and `R_p` is fresh.  This is the inductive discharge across all passes
that the header records as missing. -/
theorem indepFun_scale_stageErr (h : RotationStream ι q W R κ x₀ μ) (p : ℕ) :
    IndepFun (fun ω => ‖resid (rotScheme ι q (R ω)) (W ω) p‖ ^ 2)
      (fun ω => ‖unitVec (rotScheme ι q (R ω)) (W ω) p
        - q (unitVec (rotScheme ι q (R ω)) (W ω) p)‖ ^ 2) μ := by
  haveI := aux_isProb h
  have key := indepFun_smul_of_indepPair (R := fun ω => R ω p)
    (r := fun ω => residDirPast q p (W ω, fun i : Fin p => R ω i))
    (Z := fun ω => ‖resid (rotScheme ι q (R ω)) (W ω) p‖ ^ 2)
    (h.meas_R p) (aux_meas_dir h p) (aux_meas_scale h p) (h.law_R p)
    (indepFun_rot_dirScale h p) (aux_orbit h p)
  have hmeasφ : Measurable (fun u : E => ‖u - q u‖ ^ 2) :=
    (measurable_id.sub h.meas_q).norm.pow_const 2
  have hrw : ∀ ω, R ω p • residDirPast q p (W ω, fun i : Fin p => R ω i)
      = unitVec (rotScheme ι q (R ω)) (W ω) p :=
    fun ω => (unitVec_rotScheme_eq_smul h.implements q (W ω) (R ω) p).symm
  simp only [hrw] at key
  exact (key.comp hmeasφ (measurable_id (α := ℝ))).symm

/-- **`IsotropicPasses.integrable`, derived** by induction along the passes, using
`IndepFun.integrable_mul` at each step. -/
theorem integrable_normSq_resid (h : RotationStream ι q W R κ x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀)) (p : ℕ) :
    Integrable (fun ω => ‖resid (rotScheme ι q (R ω)) (W ω) p‖ ^ 2) μ := by
  induction p with
  | zero => simpa using hW
  | succ p ih =>
      have hmul := (indepFun_scale_stageErr h p).integrable_mul ih (integrable_stageErr h hint p)
      have heq : (fun ω => ‖resid (rotScheme ι q (R ω)) (W ω) (p + 1)‖ ^ 2)
          = (fun ω => ‖resid (rotScheme ι q (R ω)) (W ω) p‖ ^ 2) *
            (fun ω => ‖unitVec (rotScheme ι q (R ω)) (W ω) p
              - q (unitVec (rotScheme ι q (R ω)) (W ω) p)‖ ^ 2) := by
        funext ω; exact norm_resid_succ_sq _ _ p
      rw [heq]; exact hmul

/-- **The bridge.**  A rotation stream satisfying `RotationStream` — hypotheses about the
*inputs* only — supplies `IsotropicPasses` with `ρ² = stageDistortion q (orbitLaw κ x₀)`.  All
four fields, including the two substantive ones, are derived. -/
theorem isotropicPasses_of_rotationStream (h : RotationStream ι q W R κ x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀)) :
    IsotropicPasses (fun ω => rotScheme ι q (R ω)) W
      (Real.sqrt (stageDistortion q (orbitLaw κ x₀))) μ where
  integrable p := integrable_normSq_resid h hW hint p
  integrable_err p := integrable_stageErr h hint p
  indep p := indepFun_scale_stageErr h p
  distortion p := by
    rw [Real.sq_sqrt (stageDistortion_nonneg q (orbitLaw κ x₀))]
    exact integral_stageErr_eq_stageDistortion h p

/-- **Proposition 2, unconditionally**, for a re-isotropised progressive quantiser: the geometric
decay law with the constant *identified* as the stage distortion of the codebook against the
direction law, rather than assumed to exist. -/
theorem integral_norm_resid_sq_rotationStream (h : RotationStream ι q W R κ x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι q (R ω)) (W ω) P‖ ^ 2 ∂μ
      = stageDistortion q (orbitLaw κ x₀) ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  have hkey := integral_norm_resid_sq (isotropicPasses_of_rotationStream h hW hint) P
  rwa [pow_mul, Real.sq_sqrt (stageDistortion_nonneg q (orbitLaw κ x₀))] at hkey

/-- The direction law is *the same at every pair of passes* — the citable form of
"re-isotropisation restores isotropy at every stage". -/
theorem law_unitVec_eq_law_unitVec (h : RotationStream ι q W R κ x₀ μ) (p p' : ℕ) :
    μ.map (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p)
      = μ.map (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p') := by
  rw [law_unitVec_eq_orbitLaw h p, law_unitVec_eq_orbitLaw h p']

/-- **Eq. (2.5), unconditionally**, for a deterministic weight row: the RMS relative error after
`P` passes is exactly `ρ^P` with `ρ = √(stage distortion)`. -/
theorem rms_relative_error_rotationStream {w : E} (hw : w ≠ 0)
    (h : RotationStream ι q (fun _ => w) R κ x₀ μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) :
    Real.sqrt (∫ ω, ‖resid (rotScheme ι q (R ω)) w P‖ ^ 2 ∂μ) / ‖w‖
      = Real.sqrt (stageDistortion q (orbitLaw κ x₀)) ^ P :=
  rms_relative_error hw (Real.sqrt_nonneg _)
    (isotropicPasses_of_rotationStream h (integrable_const _) hint) P

end BridgeAssembly
/-! ## §16  Corollary 1 for every codebook

Corollary 1 (§6) is stated for scalar codebooks applied coordinate-wise, and §9 records that its
optimality is optimality *within a class*.  This section removes the word *scalar*.  A *codebook*
is any set `C ⊆ E` of reproduction points, and a stage quantiser *implements* `C` if it always
returns a nearest codeword.  The scalar Lloyd–Max stage (`C` a product of `2^b`-point sets),
QuIP#'s E8P stage (`C` a finite subset of the `E₈` lattice, one copy per block of eight) and
QTIP's trellis stage (`C` the codeword set of the trellis code, which the Viterbi search rounds
to exactly) are all of this form; §17 checks the per-block case.

The objective is `codebookDistortion C σ = E_σ dist(u, C)²` against the direction law
`σ = orbitLaw κ x₀`, and §11's `law_unitVec_eq_orbitLaw` makes `σ` the same measure at every
pass.  So whichever codebook minimises it over a class is optimal at every pass, and no per-pass
re-fitting can improve on it in expectation (§18 states this at the residual level).  Nothing
here uses coordinates, block structure, or finiteness of `C`; finiteness enters only in
`exists_isNearest`, which supplies a nearest-codeword map for every finite codebook. -/

section Codebook

variable {E : Type*} [NormedAddCommGroup E] [MeasurableSpace E]
variable {C C' : Set E} {Q Q' : E → E} {σ : Measure E}

/-- `Q` implements the codebook `C`: it takes values in `C` and always picks a nearest point. -/
def IsNearest (C : Set E) (Q : E → E) : Prop :=
  ∀ x, Q x ∈ C ∧ ∀ c ∈ C, ‖x - Q x‖ ≤ ‖x - c‖

/-- The distortion of a codebook against the direction law: the expected squared distance from
a direction to its nearest codeword.  This is `ρ²` as a functional of the codebook alone. -/
noncomputable def codebookDistortion (C : Set E) (σ : Measure E) : ℝ :=
  ∫ u, Metric.infDist u C ^ 2 ∂σ

omit [MeasurableSpace E] in
/-- A nearest-codeword quantiser attains the distance to the codebook. -/
theorem norm_sub_eq_infDist (hQ : IsNearest C Q) (x : E) :
    ‖x - Q x‖ = Metric.infDist x C := by
  refine le_antisymm ?_ ?_
  · refine not_lt.mp fun hlt => ?_
    obtain ⟨c, hc, hlt'⟩ := (Metric.infDist_lt_iff ⟨Q x, (hQ x).1⟩).1 hlt
    rw [dist_eq_norm] at hlt'
    exact absurd ((hQ x).2 c hc) (not_le.mpr hlt')
  · rw [← dist_eq_norm]
    exact Metric.infDist_le_dist_of_mem (hQ x).1

/-- Hence its stage distortion is the codebook distortion. -/
theorem stageDistortion_eq_codebookDistortion (hQ : IsNearest C Q) (σ : Measure E) :
    stageDistortion Q σ = codebookDistortion C σ := by
  unfold stageDistortion codebookDistortion
  simp_rw [norm_sub_eq_infDist hQ]

/-- Any quantiser with values in `C'` errs at least the distance to `C'`, in expectation. -/
theorem codebookDistortion_le_stageDistortion (hQ' : ∀ x, Q' x ∈ C')
    (hint : Integrable (fun u => ‖u - Q' u‖ ^ 2) σ) :
    codebookDistortion C' σ ≤ stageDistortion Q' σ :=
  integral_mono_of_nonneg (Filter.Eventually.of_forall fun _ => by positivity) hint
    (Filter.Eventually.of_forall fun x => by
      show Metric.infDist x C' ^ 2 ≤ ‖x - Q' x‖ ^ 2
      rw [← dist_eq_norm]
      exact pow_le_pow_left₀ Metric.infDist_nonneg (Metric.infDist_le_dist_of_mem (hQ' x)) 2)

/-- Nearest-codeword rounding is optimal among all maps into the same codebook. -/
theorem stageDistortion_le_of_isNearest (hQ : IsNearest C Q) (hQ' : ∀ x, Q' x ∈ C)
    (hint : Integrable (fun u => ‖u - Q' u‖ ^ 2) σ) :
    stageDistortion Q σ ≤ stageDistortion Q' σ := by
  rw [stageDistortion_eq_codebookDistortion hQ]
  exact codebookDistortion_le_stageDistortion hQ' hint

/-- **Corollary 1, for every codebook class.**  If `C` minimises the codebook distortion over a
class `𝒞` of codebooks, all `2^b`-level product codebooks, all E8P copies, all trellis codes of
a given rate, or all of them together, then a nearest-codeword stage for `C` has stage
distortion no larger than any stage with values in any member of `𝒞`.  By
`stage_rho_pow_le_of_le` (§9) its `P`-pass envelope is then no larger for every `P`. -/
theorem stageDistortion_le_of_codebook_optimal {𝒞 : Set (Set E)}
    (hmin : ∀ C'' ∈ 𝒞, codebookDistortion C σ ≤ codebookDistortion C'' σ)
    (hQ : IsNearest C Q) (hC' : C' ∈ 𝒞) (hQ' : ∀ x, Q' x ∈ C')
    (hint : Integrable (fun u => ‖u - Q' u‖ ^ 2) σ) :
    stageDistortion Q σ ≤ stageDistortion Q' σ := by
  rw [stageDistortion_eq_codebookDistortion hQ]
  exact (hmin C' hC').trans (codebookDistortion_le_stageDistortion hQ' hint)

omit [MeasurableSpace E] in
/-- Every finite nonempty codebook has a nearest-codeword quantiser. -/
theorem exists_isNearest (hC : C.Finite) (hne : C.Nonempty) : ∃ Q : E → E, IsNearest C Q := by
  choose Q hQ using fun x => Set.exists_min_image C (fun c => ‖x - c‖) hC hne
  exact ⟨Q, hQ⟩

end Codebook
/-! ## §17  Block stages: the vector quadrature for `ρ`

Proposition 2 (§3, §11) never used coordinates.  What *is* tied to the scalar stage is the
evaluation of `ρ²` as `d · E_{f_d}[(t − q t)²]` (§3b).  This section is the same computation for
a stage that quantises `m` blocks of a space `F` at once: `F = ℝ` is the scalar stage,
`F = EuclideanSpace ℝ (Fin 8)` the `E₈` stage, `F = EuclideanSpace ℝ (Fin T)` a trellis stage of
length `T`.  The answer has the same shape, `ρ² = m · E_ν‖x − q_B x‖²` with `ν` the common law of
one block, and `E_ν‖x‖² = 1/m` is again forced by `‖u‖ = 1`.  Only linearity of expectation is
used: the blocks need not be independent, only identically distributed, and §17b derives that
from invariance of the direction law whenever the rotation group contains the block swaps. -/

section Block

variable {m : ℕ} {F : Type*} [NormedAddCommGroup F] [MeasurableSpace F]

/-- `m` blocks of `F` with the Euclidean norm across blocks.  `BlockSpace m ℝ` is
`EuclideanSpace ℝ (Fin m)` definitionally. -/
abbrev BlockSpace (m : ℕ) (F : Type*) := PiLp 2 (fun _ : Fin m => F)

/-- The stage that applies one block quantiser `qB` to every block. -/
def blockStage (qB : F → F) (v : BlockSpace m F) : BlockSpace m F :=
  WithLp.toLp 2 fun j => qB (v j)

omit [NormedAddCommGroup F] in
theorem measurable_block (j : Fin m) : Measurable (fun v : BlockSpace m F => v j) :=
  (measurable_pi_apply j).comp (WithLp.measurable_ofLp 2 _)

omit [NormedAddCommGroup F] in
theorem measurable_blockStage {qB : F → F} (hq : Measurable qB) :
    Measurable (blockStage (m := m) qB) :=
  (WithLp.measurable_toLp 2 _).comp (measurable_pi_lambda _ fun j => hq.comp (measurable_block j))

/-- The scalar stages of §9 are the block stages at `F = ℝ`, definitionally. -/
theorem scalarStages_eq_blockStage (d : ℕ) (C : Set (ℝ → ℝ)) :
    scalarStages d C = {Q | ∃ cb ∈ C, Q = blockStage cb} := rfl

omit [MeasurableSpace F] in
/-- The squared stage error splits over blocks, exactly and pointwise. -/
theorem norm_sub_blockStage_sq (qB : F → F) (v : BlockSpace m F) :
    ‖v - blockStage qB v‖ ^ 2 = ∑ j, ‖v j - qB (v j)‖ ^ 2 := by
  rw [PiLp.norm_sq_eq_of_L2]; simp [blockStage]

omit [MeasurableSpace F] in
/-- Nearest-codeword per block is nearest-codeword for the product codebook.  So coordinate-wise
Lloyd–Max rounding implements the product of `2^b`-point sets and per-block E8P rounding
implements the product of E8P copies, in the sense of §16. -/
theorem isNearest_blockStage {qB : F → F} {CB : Set F} (h : IsNearest CB qB) :
    IsNearest {v : BlockSpace m F | ∀ j, v j ∈ CB} (blockStage qB) := by
  intro v
  refine ⟨fun j => by simpa [blockStage] using (h (v j)).1, fun c hc => ?_⟩
  rw [← pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero,
    norm_sub_blockStage_sq, PiLp.norm_sq_eq_of_L2]
  simp only [PiLp.sub_apply]
  exact Finset.sum_le_sum fun j _ => pow_le_pow_left₀ (norm_nonneg _) ((h (v j)).2 _ (hc j)) 2

variable {σ : Measure (BlockSpace m F)} {ν : Measure F}

private theorem aux_integrable_block (qB : F → F) (hmarg : ∀ j, σ.map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) (j : Fin m) :
    Integrable (fun v : BlockSpace m F => ‖v j - qB (v j)‖ ^ 2) σ := by
  have hasm : AEStronglyMeasurable (fun x => ‖x - qB x‖ ^ 2) (σ.map (fun v => v j)) := by
    rw [hmarg j]; exact hint.aestronglyMeasurable
  exact (integrable_map_measure hasm (measurable_block j).aemeasurable).mp
    (by rw [hmarg j]; exact hint)

private theorem aux_integral_block (qB : F → F) (hmarg : ∀ j, σ.map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) (j : Fin m) :
    ∫ v, ‖v j - qB (v j)‖ ^ 2 ∂σ = ∫ x, ‖x - qB x‖ ^ 2 ∂ν := by
  have hasm : AEStronglyMeasurable (fun x => ‖x - qB x‖ ^ 2) (σ.map (fun v => v j)) := by
    rw [hmarg j]; exact hint.aestronglyMeasurable
  rw [← hmarg j, integral_map (measurable_block j).aemeasurable hasm]

/-- Integrability of the stage error, from integrability of one block's error. -/
theorem integrable_blockStage_err (qB : F → F) (hmarg : ∀ j, σ.map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) :
    Integrable (fun v : BlockSpace m F => ‖v - blockStage qB v‖ ^ 2) σ := by
  simp_rw [norm_sub_blockStage_sq]
  exact integrable_finsetSum _ fun j _ => aux_integrable_block qB hmarg hint j

/-- **The block quadrature.**  If every block of the direction has the same law `ν`, the stage
distortion of the block stage is `m · E_ν‖x − q_B x‖²`.  This is `integral_norm_sub_cb_sq` (§3b)
with the coordinate replaced by a block; no independence between blocks is used. -/
theorem stageDistortion_blockStage (qB : F → F) (hmarg : ∀ j, σ.map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) :
    stageDistortion (blockStage qB) σ = m * ∫ x, ‖x - qB x‖ ^ 2 ∂ν := by
  unfold stageDistortion
  simp_rw [norm_sub_blockStage_sq]
  rw [integral_finsetSum _ (fun j _ => aux_integrable_block qB hmarg hint j)]
  simp_rw [aux_integral_block qB hmarg hint]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- `E_ν‖x‖² = 1/m` is forced by `‖u‖ = 1` and identically distributed blocks, exactly as
`second_moment_eq_inv_card` (§3b) forces `E_{f_d}[t²] = 1/d`. -/
theorem integral_norm_sq_block [IsProbabilityMeasure σ] (hm : m ≠ 0)
    (hmarg : ∀ j, σ.map (fun v => v j) = ν) (hint : Integrable (fun x : F => ‖x‖ ^ 2) ν)
    (hunit : ∀ᵐ v ∂σ, ‖v‖ = 1) :
    ∫ x, ‖x‖ ^ 2 ∂ν = 1 / (m : ℝ) := by
  have hm' : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hm
  have hint' : Integrable (fun x : F => ‖x - (fun _ => 0) x‖ ^ 2) ν := by simpa using hint
  have h1 : ∫ v, ∑ j, ‖v j‖ ^ 2 ∂σ = 1 := by
    have hone : ∀ᵐ v ∂σ, ∑ j, ‖v j‖ ^ 2 = 1 := by
      filter_upwards [hunit] with v hv
      rw [← PiLp.norm_sq_eq_of_L2, hv, one_pow]
    rw [integral_congr_ae hone]; simp
  have hmapint := fun j => by simpa using aux_integrable_block (σ := σ) (fun _ => 0) hmarg hint' j
  have hblock := fun j => by simpa using aux_integral_block (σ := σ) (fun _ => 0) hmarg hint' j
  rw [integral_finsetSum _ (fun j _ => hmapint j)] at h1
  simp_rw [hblock] at h1
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h1
  field_simp
  linarith [h1]

/-- **Eq. (2.3) for a block stage**: `ρ² = E_ν‖x − q_B x‖² / E_ν‖x‖²`, the normalised block
distortion, with no reference to `m` or `d`. -/
theorem stageDistortion_blockStage_eq_ratio [IsProbabilityMeasure σ] (hm : m ≠ 0) (qB : F → F)
    (hmarg : ∀ j, σ.map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) (hint₂ : Integrable (fun x : F => ‖x‖ ^ 2) ν)
    (hunit : ∀ᵐ v ∂σ, ‖v‖ = 1) :
    stageDistortion (blockStage qB) σ = (∫ x, ‖x - qB x‖ ^ 2 ∂ν) / ∫ x, ‖x‖ ^ 2 ∂ν := by
  rw [stageDistortion_blockStage qB hmarg hint, integral_norm_sq_block hm hmarg hint₂ hunit]
  have hm' : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hm
  field_simp

/-- At `F = ℝ` the block quadrature is §3b's `rhoSq`: the scalar quadrature is the block
quadrature with one-dimensional blocks. -/
theorem stageDistortion_blockStage_real {σ : Measure (EuclideanSpace ℝ (Fin m))} {ν : Measure ℝ}
    (cb : ℝ → ℝ) (hmarg : ∀ j, σ.map (fun v => v j) = ν)
    (hint : Integrable (fun t => (t - cb t) ^ 2) ν) :
    stageDistortion (blockStage cb) σ = rhoSq m cb ν := by
  have hint' : Integrable (fun t : ℝ => ‖t - cb t‖ ^ 2) ν := by
    simpa [Real.norm_eq_abs, sq_abs] using hint
  rw [stageDistortion_blockStage cb hmarg hint', rhoSq]
  simp [Real.norm_eq_abs, sq_abs]

end Block

/-! ### §17b  Identically distributed blocks, from invariance of the direction law

`stageDistortion_blockStage` assumes every block has the same law.  For the paper's uniform law
that is permutation invariance; here it is derived from the rotation stream alone: the orbit law
is invariant under the group when `κ` is left-invariant (Haar on a compact group is), so if the
group contains an element that moves block `j'` to block `j`, the two block marginals agree. -/

section BlockExchange

variable {G : Type*} [Group G] [MeasurableSpace G] [MeasurableMul G]
variable {X : Type*} [MeasurableSpace X] [MulAction G X] [MeasurableSMul G X]

/-- Under a left-invariant `κ` the orbit law is `G`-invariant. -/
theorem orbitLaw_map_smul (κ : Measure G) [κ.IsMulLeftInvariant] (h : G) (x₀ : X) :
    (orbitLaw κ x₀).map (fun v => h • v) = orbitLaw κ x₀ := by
  unfold orbitLaw
  rw [Measure.map_map (measurable_const_smul h) (measurable_smul_const x₀)]
  have hcomp : (fun v : X => h • v) ∘ (fun g : G => g • x₀)
      = (fun g : G => g • x₀) ∘ (fun g : G => h * g) := by
    funext g; simp [mul_smul]
  rw [hcomp, ← Measure.map_map (measurable_smul_const x₀) (measurable_const_mul h),
    map_mul_left_eq_self]

variable {m : ℕ} {F : Type*} [NormedAddCommGroup F] [MeasurableSpace F]
  [MulAction G (BlockSpace m F)] [MeasurableSMul G (BlockSpace m F)]

omit [NormedAddCommGroup F] in
/-- If some rotation carries block `j'` to block `j`, the two block marginals of the direction
law coincide.  With all block swaps available (they are in `O(d)`), every block has the same
law, which is the `hmarg` hypothesis of `stageDistortion_blockStage`. -/
theorem blockMarginal_eq_of_swap (κ : Measure G) [κ.IsMulLeftInvariant] (x₀ : BlockSpace m F)
    {j j' : Fin m} {h : G} (hswap : ∀ v : BlockSpace m F, (h • v) j = v j') :
    (orbitLaw κ x₀).map (fun v => v j) = (orbitLaw κ x₀).map (fun v => v j') := by
  conv_lhs => rw [← orbitLaw_map_smul κ h x₀]
  rw [Measure.map_map (measurable_block j) (measurable_const_smul h)]
  congr 1
  funext v
  exact hswap v

end BlockExchange
/-! ## §18  Every pass, for every stage

§16 and §17 compare stage distortions.  This section pushes the comparison through §11 to the
quantity the paper measures, the expected squared residual after `P` passes, for any two stages
run through the same rotation stream from the same weight rows.  Nothing is assumed about the
stages beyond what `RotationStream` already asks of them (measurable, never exact), so the
statements cover scalar, `E₈` and trellis stages alike; `integral_norm_resid_sq_blockStage` is
Proposition 2 for a block stage with `ρ²` supplied by §17's block quadrature. -/

section EveryPass

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G] [MeasurableMul G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G}
  [κ.IsMulRightInvariant] {x₀ : E}

omit [BorelSpace E] [SecondCountableTopology E] [MeasurableInv G] [MeasurableMul G]
  [IsProbabilityMeasure μ] [κ.IsMulRightInvariant] in
/-- The direction law lives on the unit sphere: `G` acts by isometries and `‖x₀‖ = 1`. -/
theorem norm_eq_one_ae_orbitLaw [OpensMeasurableSpace E]
    (hι : ∀ (g : G) (x : E), ι g x = g • x) (hx₀ : ‖x₀‖ = 1) :
    ∀ᵐ v ∂(orbitLaw κ x₀), ‖v‖ = 1 := by
  unfold orbitLaw
  refine (ae_map_iff (measurable_smul_const x₀).aemeasurable (p := fun v : E => ‖v‖ = 1)
    (measurable_norm (measurableSet_singleton (1 : ℝ)))).2 ?_
  exact Filter.Eventually.of_forall fun g => by rw [← hι, LinearIsometryEquiv.norm_map, hx₀]

/-- **Corollary 1 at the residual level, for every stage.**  Of two stages driven by the same
rotation stream from the same weight rows, the one with the smaller stage distortion has the
smaller expected squared residual after every number of passes. -/
theorem integral_norm_resid_sq_le_of_stageDistortion_le {q q' : E → E}
    (h : RotationStream ι q W R κ x₀ μ) (h' : RotationStream ι q' W R κ x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀))
    (hint' : Integrable (fun u : E => ‖u - q' u‖ ^ 2) (orbitLaw κ x₀))
    (hle : stageDistortion q (orbitLaw κ x₀) ≤ stageDistortion q' (orbitLaw κ x₀)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι q (R ω)) (W ω) P‖ ^ 2 ∂μ
      ≤ ∫ ω, ‖resid (rotScheme ι q' (R ω)) (W ω) P‖ ^ 2 ∂μ := by
  rw [integral_norm_resid_sq_rotationStream h hW hint P,
    integral_norm_resid_sq_rotationStream h' hW hint' P]
  exact mul_le_mul_of_nonneg_right (pow_le_pow_left₀ (stageDistortion_nonneg _ _) hle P)
    (integral_nonneg fun _ => by positivity)

/-- **Corollary 1 for every codebook, at the residual level.**  A nearest-codeword stage for the
codebook that minimises `codebookDistortion` over a class beats every stage with values in any
member of that class, at every pass count.  The paper's Corollary 1 is the case where the class
is the `2^b`-level product codebooks; the E8P and trellis classes are the same theorem. -/
theorem integral_norm_resid_sq_le_of_codebook_optimal {𝒞 : Set (Set E)} {C C' : Set E}
    {q q' : E → E}
    (hmin : ∀ C'' ∈ 𝒞, codebookDistortion C (orbitLaw κ x₀) ≤ codebookDistortion C'' (orbitLaw κ x₀))
    (hQ : IsNearest C q) (hC' : C' ∈ 𝒞) (hQ' : ∀ x, q' x ∈ C')
    (h : RotationStream ι q W R κ x₀ μ) (h' : RotationStream ι q' W R κ x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (orbitLaw κ x₀))
    (hint' : Integrable (fun u : E => ‖u - q' u‖ ^ 2) (orbitLaw κ x₀)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι q (R ω)) (W ω) P‖ ^ 2 ∂μ
      ≤ ∫ ω, ‖resid (rotScheme ι q' (R ω)) (W ω) P‖ ^ 2 ∂μ :=
  integral_norm_resid_sq_le_of_stageDistortion_le h h' hW hint hint'
    (stageDistortion_le_of_codebook_optimal hmin hQ hC' hQ' hint') P

end EveryPass

section BlockEveryPass

variable {m : ℕ} {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
  [MeasurableSpace F] [BorelSpace F] [SecondCountableTopology F]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G (BlockSpace m F)]
  [MeasurableSMul₂ G (BlockSpace m F)] [MeasurableInv G] [MeasurableMul G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : G → (BlockSpace m F ≃ₗᵢ[ℝ] BlockSpace m F)} {W : Ω → BlockSpace m F}
  {R : Ω → ℕ → G} {κ : Measure G} [κ.IsMulRightInvariant] {x₀ : BlockSpace m F}

/-- **Proposition 2 for a block stage**, with `ρ²` evaluated by the block quadrature: for an
`E₈` stage `m = d/8` and `ν` is the law of eight coordinates of the direction, for a trellis
stage of length `T` it is `m = d/T`, and for the scalar stage `m = d` and `ν = f_d`. -/
theorem integral_norm_resid_sq_blockStage {qB : F → F} {ν : Measure F}
    (h : RotationStream ι (blockStage qB) W R κ x₀ μ)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hmarg : ∀ j, (orbitLaw κ x₀).map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι (blockStage qB) (R ω)) (W ω) P‖ ^ 2 ∂μ
      = (m * ∫ x, ‖x - qB x‖ ^ 2 ∂ν) ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  rw [integral_norm_resid_sq_rotationStream h hW (integrable_blockStage_err qB hmarg hint) P,
    stageDistortion_blockStage qB hmarg hint]

/-- The same with `ρ²` in the normalised form of eq. (2.3), `E_ν‖x − q_B x‖² / E_ν‖x‖²`; the
sphere constraint `‖u‖ = 1` is now derived from the stream rather than assumed. -/
theorem integral_norm_resid_sq_blockStage_ratio {qB : F → F} {ν : Measure F} (hm : m ≠ 0)
    (h : RotationStream ι (blockStage qB) W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hmarg : ∀ j, (orbitLaw κ x₀).map (fun v => v j) = ν)
    (hint : Integrable (fun x => ‖x - qB x‖ ^ 2) ν) (hint₂ : Integrable (fun x : F => ‖x‖ ^ 2) ν)
    (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι (blockStage qB) (R ω)) (W ω) P‖ ^ 2 ∂μ
      = ((∫ x, ‖x - qB x‖ ^ 2 ∂ν) / ∫ x, ‖x‖ ^ 2 ∂ν) ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  haveI : IsProbabilityMeasure κ := by
    rw [← h.law_R 0]; exact Measure.isProbabilityMeasure_map (h.meas_R 0).aemeasurable
  haveI : IsProbabilityMeasure (orbitLaw κ x₀) :=
    Measure.isProbabilityMeasure_map (measurable_smul_const x₀).aemeasurable
  rw [integral_norm_resid_sq_rotationStream h hW (integrable_blockStage_err qB hmarg hint) P,
    stageDistortion_blockStage_eq_ratio hm qB hmarg hint hint₂
      (norm_eq_one_ae_orbitLaw h.implements hx₀)]

end BlockEveryPass
/-! ## §19  The uniform measure on the sphere, and its polar decomposition

Header item 1 records that Lemma 1's content, that a coordinate of a uniform point on `S^{d−1}`
has density `f_d`, was absent because "Mathlib has no usable handle on the uniform measure on a
sphere".  It does: `MeasureTheory.Measure.toSphere` is the surface measure obtained from
Lebesgue measure by the cone construction, and `measurePreserving_homeomorphUnitSphereProd` is
the polar decomposition `volume ≅ toSphere ⊗ r^{d−1} dr` on `E ∖ {0}`.  §19 to §21 use it to
prove Lemma 1, in the block form the vector stages of §17 need.

This section has the tools: the polar formula for a Lebesgue integral in `lintegral` form
(`lintegral_eq_polar`), the radial constant `∫₀¹ rⁿ dr`, and the two-dimensional polar change
of variables on a quadrant, which is where the two radii of a block decomposition `E = U × V`
meet.  Everything is done in `ℝ≥0∞`, so no integrability side condition ever arises. -/

open Measure Set Metric

section Polar
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E] [Nontrivial E]

/-- Polar decomposition of a Lebesgue integral on a finite-dimensional inner product space. -/
theorem lintegral_eq_polar (F : E → ℝ≥0∞) (hF : Measurable F) :
    ∫⁻ x, F x = ∫⁻ u : sphere (0 : E) 1, (∫⁻ r in Set.Ioi (0 : ℝ),
      ENNReal.ofReal (r ^ (Module.finrank ℝ E - 1)) * F (r • (u : E))) ∂volume.toSphere := by
  have h1 : ∫⁻ x, F x = ∫⁻ x in ({0}ᶜ : Set E), F x := by rw [restrict_compl_singleton]
  have h2 : ∫⁻ x in ({0}ᶜ : Set E), F x
      = ∫⁻ x : ({0}ᶜ : Set E), F x ∂(volume.comap Subtype.val) := by
    rw [← setLIntegral_univ (μ := volume.comap Subtype.val),
      setLIntegral_subtype (measurableSet_singleton _).compl Set.univ F, Set.image_univ, Subtype.range_coe]
  have h3 := (measurePreserving_homeomorphUnitSphereProd (volume : Measure E)).lintegral_comp_emb
    (Homeomorph.measurableEmbedding _)
    (fun p : sphere (0 : E) 1 × Set.Ioi (0 : ℝ) => F ((p.2 : ℝ) • (p.1 : E)))
  have h4 : ∀ a : ({0}ᶜ : Set E),
      F (((homeomorphUnitSphereProd E a).2 : ℝ) • ((homeomorphUnitSphereProd E a).1 : E)) = F a := by
    intro a
    rw [homeomorphUnitSphereProd_apply_snd_coe, homeomorphUnitSphereProd_apply_fst_coe, smul_smul,
      mul_inv_cancel₀ (norm_ne_zero_iff.mpr (Set.mem_compl_singleton_iff.mp a.2)), one_smul]
  simp only [h4] at h3
  have hmeas : Measurable (fun p : sphere (0 : E) 1 × Set.Ioi (0 : ℝ) => F ((p.2 : ℝ) • (p.1 : E))) := by
    fun_prop
  rw [h1, h2, h3, lintegral_prod _ hmeas.aemeasurable]
  refine lintegral_congr fun u => ?_
  rw [volumeIoiPow, lintegral_withDensity_eq_lintegral_mul _ (by fun_prop) (by fun_prop)]
  have hpt : ((fun r : Set.Ioi (0 : ℝ) => ENNReal.ofReal ((r : ℝ) ^ (Module.finrank ℝ E - 1)))
      * fun y : Set.Ioi (0 : ℝ) => F (((u, y).2 : ℝ) • ((u, y).1 : E)))
      = fun r : Set.Ioi (0 : ℝ) =>
        (fun t : ℝ => ENNReal.ofReal (t ^ (Module.finrank ℝ E - 1)) * F (t • (u : E))) r := by
    funext r; rfl
  rw [hpt, ← setLIntegral_univ (μ := volume.comap Subtype.val),
    setLIntegral_subtype measurableSet_Ioi Set.univ
      (fun t : ℝ => ENNReal.ofReal (t ^ (Module.finrank ℝ E - 1)) * F (t • (u : E))),
    Set.image_univ, Subtype.range_coe]

end Polar

section Radial

/-- `∫₀¹ rⁿ dr`, in the form the polar formula produces it. -/
noncomputable def radialConst (n : ℕ) : ℝ≥0∞ :=
  ∫⁻ r in Set.Ioi (0 : ℝ), ENNReal.ofReal (r ^ n) * (Set.Iio (1 : ℝ)).indicator 1 r

theorem radialConst_eq (n : ℕ) : radialConst n = ENNReal.ofReal (1 / (n + 1)) := by
  have h1 : radialConst n = ∫⁻ r in Set.Ioo (0 : ℝ) 1, ENNReal.ofReal (r ^ n) := by
    unfold radialConst
    rw [setLIntegral_congr_fun measurableSet_Ioi (g := (Set.Iio (1 : ℝ)).indicator
        fun r => ENNReal.ofReal (r ^ n)) (fun r _ => by
          by_cases h : r < 1 <;> simp [Set.indicator, h]),
      lintegral_indicator measurableSet_Iio, Measure.restrict_restrict measurableSet_Iio,
      Set.inter_comm, Set.Ioi_inter_Iio]
  have h2 : ∫⁻ r in Set.Ioo (0 : ℝ) 1, ENNReal.ofReal (r ^ n) = ENNReal.ofReal (1 / (n + 1)) := by
    rw [← ofReal_integral_eq_lintegral_ofReal]
    · rw [← integral_Ioc_eq_integral_Ioo, ← intervalIntegral.integral_of_le zero_le_one,
        integral_pow]
      simp
    · exact (intervalIntegral.intervalIntegrable_pow n).1.mono_set Ioo_subset_Ioc_self
    · exact (ae_restrict_mem measurableSet_Ioo).mono fun r hr => pow_nonneg hr.1.le n
  rw [h1, h2]

end Radial

section Polar2D

/-- The angular weight `cos^a θ · sin^b θ`. -/
noncomputable def angWeight (a b : ℕ) (θ : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (Real.cos θ ^ a * Real.sin θ ^ b)

theorem measurable_angWeight (a b : ℕ) : Measurable (angWeight a b) := by
  unfold angWeight; fun_prop

/-- The quadrant integrand: `ρ^a s^b · g(ρ/√(ρ²+s²)) · 1[ρ²+s²<1]`. -/
noncomputable def quadFun (a b : ℕ) (g : ℝ → ℝ≥0∞) (p : ℝ × ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (p.1 ^ a) * (ENNReal.ofReal (p.2 ^ b) *
    (g ((√(p.1 ^ 2 + p.2 ^ 2))⁻¹ * p.1) * (Set.Iio (1 : ℝ)).indicator 1 (p.1 ^ 2 + p.2 ^ 2)))

theorem measurable_quadFun (a b : ℕ) {g : ℝ → ℝ≥0∞} (hg : Measurable g) :
    Measurable (quadFun a b g) := by
  unfold quadFun
  exact (by fun_prop : Measurable fun p : ℝ × ℝ => ENNReal.ofReal (p.1 ^ a)).mul
    ((by fun_prop : Measurable fun p : ℝ × ℝ => ENNReal.ofReal (p.2 ^ b)).mul
      ((hg.comp (by fun_prop)).mul
        ((measurable_const.indicator measurableSet_Iio).comp (by fun_prop))))

/-- The radial factor after the change of variables. -/
noncomputable def radFun (n : ℕ) (R : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (R ^ n) * (Set.Iio (1 : ℝ)).indicator 1 R

theorem measurable_radFun (n : ℕ) : Measurable (radFun n) :=
  (by fun_prop : Measurable fun R : ℝ => ENNReal.ofReal (R ^ n)).mul
    (measurable_const.indicator measurableSet_Iio)

/-- The angular factor, cut off to the first quadrant. -/
noncomputable def angFun (a b : ℕ) (g : ℝ → ℝ≥0∞) (θ : ℝ) : ℝ≥0∞ :=
  (Set.Ioo 0 (Real.pi / 2)).indicator (fun θ => angWeight a b θ * g (Real.cos θ)) θ

theorem measurable_angFun (a b : ℕ) {g : ℝ → ℝ≥0∞} (hg : Measurable g) :
    Measurable (angFun a b g) :=
  ((measurable_angWeight a b).mul (hg.comp Real.measurable_cos)).indicator measurableSet_Ioo

/-- **Pointwise polar identity in the first quadrant.** -/
theorem quadFun_polar (a b : ℕ) (g : ℝ → ℝ≥0∞) {R θ : ℝ} (hR : 0 < R)
    (hθ : θ ∈ Set.Ioo (-Real.pi) Real.pi) :
    ENNReal.ofReal R * (Set.Ioi (0 : ℝ) ×ˢ Set.Ioi (0 : ℝ)).indicator (quadFun a b g)
        (R * Real.cos θ, R * Real.sin θ)
      = radFun (a + b + 1) R * angFun a b g θ := by
  by_cases hq : θ ∈ Set.Ioo 0 (Real.pi / 2)
  · have hcos : 0 < Real.cos θ := Real.cos_pos_of_mem_Ioo ⟨by linarith [hq.1, Real.pi_pos], hq.2⟩
    have hsin : 0 < Real.sin θ :=
      Real.sin_pos_of_pos_of_lt_pi hq.1 (by linarith [hq.2, Real.pi_pos])
    have hmem : (R * Real.cos θ, R * Real.sin θ) ∈ Set.Ioi (0 : ℝ) ×ˢ Set.Ioi (0 : ℝ) :=
      ⟨mul_pos hR hcos, mul_pos hR hsin⟩
    rw [Set.indicator_of_mem hmem, angFun, Set.indicator_of_mem hq, quadFun, radFun, angWeight]
    simp only
    have hsq : (R * Real.cos θ) ^ 2 + (R * Real.sin θ) ^ 2 = R ^ 2 := by
      have := Real.cos_sq_add_sin_sq θ
      nlinarith [this]
    have hsqrt : √((R * Real.cos θ) ^ 2 + (R * Real.sin θ) ^ 2) = R := by
      rw [hsq, Real.sqrt_sq hR.le]
    have hind : (Set.Iio (1 : ℝ)).indicator (1 : ℝ → ℝ≥0∞) ((R * Real.cos θ) ^ 2 + (R * Real.sin θ) ^ 2)
        = (Set.Iio (1 : ℝ)).indicator (1 : ℝ → ℝ≥0∞) R := by
      rw [hsq]
      by_cases h1 : R < 1
      · rw [Set.indicator_of_mem (Set.mem_Iio.mpr (by nlinarith)),
          Set.indicator_of_mem (Set.mem_Iio.mpr h1)]
        rfl
      · have h1' : 1 ≤ R := not_lt.mp h1
        rw [Set.indicator_of_notMem
            (fun h' => absurd (Set.mem_Iio.mp h') (not_lt.mpr (by nlinarith))),
          Set.indicator_of_notMem (fun h' => h1 (Set.mem_Iio.mp h'))]
    rw [hsqrt, hind, inv_mul_cancel_left₀ hR.ne', mul_pow, mul_pow,
      ENNReal.ofReal_mul (pow_nonneg hR.le _), ENNReal.ofReal_mul (pow_nonneg hR.le _),
      ENNReal.ofReal_mul (pow_nonneg hcos.le _), pow_succ, pow_add,
      ENNReal.ofReal_mul (mul_nonneg (pow_nonneg hR.le _) (pow_nonneg hR.le _)),
      ENNReal.ofReal_mul (pow_nonneg hR.le _)]
    ring
  · have hnot : (R * Real.cos θ, R * Real.sin θ) ∉ Set.Ioi (0 : ℝ) ×ˢ Set.Ioi (0 : ℝ) := by
      rintro ⟨h1, h2⟩
      simp only [Set.mem_Ioi] at h1 h2
      have hcos : 0 < Real.cos θ := pos_of_mul_pos_right h1 hR.le
      have hsin : 0 < Real.sin θ := pos_of_mul_pos_right h2 hR.le
      apply hq
      constructor
      · by_contra h
        exact absurd hsin (not_lt.mpr
          (Real.sin_nonpos_of_nonpos_of_neg_pi_le (not_lt.mp h) hθ.1.le))
      · by_contra h
        exact absurd hcos (not_lt.mpr
          (Real.cos_nonpos_of_pi_div_two_le_of_le (not_lt.mp h) (by linarith [hθ.2, Real.pi_pos])))
    rw [Set.indicator_of_notMem hnot, angFun, Set.indicator_of_notMem hq, mul_zero, mul_zero]

/-- **The 2D polar step.**  The quadrant double integral factors as the radial constant times
the angular integral over `(0, π/2)`. -/
theorem lintegral_quadFun (a b : ℕ) {g : ℝ → ℝ≥0∞} (hg : Measurable g) :
    ∫⁻ ρ in Set.Ioi (0 : ℝ), ∫⁻ s in Set.Ioi (0 : ℝ), quadFun a b g (ρ, s)
      = radialConst (a + b + 1) *
          ∫⁻ θ in Set.Ioo 0 (Real.pi / 2), angWeight a b θ * g (Real.cos θ) := by
  have hmq := measurable_quadFun a b hg
  have h1 : ∫⁻ ρ in Set.Ioi (0 : ℝ), ∫⁻ s in Set.Ioi (0 : ℝ), quadFun a b g (ρ, s)
      = ∫⁻ p, (Set.Ioi (0 : ℝ) ×ˢ Set.Ioi (0 : ℝ)).indicator (quadFun a b g) p := by
    rw [← lintegral_prod _ hmq.aemeasurable, Measure.prod_restrict,
      lintegral_indicator (measurableSet_Ioi.prod measurableSet_Ioi), volume_eq_prod]
  have h2 : ∫⁻ p, (Set.Ioi (0 : ℝ) ×ˢ Set.Ioi (0 : ℝ)).indicator (quadFun a b g) p
      = ∫⁻ p in polarCoord.target, radFun (a + b + 1) p.1 * angFun a b g p.2 := by
    rw [← lintegral_comp_polarCoord_symm, polarCoord_target]
    refine setLIntegral_congr_fun (measurableSet_Ioi.prod measurableSet_Ioo) fun p hp => ?_
    rw [polarCoord_symm_apply, smul_eq_mul]
    exact quadFun_polar a b g hp.1 hp.2
  have h3 : ∫⁻ p in polarCoord.target, radFun (a + b + 1) p.1 * angFun a b g p.2
      = radialConst (a + b + 1) * ∫⁻ θ in Set.Ioo (-Real.pi) Real.pi, angFun a b g θ := by
    rw [polarCoord_target, volume_eq_prod, ← Measure.prod_restrict,
      lintegral_prod_mul (measurable_radFun _).aemeasurable (measurable_angFun a b hg).aemeasurable]
    rfl
  have h4 : ∫⁻ θ in Set.Ioo (-Real.pi) Real.pi, angFun a b g θ
      = ∫⁻ θ in Set.Ioo 0 (Real.pi / 2), angWeight a b θ * g (Real.cos θ) := by
    unfold angFun
    rw [lintegral_indicator measurableSet_Ioo, Measure.restrict_restrict measurableSet_Ioo,
      Set.inter_eq_left.mpr (Set.Ioo_subset_Ioo (by linarith [Real.pi_pos]) (by linarith [Real.pi_pos]))]
  rw [h1, h2, h3, h4]

end Polar2D
/-! ## §20  Lemma 1, block form: the marginal of a block of coordinates

For `E = U × V` (Euclidean product) and `u` uniform on the sphere of `E`, the `U`-block of `u`
has density proportional to `(1 − ‖x‖²)^{(dim V − 2)/2}` on the unit ball of `U`.  The proof is
the classical one, made exact: integrate a test function `ψ(u_U)` cut off at the unit ball, once
by the polar formula on `E`, once by the product structure and the polar formulas on `U` and `V`
followed by a polar change on the two radii.  The radial factors agree and cancel, leaving an
angular integral on the sphere of `U` (`lintegral_sphere_fst`); the same angular integral is what
the candidate density produces after the substitution `ρ = cos θ` (`lintegral_ball_density`).
`toSphere_map_fst` states the result as an equality of measures. -/

section Block
variable {U V : Type*} [NormedAddCommGroup U] [InnerProductSpace ℝ U] [MeasurableSpace U]
  [BorelSpace U] [FiniteDimensional ℝ U] [Nontrivial U]
  [NormedAddCommGroup V] [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V]
  [FiniteDimensional ℝ V] [Nontrivial V]

/-- The test function: `ψ` of the `U`-block of the direction, cut off at the unit ball. -/
noncomputable def testFun (ψ : U → ℝ≥0∞) (z : WithLp 2 (U × V)) : ℝ≥0∞ :=
  ψ (‖z‖⁻¹ • (WithLp.ofLp z).1) * (ball (0 : WithLp 2 (U × V)) 1).indicator 1 z

theorem measurable_testFun {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) : Measurable (testFun (V := V) ψ) := by
  unfold testFun
  exact (hψ.comp (by fun_prop)).mul (measurable_const.indicator measurableSet_ball)

theorem testFun_smul_sphere (ψ : U → ℝ≥0∞) (u : sphere (0 : WithLp 2 (U × V)) 1) {r : ℝ}
    (hr : 0 < r) :
    testFun ψ (r • (u : WithLp 2 (U × V))) = ψ (WithLp.ofLp (u : WithLp 2 (U × V))).1
      * (Set.Iio (1 : ℝ)).indicator 1 r := by
  have hu : ‖(u : WithLp 2 (U × V))‖ = 1 := mem_sphere_zero_iff_norm.mp u.2
  have hn : ‖r • (u : WithLp 2 (U × V))‖ = r := by
    rw [norm_smul, hu, mul_one, Real.norm_eq_abs, abs_of_pos hr]
  unfold testFun
  rw [hn]
  congr 1
  · rw [WithLp.ofLp_smul, Prod.smul_fst, inv_smul_smul₀ hr.ne']
  · by_cases h : r < 1
    · rw [Set.indicator_of_mem (by rwa [mem_ball_zero_iff, hn]),
        Set.indicator_of_mem (Set.mem_Iio.mpr h)]
      rfl
    · rw [Set.indicator_of_notMem (by rwa [mem_ball_zero_iff, hn]),
        Set.indicator_of_notMem (fun h' => h (Set.mem_Iio.mp h'))]

/-- **Left side.**  The polar formula on `E` evaluates the test integral to the sphere
marginal times the radial constant. -/
theorem lintegral_testFun_eq_sphere {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) :
    ∫⁻ z, testFun (V := V) ψ z
      = (∫⁻ u : sphere (0 : WithLp 2 (U × V)) 1, ψ (WithLp.ofLp (u : WithLp 2 (U × V))).1
          ∂volume.toSphere) * radialConst (Module.finrank ℝ (WithLp 2 (U × V)) - 1) := by
  rw [lintegral_eq_polar _ (measurable_testFun hψ), ← lintegral_mul_const _ (by fun_prop)]
  refine lintegral_congr fun u => ?_
  have hm : Measurable (fun r : ℝ => ENNReal.ofReal
      (r ^ (Module.finrank ℝ (WithLp 2 (U × V)) - 1)) * (Set.Iio (1 : ℝ)).indicator 1 r) :=
    (by fun_prop : Measurable fun r : ℝ => ENNReal.ofReal
      (r ^ (Module.finrank ℝ (WithLp 2 (U × V)) - 1))).mul
      (measurable_const.indicator measurableSet_Iio)
  rw [radialConst, ← lintegral_const_mul _ hm]
  refine setLIntegral_congr_fun measurableSet_Ioi fun r hr => ?_
  rw [testFun_smul_sphere ψ u hr]
  ring

/-- The integrand of the `V`-slice at a fixed `x : U`, as a function of `‖y‖`. -/
noncomputable def sliceFun (ψ : U → ℝ≥0∞) (x : U) (s : ℝ) : ℝ≥0∞ :=
  ψ ((√(‖x‖ ^ 2 + s ^ 2))⁻¹ • x) * (Set.Iio (1 : ℝ)).indicator 1 (‖x‖ ^ 2 + s ^ 2)

omit [Nontrivial U] [Nontrivial V] in
theorem measurable_sliceFun_uncurry {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) :
    Measurable (fun p : U × ℝ => sliceFun ψ p.1 p.2) := by
  unfold sliceFun
  exact (hψ.comp (by fun_prop)).mul
    ((measurable_const.indicator measurableSet_Iio).comp (by fun_prop))

omit [Nontrivial U] [Nontrivial V] in
theorem testFun_toLp (ψ : U → ℝ≥0∞) (x : U) (y : V) :
    testFun ψ (WithLp.toLp 2 (x, y)) = sliceFun ψ x ‖y‖ := by
  unfold testFun sliceFun
  rw [WithLp.prod_norm_eq_of_L2]
  simp only [WithLp.fst, WithLp.snd]
  congr 1
  have hmem : WithLp.toLp 2 (x, y) ∈ ball (0 : WithLp 2 (U × V)) 1 ↔ ‖x‖ ^ 2 + ‖y‖ ^ 2 < 1 := by
    rw [mem_ball_zero_iff, WithLp.prod_norm_eq_of_L2, Real.sqrt_lt' one_pos, one_pow]
    simp only [WithLp.fst, WithLp.snd]
  by_cases h : ‖x‖ ^ 2 + ‖y‖ ^ 2 < 1
  · rw [Set.indicator_of_mem (hmem.mpr h), Set.indicator_of_mem (Set.mem_Iio.mpr h)]
    rfl
  · rw [Set.indicator_of_notMem (fun h' => h (hmem.mp h')),
      Set.indicator_of_notMem (fun h' => h (Set.mem_Iio.mp h'))]

/-- **Right side, first two steps.**  Split `E = U × V`, then apply the polar formula on `V`. -/
theorem lintegral_testFun_eq_slices {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) :
    ∫⁻ z, testFun (V := V) ψ z
      = ∫⁻ x, (∫⁻ s in Set.Ioi (0 : ℝ), ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1)) * sliceFun ψ x s)
          * volume.toSphere (Set.univ : Set (sphere (0 : V) 1)) := by
  rw [← (WithLp.volume_preserving_toLp U V).lintegral_comp (measurable_testFun hψ),
    volume_eq_prod, lintegral_prod (fun a : U × V => testFun ψ (WithLp.toLp 2 a))
      ((measurable_testFun hψ).comp (WithLp.measurable_toLp 2 (U × V))).aemeasurable]
  refine lintegral_congr fun x => ?_
  simp_rw [testFun_toLp]
  have hm : Measurable fun y : V => sliceFun ψ x ‖y‖ :=
    (measurable_sliceFun_uncurry hψ).comp (measurable_const.prodMk measurable_norm)
  rw [lintegral_eq_polar _ hm]
  have hw : ∀ w : sphere (0 : V) 1,
      (∫⁻ s in Set.Ioi (0 : ℝ), ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1))
        * sliceFun ψ x ‖s • (w : V)‖)
      = ∫⁻ s in Set.Ioi (0 : ℝ), ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1)) * sliceFun ψ x s := by
    intro w
    refine setLIntegral_congr_fun measurableSet_Ioi fun s hs => ?_
    rw [norm_smul, mem_sphere_zero_iff_norm.mp w.2, mul_one, Real.norm_eq_abs, abs_of_pos hs]
  simp_rw [hw]
  rw [lintegral_const]

omit [Nontrivial U] [Nontrivial V] in
theorem measurable_sliceIntegral {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) :
    Measurable fun x : U =>
      ∫⁻ s in Set.Ioi (0 : ℝ), ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1)) * sliceFun ψ x s :=
  Measurable.lintegral_prod_right
    ((by fun_prop : Measurable fun p : U × ℝ =>
      ENNReal.ofReal (p.2 ^ (Module.finrank ℝ V - 1))).mul (measurable_sliceFun_uncurry hψ))

/-- **Right side, third step.**  The polar formula on `U`. -/
theorem lintegral_testFun_eq_triple {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) :
    ∫⁻ z, testFun (V := V) ψ z
      = (∫⁻ v : sphere (0 : U) 1, (∫⁻ ρ in Set.Ioi (0 : ℝ),
          ENNReal.ofReal (ρ ^ (Module.finrank ℝ U - 1)) *
            ∫⁻ s in Set.Ioi (0 : ℝ), ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1))
              * sliceFun ψ (ρ • (v : U)) s) ∂volume.toSphere)
        * volume.toSphere (Set.univ : Set (sphere (0 : V) 1)) := by
  rw [lintegral_testFun_eq_slices hψ, lintegral_mul_const _ (measurable_sliceIntegral hψ),
    lintegral_eq_polar _ (measurable_sliceIntegral hψ)]

end Block

section Assemble
variable {U V : Type*} [NormedAddCommGroup U] [InnerProductSpace ℝ U] [MeasurableSpace U]
  [BorelSpace U] [FiniteDimensional ℝ U] [Nontrivial U]
  [NormedAddCommGroup V] [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V]
  [FiniteDimensional ℝ V] [Nontrivial V]

/-- The slice integrand at `ρ • v` is the quadrant integrand with `g t = ψ (t • v)`. -/
theorem sliceFun_smul_sphere (ψ : U → ℝ≥0∞) (v : sphere (0 : U) 1) {ρ : ℝ} (hρ : 0 < ρ) (s : ℝ) :
    ENNReal.ofReal (ρ ^ (Module.finrank ℝ U - 1)) *
      (ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1)) * sliceFun ψ (ρ • (v : U)) s)
      = quadFun (Module.finrank ℝ U - 1) (Module.finrank ℝ V - 1) (fun t => ψ (t • (v : U))) (ρ, s) := by
  have hv : ‖(v : U)‖ = 1 := mem_sphere_zero_iff_norm.mp v.2
  have hn : ‖ρ • (v : U)‖ = ρ := by
    rw [norm_smul, hv, mul_one, Real.norm_eq_abs, abs_of_pos hρ]
  unfold sliceFun quadFun
  simp only [hn, smul_smul]

omit [Nontrivial U] [Nontrivial V] in
theorem finrank_withLp_prod :
    Module.finrank ℝ (WithLp 2 (U × V)) = Module.finrank ℝ U + Module.finrank ℝ V := by
  rw [(WithLp.linearEquiv 2 ℝ (U × V)).finrank_eq, Module.finrank_prod]

theorem finrank_withLp_prod_sub_one :
    Module.finrank ℝ (WithLp 2 (U × V)) - 1
      = (Module.finrank ℝ U - 1) + (Module.finrank ℝ V - 1) + 1 := by
  have hU : 0 < Module.finrank ℝ U := Module.finrank_pos
  have hV : 0 < Module.finrank ℝ V := Module.finrank_pos
  rw [finrank_withLp_prod]
  omega

set_option maxHeartbeats 1000000 in
/-- **The sphere marginal in polar form.**  For the uniform measure on the sphere of
`E = U × V`, the law of the `U`-block, integrated against `ψ`, is a constant times an angular
integral on the sphere of `U`. -/
theorem lintegral_sphere_fst {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) :
    ∫⁻ u : sphere (0 : WithLp 2 (U × V)) 1, ψ (WithLp.ofLp (u : WithLp 2 (U × V))).1
        ∂volume.toSphere
      = volume.toSphere (Set.univ : Set (sphere (0 : V) 1)) *
          ∫⁻ v : sphere (0 : U) 1, (∫⁻ θ in Set.Ioo 0 (Real.pi / 2),
            angWeight (Module.finrank ℝ U - 1) (Module.finrank ℝ V - 1) θ * ψ (Real.cos θ • (v : U)))
            ∂volume.toSphere := by
  have hL := lintegral_testFun_eq_sphere (V := V) hψ
  have hR := lintegral_testFun_eq_triple (V := V) hψ
  have hinner : ∀ v : sphere (0 : U) 1,
      (∫⁻ ρ in Set.Ioi (0 : ℝ), ENNReal.ofReal (ρ ^ (Module.finrank ℝ U - 1)) *
          ∫⁻ s in Set.Ioi (0 : ℝ), ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1))
            * sliceFun ψ (ρ • (v : U)) s)
      = radialConst (Module.finrank ℝ (WithLp 2 (U × V)) - 1) *
          ∫⁻ θ in Set.Ioo 0 (Real.pi / 2),
            angWeight (Module.finrank ℝ U - 1) (Module.finrank ℝ V - 1) θ
              * ψ (Real.cos θ • (v : U)) := by
    intro v
    have hg : Measurable fun t : ℝ => ψ (t • (v : U)) := hψ.comp (by fun_prop)
    rw [finrank_withLp_prod_sub_one, ← lintegral_quadFun _ _ hg]
    refine setLIntegral_congr_fun measurableSet_Ioi fun ρ hρ => ?_
    have hm : Measurable fun s : ℝ =>
        ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1)) * sliceFun ψ (ρ • (v : U)) s := by
      exact (by fun_prop : Measurable fun s : ℝ =>
        ENNReal.ofReal (s ^ (Module.finrank ℝ V - 1))).mul
          ((measurable_sliceFun_uncurry hψ).comp (measurable_const.prodMk measurable_id))
    rw [← lintegral_const_mul _ hm]
    exact setLIntegral_congr_fun measurableSet_Ioi fun s _ => sliceFun_smul_sphere ψ v hρ s
  have hF : Measurable fun p : sphere (0 : U) 1 × ℝ =>
      angWeight (Module.finrank ℝ U - 1) (Module.finrank ℝ V - 1) p.2 * ψ (Real.cos p.2 • (p.1 : U)) :=
    ((measurable_angWeight _ _).comp measurable_snd).mul
      (hψ.comp ((Real.measurable_cos.comp measurable_snd).smul
        (measurable_subtype_coe.comp measurable_fst)))
  have hA : Measurable fun v : sphere (0 : U) 1 => ∫⁻ θ in Set.Ioo 0 (Real.pi / 2),
      angWeight (Module.finrank ℝ U - 1) (Module.finrank ℝ V - 1) θ * ψ (Real.cos θ • (v : U)) :=
    hF.lintegral_prod_right'
  have hR' : ∫⁻ z, testFun (V := V) ψ z
      = (radialConst (Module.finrank ℝ (WithLp 2 (U × V)) - 1) *
          ∫⁻ v : sphere (0 : U) 1, (∫⁻ θ in Set.Ioo 0 (Real.pi / 2),
            angWeight (Module.finrank ℝ U - 1) (Module.finrank ℝ V - 1) θ * ψ (Real.cos θ • (v : U)))
            ∂volume.toSphere)
        * volume.toSphere (Set.univ : Set (sphere (0 : V) 1)) := by
    rw [hR, ← lintegral_const_mul _ hA]
    congr 1
    exact lintegral_congr hinner
  rw [hR'] at hL
  have h0 : radialConst (Module.finrank ℝ (WithLp 2 (U × V)) - 1) ≠ 0 := by
    rw [radialConst_eq]; simp; positivity
  have htop : radialConst (Module.finrank ℝ (WithLp 2 (U × V)) - 1) ≠ ⊤ := by
    rw [radialConst_eq]; exact ENNReal.ofReal_ne_top
  have key : (∫⁻ u : sphere (0 : WithLp 2 (U × V)) 1,
        ψ (WithLp.ofLp (u : WithLp 2 (U × V))).1 ∂volume.toSphere)
        * radialConst (Module.finrank ℝ (WithLp 2 (U × V)) - 1)
      = (volume.toSphere (Set.univ : Set (sphere (0 : V) 1)) *
          ∫⁻ v : sphere (0 : U) 1, (∫⁻ θ in Set.Ioo 0 (Real.pi / 2),
            angWeight (Module.finrank ℝ U - 1) (Module.finrank ℝ V - 1) θ * ψ (Real.cos θ • (v : U)))
            ∂volume.toSphere)
        * radialConst (Module.finrank ℝ (WithLp 2 (U × V)) - 1) := by
    rw [← hL]; ring
  exact (ENNReal.mul_left_inj h0 htop).mp key

end Assemble

section Density
variable {U : Type*} [NormedAddCommGroup U] [InnerProductSpace ℝ U] [MeasurableSpace U]
  [BorelSpace U] [FiniteDimensional ℝ U] [Nontrivial U]

theorem image_cos_Ioo : Real.cos '' Set.Ioo 0 (Real.pi / 2) = Set.Ioo 0 1 := by
  ext ρ
  constructor
  · rintro ⟨θ, hθ, rfl⟩
    refine ⟨Real.cos_pos_of_mem_Ioo ⟨by linarith [hθ.1, Real.pi_pos], hθ.2⟩, ?_⟩
    have := Real.cos_lt_cos_of_nonneg_of_le_pi_div_two le_rfl hθ.2.le hθ.1
    rwa [Real.cos_zero] at this
  · intro hρ
    exact ⟨Real.arccos ρ, ⟨Real.arccos_pos.mpr hρ.2, Real.arccos_lt_pi_div_two.mpr hρ.1⟩,
      Real.cos_arccos (by linarith [hρ.1]) hρ.2.le⟩

/-- The density of the `U`-block marginal, before normalisation: `(1 − ‖x‖²)^{(m−2)/2}`. -/
noncomputable def blockDensity (m : ℕ) (x : U) : ℝ≥0∞ :=
  ENNReal.ofReal ((1 - ‖x‖ ^ 2) ^ (((m : ℝ) - 2) / 2))

theorem measurable_blockDensity (m : ℕ) : Measurable (blockDensity (U := U) m) := by
  unfold blockDensity; fun_prop

/-- `sin θ · (1 − cos²θ)^{(m−2)/2} = sin^{m−1} θ` on `(0, π/2)`. -/
theorem sin_mul_rpow (m : ℕ) (hm : 1 ≤ m) {θ : ℝ} (hθ : θ ∈ Set.Ioo 0 (Real.pi / 2)) :
    Real.sin θ * (1 - Real.cos θ ^ 2) ^ (((m : ℝ) - 2) / 2) = Real.sin θ ^ (m - 1) := by
  have hs : 0 < Real.sin θ := Real.sin_pos_of_pos_of_lt_pi hθ.1 (by linarith [hθ.2, Real.pi_pos])
  have h1 : (1 - Real.cos θ ^ 2) ^ (((m : ℝ) - 2) / 2) = Real.sin θ ^ ((m : ℝ) - 2) := by
    rw [← Real.sin_sq, ← Real.rpow_two, ← Real.rpow_mul hs.le, mul_div_cancel₀ _ two_ne_zero]
  rw [h1, show Real.sin θ * Real.sin θ ^ ((m : ℝ) - 2)
      = Real.sin θ ^ (1 : ℝ) * Real.sin θ ^ ((m : ℝ) - 2) by rw [Real.rpow_one],
    ← Real.rpow_add hs, show (1 : ℝ) + ((m : ℝ) - 2) = ((m - 1 : ℕ) : ℝ) by
      rw [Nat.cast_sub hm, Nat.cast_one]; ring,
    Real.rpow_natCast]

/-- **The density side.**  The candidate density integrated against `ψ` over the unit ball of
`U` is the same angular integral as `lintegral_sphere_fst` produces. -/
theorem lintegral_ball_density {ψ : U → ℝ≥0∞} (hψ : Measurable ψ) (m : ℕ) (hm : 1 ≤ m) :
    ∫⁻ x in ball (0 : U) 1, blockDensity m x * ψ x
      = ∫⁻ v : sphere (0 : U) 1, (∫⁻ θ in Set.Ioo 0 (Real.pi / 2),
          angWeight (Module.finrank ℝ U - 1) (m - 1) θ * ψ (Real.cos θ • (v : U)))
          ∂volume.toSphere := by
  rw [← lintegral_indicator measurableSet_ball,
    lintegral_eq_polar (fun x => (ball (0 : U) 1).indicator (fun x => blockDensity m x * ψ x) x)
      (by exact ((measurable_blockDensity m).mul hψ).indicator measurableSet_ball)]
  refine lintegral_congr fun v => ?_
  have hv : ‖(v : U)‖ = 1 := mem_sphere_zero_iff_norm.mp v.2
  have hpt : ∀ ρ ∈ Set.Ioi (0 : ℝ), ENNReal.ofReal (ρ ^ (Module.finrank ℝ U - 1)) *
        (ball (0 : U) 1).indicator (fun x => blockDensity m x * ψ x) (ρ • (v : U))
      = (Set.Iio (1 : ℝ)).indicator (fun ρ => ENNReal.ofReal (ρ ^ (Module.finrank ℝ U - 1)) *
          (ENNReal.ofReal ((1 - ρ ^ 2) ^ (((m : ℝ) - 2) / 2)) * ψ (ρ • (v : U)))) ρ := by
    intro ρ hρ
    have hn : ‖ρ • (v : U)‖ = ρ := by
      rw [norm_smul, hv, mul_one, Real.norm_eq_abs, abs_of_pos hρ]
    by_cases h1 : ρ < 1
    · rw [Set.indicator_of_mem (by rwa [mem_ball_zero_iff, hn]),
        Set.indicator_of_mem (Set.mem_Iio.mpr h1), blockDensity, hn]
    · rw [Set.indicator_of_notMem (by rwa [mem_ball_zero_iff, hn]),
        Set.indicator_of_notMem (fun h' => h1 (Set.mem_Iio.mp h')), mul_zero]
  rw [setLIntegral_congr_fun measurableSet_Ioi hpt, lintegral_indicator measurableSet_Iio,
    Measure.restrict_restrict measurableSet_Iio, Set.inter_comm, Set.Ioi_inter_Iio, ← image_cos_Ioo,
    lintegral_image_eq_lintegral_abs_deriv_mul measurableSet_Ioo
      (fun θ _ => (Real.hasDerivAt_cos θ).hasDerivWithinAt)
      (Real.injOn_cos.mono (Set.Ioo_subset_Icc_self.trans
        (Set.Icc_subset_Icc le_rfl (by linarith [Real.pi_pos]))))]
  refine setLIntegral_congr_fun measurableSet_Ioo fun θ hθ => ?_
  have hs : 0 < Real.sin θ := Real.sin_pos_of_pos_of_lt_pi hθ.1 (by linarith [hθ.2, Real.pi_pos])
  have hc : 0 ≤ Real.cos θ :=
    (Real.cos_pos_of_mem_Ioo ⟨by linarith [hθ.1, Real.pi_pos], hθ.2⟩).le
  rw [abs_neg, abs_of_pos hs, angWeight, ← sin_mul_rpow m hm hθ,
    ENNReal.ofReal_mul (pow_nonneg hc _), ENNReal.ofReal_mul hs.le]
  ring

end Density

section MeasureForm
variable {U V : Type*} [NormedAddCommGroup U] [InnerProductSpace ℝ U] [MeasurableSpace U]
  [BorelSpace U] [FiniteDimensional ℝ U] [Nontrivial U]
  [NormedAddCommGroup V] [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V]
  [FiniteDimensional ℝ V] [Nontrivial V]

theorem measurable_sphere_fst :
    Measurable fun u : sphere (0 : WithLp 2 (U × V)) 1 => (WithLp.ofLp (u : WithLp 2 (U × V))).1 :=
  measurable_fst.comp ((WithLp.measurable_ofLp 2 (U × V)).comp measurable_subtype_coe)

/-- The unnormalised density measure of the `U`-block: `(1 − ‖x‖²)^{(m−2)/2}` on the unit ball. -/
noncomputable def blockMeasure (U : Type*) [NormedAddCommGroup U] [InnerProductSpace ℝ U]
    [MeasurableSpace U] [BorelSpace U] [FiniteDimensional ℝ U] (m : ℕ) : Measure U :=
  (volume.restrict (ball (0 : U) 1)).withDensity (blockDensity m)

/-- **The block marginal of the sphere measure, unnormalised.**  The pushforward of
`volume.toSphere` on `sphere (0 : U × V) 1` along the `U`-block is `σ_V(Set.univ)` times the
density `(1 − ‖x‖²)^{(dim V − 2)/2}` on the unit ball of `U`. -/
theorem toSphere_map_fst :
    (volume.toSphere : Measure (sphere (0 : WithLp 2 (U × V)) 1)).map
        (fun u : sphere (0 : WithLp 2 (U × V)) 1 => (WithLp.ofLp (u : WithLp 2 (U × V))).1)
      = volume.toSphere (Set.univ : Set (sphere (0 : V) 1)) • blockMeasure U (Module.finrank ℝ V) := by
  ext s hs
  have hψ : Measurable (s.indicator (1 : U → ℝ≥0∞)) := measurable_const.indicator hs
  rw [Measure.map_apply measurable_sphere_fst hs, Measure.smul_apply, blockMeasure,
    withDensity_apply _ hs, Measure.restrict_restrict hs, smul_eq_mul]
  have hL : volume.toSphere ((fun u : sphere (0 : WithLp 2 (U × V)) 1 =>
        (WithLp.ofLp (u : WithLp 2 (U × V))).1) ⁻¹' s)
      = ∫⁻ u : sphere (0 : WithLp 2 (U × V)) 1,
          s.indicator 1 (WithLp.ofLp (u : WithLp 2 (U × V))).1 ∂volume.toSphere := by
    rw [← lintegral_indicator_one (measurable_sphere_fst hs)]
    refine lintegral_congr fun u => ?_
    by_cases h : (WithLp.ofLp (u : WithLp 2 (U × V))).1 ∈ s
    · rw [Set.indicator_of_mem h, Set.indicator_of_mem (show u ∈ (fun u : sphere (0 : WithLp 2 (U × V)) 1 =>
        (WithLp.ofLp (u : WithLp 2 (U × V))).1) ⁻¹' s from h)]
      rfl
    · rw [Set.indicator_of_notMem h, Set.indicator_of_notMem
        (show u ∉ (fun u : sphere (0 : WithLp 2 (U × V)) 1 =>
          (WithLp.ofLp (u : WithLp 2 (U × V))).1) ⁻¹' s from h)]
  have hR : ∫⁻ x in s ∩ ball (0 : U) 1, blockDensity (Module.finrank ℝ V) x
      = ∫⁻ x in ball (0 : U) 1, blockDensity (Module.finrank ℝ V) x * s.indicator 1 x := by
    rw [← Measure.restrict_restrict hs, ← lintegral_indicator hs]
    refine lintegral_congr fun x => ?_
    by_cases h : x ∈ s
    · rw [Set.indicator_of_mem h, Set.indicator_of_mem h]
      simp
    · rw [Set.indicator_of_notMem h, Set.indicator_of_notMem h, mul_zero]
  rw [hL, hR, lintegral_sphere_fst hψ, lintegral_ball_density hψ _ Module.finrank_pos]

end MeasureForm
/-! ## §21  Lemma 1, closed: the uniform direction law and its marginals

`sphereUniform E` is the normalised surface measure, the paper's "uniform on `S^{d−1}`".  Its
block marginal is `blockLaw U (dim V)` (`sphereUniform_map_fst`), which at `U = ℝ` is exactly
the paper's `sphericalCoordLaw d` of §4b (`blockLaw_real`); the constant in `f_d` is forced by
both sides being probability measures, so §4's Beta integral is what identifies it.  The sphere
measure is natural under linear isometries (`toSphere_map_sphereMap`), which transports the
statement to any unit vector `e` of any finite-dimensional inner product space of dimension at
least two (`sphereUniform_map_inner`), and to any orthogonal decomposition `E ≃ U × V`
(`sphereUniform_map_block`).  `sphereUniformAmbient E` is the same law as a measure on `E`, the
`σ` of §9 and §17; `sphereUniformAmbient_map_block` and `sphereUniformAmbient_map_inner` are the
`hmarg` hypotheses of §17 and §3b, discharged.

What this does not touch is header item 2: that the *rotated* direction `R r/‖R r‖` has this law
still needs Haar measure on `O(d)`. -/

section Normalise

variable (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E]

/-- The uniform probability measure on the unit sphere of `E`: normalised surface measure. -/
noncomputable def sphereUniform : Measure (sphere (0 : E) 1) :=
  (volume.toSphere (Set.univ : Set (sphere (0 : E) 1)))⁻¹ • volume.toSphere

theorem toSphere_univ_ne_zero [Nontrivial E] :
    volume.toSphere (Set.univ : Set (sphere (0 : E) 1)) ≠ 0 := by
  rw [Measure.toSphere_apply_univ]
  refine mul_ne_zero ?_ (measure_ball_pos volume (0 : E) one_pos).ne'
  exact_mod_cast Module.finrank_pos.ne'

instance sphereUniform_isProbabilityMeasure [Nontrivial E] :
    IsProbabilityMeasure (sphereUniform E) := by
  constructor
  rw [sphereUniform, Measure.smul_apply, smul_eq_mul,
    ENNReal.inv_mul_cancel (toSphere_univ_ne_zero E) (measure_ne_top _ _)]

variable {E}

/-- The law of a `U`-block of a uniform direction: the density `(1 − ‖x‖²)^{(m−2)/2}` on the
unit ball, normalised. -/
noncomputable def blockLaw (U : Type*) [NormedAddCommGroup U] [InnerProductSpace ℝ U]
    [MeasurableSpace U] [BorelSpace U] [FiniteDimensional ℝ U] (m : ℕ) : Measure U :=
  (blockMeasure U m Set.univ)⁻¹ • blockMeasure U m

variable {U V : Type*} [NormedAddCommGroup U] [InnerProductSpace ℝ U] [MeasurableSpace U]
  [BorelSpace U] [FiniteDimensional ℝ U] [Nontrivial U]
  [NormedAddCommGroup V] [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V]
  [FiniteDimensional ℝ V] [Nontrivial V]

/-- **Lemma 1, block form.**  Under the uniform law on the sphere of `U × V`, the `U`-block has
law `blockLaw U (dim V)`, i.e. density proportional to `(1 − ‖x‖²)^{(dim V − 2)/2}` on the unit
ball of `U`. -/
theorem sphereUniform_map_fst :
    (sphereUniform (WithLp 2 (U × V))).map
        (fun u : sphere (0 : WithLp 2 (U × V)) 1 => (WithLp.ofLp (u : WithLp 2 (U × V))).1)
      = blockLaw U (Module.finrank ℝ V) := by
  have h1 : ((sphereUniform (WithLp 2 (U × V))).map
      (fun u : sphere (0 : WithLp 2 (U × V)) 1 => (WithLp.ofLp (u : WithLp 2 (U × V))).1)) Set.univ
      = 1 := by
    rw [Measure.map_apply measurable_sphere_fst MeasurableSet.univ, Set.preimage_univ,
      measure_univ]
  rw [sphereUniform, Measure.map_smul, toSphere_map_fst, smul_smul] at h1 ⊢
  rw [Measure.smul_apply, smul_eq_mul] at h1
  rw [blockLaw, ENNReal.eq_inv_of_mul_eq_one_left h1]

/-! ### The scalar case `U = ℝ`: the paper's `f_d` -/

theorem blockDensity_real (m : ℕ) (t : ℝ) :
    blockDensity m t = ENNReal.ofReal ((1 - t ^ 2) ^ (((m : ℝ) - 2) / 2)) := by
  rw [blockDensity, Real.norm_eq_abs, sq_abs]

/-- `f_{m+1}` is a positive constant times the block density at `U = ℝ`. -/
theorem sphericalCoordLaw_eq_smul_blockMeasure (m : ℕ) (hm : 1 ≤ m) :
    sphericalCoordLaw (m + 1)
      = ENNReal.ofReal (Real.Gamma (((m + 1 : ℕ) : ℝ) / 2) /
          (Real.sqrt Real.pi * Real.Gamma ((((m + 1 : ℕ) : ℝ) - 1) / 2))) • blockMeasure ℝ m := by
  rw [sphericalCoordLaw, blockMeasure, Real.ball_eq_Ioo, zero_sub, zero_add,
    ← Measure.restrict_congr_set Ioo_ae_eq_Icc, ← withDensity_smul _ (measurable_blockDensity m)]
  congr 1
  funext t
  have hm' : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hc : 0 ≤ Real.Gamma (((m + 1 : ℕ) : ℝ) / 2) /
      (Real.sqrt Real.pi * Real.Gamma ((((m + 1 : ℕ) : ℝ) - 1) / 2)) :=
    div_nonneg (Real.Gamma_pos_of_pos (by positivity)).le
      (mul_nonneg (Real.sqrt_nonneg _) (Real.Gamma_pos_of_pos (by push_cast; linarith)).le)
  rw [Pi.smul_apply, smul_eq_mul, blockDensity_real, sphericalPdf,
    ← ENNReal.ofReal_mul hc]
  congr 2
  push_cast
  ring_nf

/-- **Lemma 1, scalar form**: the normalised block density at `U = ℝ`, `V` of dimension `d − 1`,
is the paper's coordinate law `sphericalCoordLaw d`. -/
theorem blockLaw_real (m : ℕ) (hm : 1 ≤ m) :
    blockLaw ℝ m = sphericalCoordLaw (m + 1) := by
  have hprob := isProbabilityMeasure_sphericalCoordLaw (d := m + 1) (by omega)
  have h1 := hprob.measure_univ
  rw [sphericalCoordLaw_eq_smul_blockMeasure m hm] at h1 ⊢
  rw [Measure.smul_apply, smul_eq_mul] at h1
  rw [blockLaw, ENNReal.eq_inv_of_mul_eq_one_left h1]

end Normalise

section Naturality
open scoped Pointwise

variable {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E]
  [NormedAddCommGroup F] [InnerProductSpace ℝ F] [MeasurableSpace F] [BorelSpace F]
  [FiniteDimensional ℝ F]

/-- A linear isometry equivalence restricted to the unit spheres. -/
def sphereMap (L : E ≃ₗᵢ[ℝ] F) (u : sphere (0 : E) 1) : sphere (0 : F) 1 :=
  ⟨L u, by rw [mem_sphere_zero_iff_norm, L.norm_map]; exact mem_sphere_zero_iff_norm.mp u.2⟩

@[simp] theorem sphereMap_coe (L : E ≃ₗᵢ[ℝ] F) (u : sphere (0 : E) 1) :
    ((sphereMap L u : sphere (0 : F) 1) : F) = L u := rfl

theorem continuous_sphereMap (L : E ≃ₗᵢ[ℝ] F) : Continuous (sphereMap L) :=
  (L.continuous.comp continuous_subtype_val).subtype_mk _

theorem measurable_sphereMap (L : E ≃ₗᵢ[ℝ] F) : Measurable (sphereMap L) :=
  (continuous_sphereMap L).measurable

/-- The cone over `s` pulls back along `L` to the cone over the pullback of `s`. -/
theorem preimage_cone (L : E ≃ₗᵢ[ℝ] F) (s : Set (sphere (0 : F) 1)) :
    ⇑L ⁻¹' (Set.Ioo (0 : ℝ) 1 • (Subtype.val '' s))
      = Set.Ioo (0 : ℝ) 1 • (Subtype.val '' (sphereMap L ⁻¹' s)) := by
  ext x
  simp only [Set.mem_preimage, Set.mem_smul, Set.mem_image]
  constructor
  · rintro ⟨r, hr, y, ⟨v, hv, rfl⟩, hx⟩
    refine ⟨r, hr, L.symm v, ⟨sphereMap L.symm v, ?_, rfl⟩, ?_⟩
    · show sphereMap L (sphereMap L.symm v) ∈ s
      convert hv using 1
      ext
      simp
    · apply L.injective
      rw [← hx, map_smul, L.apply_symm_apply]
  · rintro ⟨r, hr, y, ⟨w, hw, rfl⟩, rfl⟩
    exact ⟨r, hr, L w, ⟨sphereMap L w, hw, rfl⟩, by rw [map_smul]⟩

/-- **The sphere measure is natural under linear isometries.** -/
theorem toSphere_map_sphereMap (L : E ≃ₗᵢ[ℝ] F) :
    (volume.toSphere : Measure (sphere (0 : E) 1)).map (sphereMap L)
      = (volume.toSphere : Measure (sphere (0 : F) 1)) := by
  ext s hs
  rw [Measure.map_apply (measurable_sphereMap L) hs,
    Measure.toSphere_apply' _ (measurable_sphereMap L hs), Measure.toSphere_apply' _ hs,
    ← preimage_cone, L.measurePreserving.measure_preimage_emb L.toHomeomorph.measurableEmbedding,
    L.toLinearEquiv.finrank_eq]

theorem toSphere_univ_eq (L : E ≃ₗᵢ[ℝ] F) :
    volume.toSphere (Set.univ : Set (sphere (0 : E) 1)) = volume.toSphere (Set.univ : Set (sphere (0 : F) 1)) := by
  rw [← toSphere_map_sphereMap L, Measure.map_apply (measurable_sphereMap L) MeasurableSet.univ,
    Set.preimage_univ]

/-- The uniform sphere law is natural under linear isometries. -/
theorem sphereUniform_map_sphereMap (L : E ≃ₗᵢ[ℝ] F) :
    (sphereUniform E).map (sphereMap L) = sphereUniform F := by
  rw [sphereUniform, sphereUniform, Measure.map_smul, toSphere_map_sphereMap, toSphere_univ_eq L]

end Naturality

section Coordinate

open scoped InnerProductSpace

/-- The model space `ℝ × ℝⁿ` with the Euclidean norm. -/
abbrev modelSpace (n : ℕ) := WithLp 2 (ℝ × EuclideanSpace ℝ (Fin n))

theorem finrank_modelSpace (n : ℕ) : Module.finrank ℝ (modelSpace n) = n + 1 := by
  rw [finrank_withLp_prod, Module.finrank_self, finrank_euclideanSpace_fin]
  ring

/-- The distinguished unit vector `(1, 0)` of the model space. -/
noncomputable def modelVec (n : ℕ) : modelSpace n := WithLp.toLp 2 (1, 0)

theorem norm_modelVec (n : ℕ) : ‖modelVec n‖ = 1 := by
  rw [modelVec, WithLp.norm_toLp_fst, norm_one]

theorem inner_modelVec (n : ℕ) (x : modelSpace n) : ⟪modelVec n, x⟫_ℝ = (WithLp.ofLp x).1 := by
  rw [modelVec, WithLp.prod_inner_apply]
  simp [RCLike.inner_apply]

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E]

/-- An isometry from `E` onto the model space of the same dimension. -/
noncomputable def toModel (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [FiniteDimensional ℝ E] (hd : 1 ≤ Module.finrank ℝ E) :
    E ≃ₗᵢ[ℝ] modelSpace (Module.finrank ℝ E - 1) :=
  (stdOrthonormalBasis ℝ E).repr.trans
    ((LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
      (finCongr (by rw [finrank_modelSpace]; omega))).trans
        (stdOrthonormalBasis ℝ (modelSpace (Module.finrank ℝ E - 1))).repr.symm)

/-- **Lemma 1.**  For `u` uniform on the unit sphere of `E` (`dim E ≥ 2`) and any unit vector
`e`, the coordinate `⟪e, u⟫` has the law `sphericalCoordLaw (dim E)`, i.e. the density `f_d`. -/
theorem sphereUniform_map_inner (hd : 2 ≤ Module.finrank ℝ E) {e : E} (he : ‖e‖ = 1) :
    (sphereUniform E).map (fun u : sphere (0 : E) 1 => ⟪e, (u : E)⟫_ℝ)
      = sphericalCoordLaw (Module.finrank ℝ E) := by
  haveI : Nonempty (Fin (Module.finrank ℝ E - 1)) := ⟨⟨0, by omega⟩⟩
  haveI : Nontrivial (EuclideanSpace ℝ (Fin (Module.finrank ℝ E - 1))) :=
    (WithLp.equiv 2 _).symm.injective.nontrivial
  haveI : Nontrivial E := Module.nontrivial_of_finrank_pos (R := ℝ) (by omega)
  let L₀ := toModel E (by omega)
  obtain ⟨M, hM⟩ := exists_isometry_map_eq (u := L₀ e) (v := modelVec _)
    (by rw [L₀.norm_map, he]) (norm_modelVec _)
  let L := L₀.trans M
  have hL : L e = modelVec _ := by simp [L, hM]
  have hcoord : ∀ u : E, ⟪e, u⟫_ℝ = (WithLp.ofLp (L u)).1 := by
    intro u
    rw [← L.inner_map_map e u, hL, inner_modelVec]
  have hcomp : (fun u : sphere (0 : E) 1 => ⟪e, (u : E)⟫_ℝ)
      = (fun u : sphere (0 : modelSpace (Module.finrank ℝ E - 1)) 1 =>
          (WithLp.ofLp (u : modelSpace (Module.finrank ℝ E - 1))).1) ∘ sphereMap L := by
    funext u
    simp [hcoord]
  rw [hcomp, ← Measure.map_map measurable_sphere_fst (measurable_sphereMap L),
    sphereUniform_map_sphereMap L, sphereUniform_map_fst, finrank_euclideanSpace_fin,
    blockLaw_real _ (by omega)]
  congr 1
  omega

/-- **Lemma 1, block form, for any orthogonal decomposition.**  If `L : E ≃ U × V` is a linear
isometry, the `U`-component of a uniform direction has law `blockLaw U (dim V)`.  For
`E = ℝ^d`, `U` the coordinates of one block of eight and `V` the rest, this is the law `f_{d,8}`
that the `E₈` stage's `ρ` is a quadrature over; for a trellis block of length `T` it is
`f_{d,T}`. -/
theorem sphereUniform_map_block {U V : Type*} [NormedAddCommGroup U] [InnerProductSpace ℝ U]
    [MeasurableSpace U] [BorelSpace U] [FiniteDimensional ℝ U] [Nontrivial U]
    [NormedAddCommGroup V] [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V]
    [FiniteDimensional ℝ V] [Nontrivial V] (L : E ≃ₗᵢ[ℝ] WithLp 2 (U × V)) :
    (sphereUniform E).map (fun u : sphere (0 : E) 1 => (WithLp.ofLp (L u)).1)
      = blockLaw U (Module.finrank ℝ V) := by
  have hcomp : (fun u : sphere (0 : E) 1 => (WithLp.ofLp (L u)).1)
      = (fun u : sphere (0 : WithLp 2 (U × V)) 1 => (WithLp.ofLp (u : WithLp 2 (U × V))).1)
          ∘ sphereMap L := by
    funext u; rfl
  rw [hcomp, ← Measure.map_map measurable_sphere_fst (measurable_sphereMap L),
    sphereUniform_map_sphereMap L, sphereUniform_map_fst]

/-- The uniform direction law as a measure on the ambient space `E`, supported on the sphere:
the measure §11 calls `orbitLaw κ x₀` when the rotation is Haar, and the `σ` against which
§9 and §17 evaluate `ρ²`. -/
noncomputable def sphereUniformAmbient (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [MeasurableSpace E] [BorelSpace E] [FiniteDimensional ℝ E] : Measure E :=
  (sphereUniform E).map Subtype.val

instance [Nontrivial E] : IsProbabilityMeasure (sphereUniformAmbient E) :=
  Measure.isProbabilityMeasure_map measurable_subtype_coe.aemeasurable

/-- The block marginal of the ambient uniform law: the `hmarg` hypothesis of §17's block
quadrature, discharged for the uniform direction. -/
theorem sphereUniformAmbient_map_block {U V : Type*} [NormedAddCommGroup U]
    [InnerProductSpace ℝ U] [MeasurableSpace U] [BorelSpace U] [FiniteDimensional ℝ U]
    [Nontrivial U] [NormedAddCommGroup V] [InnerProductSpace ℝ V] [MeasurableSpace V]
    [BorelSpace V] [FiniteDimensional ℝ V] [Nontrivial V] (L : E ≃ₗᵢ[ℝ] WithLp 2 (U × V)) :
    (sphereUniformAmbient E).map (fun x : E => (WithLp.ofLp (L x)).1)
      = blockLaw U (Module.finrank ℝ V) := by
  have hg : Measurable fun x : E => (WithLp.ofLp (L x)).1 := by
    exact measurable_fst.comp ((WithLp.measurable_ofLp 2 (U × V)).comp L.continuous.measurable)
  rw [sphereUniformAmbient, Measure.map_map hg measurable_subtype_coe]
  exact sphereUniform_map_block L

/-- The coordinate marginal of the ambient uniform law: the paper's Lemma 1 in the form §3b
consumes it (`hmarg : ∀ i, μ.map (fun ω => U ω i) = ν` with `ν = sphericalCoordLaw d`). -/
theorem sphereUniformAmbient_map_inner (hd : 2 ≤ Module.finrank ℝ E) {e : E} (he : ‖e‖ = 1) :
    (sphereUniformAmbient E).map (fun x : E => ⟪e, x⟫_ℝ)
      = sphericalCoordLaw (Module.finrank ℝ E) := by
  have hg : Measurable fun x : E => ⟪e, x⟫_ℝ := (continuous_const.inner continuous_id).measurable
  rw [sphereUniformAmbient, Measure.map_map hg measurable_subtype_coe]
  exact sphereUniform_map_inner hd he

end Coordinate
/-! ## §22  The Haar identification: the direction law *is* the uniform law

Header item 2 was the last gap between §11 and the paper: `orbitLaw κ x₀`, the law of the
rotated direction, was not identified with the uniform law on the sphere.  The identification
needs no construction of Haar measure at all.  If `κ` is a right-invariant probability measure
on a group acting on `E` by isometries and transitively on the unit sphere, the hypotheses
`RotationStream` already carries, then `orbitLaw κ x₀ = sphereUniformAmbient E`
(`orbitLaw_eq_sphereUniformAmbient`).  The proof is the uniqueness-of-invariant-measure step of
Proposition 1 done by hand: the uniform law is `G`-invariant (§21), so integrating a test
function against it and averaging over `κ` changes nothing; swapping the integrals and using
that the orbit law is constant along the orbit (§5) returns the orbit law.

Consequences: `u_p` is uniform on the sphere at every pass (`law_unitVec_eq_sphereUniform`),
`ρ²` is the stage distortion against the uniform law (`integral_norm_resid_sq_uniform`), and
every coordinate of the uniform direction on `ℝ^d` has the law `sphericalCoordLaw d`
(`sphereUniformAmbient_map_coord`), which is the `hmarg` hypothesis of §3b. -/

open scoped InnerProductSpace

section OrbitLawSphere

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableMul G]

/-- The ambient uniform law is invariant under every element of a group acting by isometries. -/
theorem sphereUniformAmbient_map_smul {ι : G → (E ≃ₗᵢ[ℝ] E)}
    (hι : ∀ (g : G) (x : E), ι g x = g • x) (g : G) :
    (sphereUniformAmbient E).map (fun x => g • x) = sphereUniformAmbient E := by
  have h1 : (fun x : E => g • x) ∘ (Subtype.val : sphere (0 : E) 1 → E)
      = Subtype.val ∘ sphereMap (ι g) := by
    funext u; simp [hι]
  rw [sphereUniformAmbient, Measure.map_map (measurable_const_smul g) measurable_subtype_coe, h1,
    ← Measure.map_map measurable_subtype_coe (measurable_sphereMap _), sphereUniform_map_sphereMap]

/-- The ambient uniform law lives on the unit sphere. -/
theorem sphereUniformAmbient_ae_norm : ∀ᵐ x ∂(sphereUniformAmbient E), ‖x‖ = 1 := by
  rw [sphereUniformAmbient, ae_map_iff measurable_subtype_coe.aemeasurable
    (p := fun x : E => ‖x‖ = 1) (measurable_norm (measurableSet_singleton (1 : ℝ)))]
  exact Filter.Eventually.of_forall fun u => mem_sphere_zero_iff_norm.mp u.2

theorem lintegral_orbitLaw (κ : Measure G) (x : E) {f : E → ℝ≥0∞} (hf : Measurable f) :
    ∫⁻ y, f y ∂(orbitLaw κ x) = ∫⁻ g, f (g • x) ∂κ := by
  rw [orbitLaw, lintegral_map hf (measurable_smul_const x)]

/-- **The Haar identification.**  If `κ` is a right-invariant probability measure on a group `G`
acting on `E` by isometries and transitively on the unit sphere, then the law of `g • x₀` for
`g ∼ κ` is the uniform law on the sphere.  This is the uniqueness-of-invariant-measure step of
Proposition 1, proved directly: the uniform law is `G`-invariant (§21), so integrating a test
function against it, then averaging over `κ`, swapping the integrals, and using that the orbit
law is constant along the orbit (§5) returns the orbit law. -/
theorem orbitLaw_eq_sphereUniformAmbient [Nontrivial E] {ι : G → (E ≃ₗᵢ[ℝ] E)}
    (hι : ∀ (g : G) (x : E), ι g x = g • x)
    (κ : Measure G) [IsProbabilityMeasure κ] [κ.IsMulRightInvariant] {x₀ : E}
    (htrans : ∀ x : E, ‖x‖ = 1 → x ∈ MulAction.orbit G x₀) :
    orbitLaw κ x₀ = sphereUniformAmbient E := by
  have key : ∀ f : E → ℝ≥0∞, Measurable f →
      ∫⁻ y, f y ∂(orbitLaw κ x₀) = ∫⁻ y, f y ∂(sphereUniformAmbient E) := by
    intro f hf
    have h1 : ∀ g : G, ∫⁻ x, f (g • x) ∂(sphereUniformAmbient E)
        = ∫⁻ y, f y ∂(sphereUniformAmbient E) := by
      intro g
      rw [← lintegral_map hf (measurable_const_smul g), sphereUniformAmbient_map_smul hι g]
    have h2 : ∫⁻ y, f y ∂(sphereUniformAmbient E)
        = ∫⁻ g, ∫⁻ x, f (g • x) ∂(sphereUniformAmbient E) ∂κ := by
      simp_rw [h1]; rw [lintegral_const, measure_univ, mul_one]
    have hmeas : Measurable (fun p : G × E => f (p.1 • p.2)) := hf.comp measurable_smul
    have h3 : ∫⁻ g, ∫⁻ x, f (g • x) ∂(sphereUniformAmbient E) ∂κ
        = ∫⁻ x, ∫⁻ g, f (g • x) ∂κ ∂(sphereUniformAmbient E) :=
      lintegral_lintegral_swap hmeas.aemeasurable
    have h4 : ∀ᵐ x ∂(sphereUniformAmbient E),
        ∫⁻ g, f (g • x) ∂κ = ∫⁻ y, f y ∂(orbitLaw κ x₀) := by
      filter_upwards [sphereUniformAmbient_ae_norm (E := E)] with x hx
      rw [← lintegral_orbitLaw κ x hf, orbitLaw_eq_of_mem_orbit κ (htrans x hx)]
    rw [h2, h3, lintegral_congr_ae h4, lintegral_const, measure_univ, mul_one]
  refine Measure.ext fun s hs => ?_
  have := key (s.indicator 1) (measurable_const.indicator hs)
  rwa [lintegral_indicator_one hs, lintegral_indicator_one hs] at this

end OrbitLawSphere

section RotationStreamUniform

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] [FiniteDimensional ℝ E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G] [MeasurableMul G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : G → (E ≃ₗᵢ[ℝ] E)} {q : E → E} {W : Ω → E} {R : Ω → ℕ → G} {κ : Measure G}
  [κ.IsMulRightInvariant] {x₀ : E}

/-- For a rotation stream with a unit base point, the direction law of §11 *is* the uniform
law on the sphere: header item 2, discharged from the stream's own hypotheses. -/
theorem RotationStream.orbitLaw_eq (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1) :
    orbitLaw κ x₀ = sphereUniformAmbient E := by
  haveI : IsProbabilityMeasure κ := by
    rw [← h.law_R 0]; exact Measure.isProbabilityMeasure_map (h.meas_R 0).aemeasurable
  haveI : Nontrivial E := ⟨⟨x₀, 0, by rintro rfl; simp at hx₀⟩⟩
  exact orbitLaw_eq_sphereUniformAmbient h.implements κ h.transitive

/-- **Re-isotropisation, closed**: the unit direction `u_p` is uniform on the sphere at every
pass. -/
theorem law_unitVec_eq_sphereUniform (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1)
    (p : ℕ) :
    μ.map (fun ω => unitVec (rotScheme ι q (R ω)) (W ω) p) = sphereUniformAmbient E := by
  rw [law_unitVec_eq_orbitLaw h p, h.orbitLaw_eq hx₀]

/-- **Proposition 2 with the uniform law**: `ρ²` is the stage distortion against the uniform
direction, the quantity Table 2 measures. -/
theorem integral_norm_resid_sq_uniform (h : RotationStream ι q W R κ x₀ μ) (hx₀ : ‖x₀‖ = 1)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun u : E => ‖u - q u‖ ^ 2) (sphereUniformAmbient E)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme ι q (R ω)) (W ω) P‖ ^ 2 ∂μ
      = stageDistortion q (sphereUniformAmbient E) ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  rw [← h.orbitLaw_eq hx₀] at hint ⊢
  exact integral_norm_resid_sq_rotationStream h hW hint P

end RotationStreamUniform

section Coordinates

/-- Lemma 1 for the coordinates of `ℝ^d`: every coordinate of the uniform direction has law
`sphericalCoordLaw d`. -/
theorem sphereUniformAmbient_map_coord {d : ℕ} (hd : 2 ≤ d) (i : Fin d) :
    (sphereUniformAmbient (EuclideanSpace ℝ (Fin d))).map (fun x => x i) = sphericalCoordLaw d := by
  have h := sphereUniformAmbient_map_inner (E := EuclideanSpace ℝ (Fin d))
    (by rw [finrank_euclideanSpace_fin]; exact hd) (e := EuclideanSpace.single i (1 : ℝ))
    (by rw [PiLp.norm_single 2 _, norm_one])
  rw [finrank_euclideanSpace_fin] at h
  convert h using 2
  funext x
  rw [EuclideanSpace.inner_single_left]
  simp

end Coordinates
/-! ## §23  `O(d)` with its Haar measure: the paper's setting, instantiated

§22 identifies the direction law for any group and any right-invariant probability measure with
the right properties.  This section exhibits the paper's own: `O(d)` as Mathlib's
`Matrix.orthogonalGroup`, made a Borel space, shown compact (a closed subset of a box, since the
entries of an orthogonal matrix lie in `[-1, 1]`), given its normalised Haar measure `haarOd d`,
shown right-invariant by `isMulRightInvariant_of_compactSpace`, a general fact about compact
groups proved from uniqueness of Haar measure.  Mathlib's existing action of `O(d)` on `ℝ^d` is
matrix-vector multiplication; it is continuous, hence jointly measurable, by isometries
(`inner_smul`), and transitive on the unit sphere (`transitive`, from §5b and the fact that the
matrix of an isometry in an orthonormal basis is orthogonal).  `orbitLaw_haar` is then
Proposition 1 in full: the law of `g • x₀` for `g` Haar on `O(d)` is the uniform law. -/

/-! ### Right-invariance of Haar measure on a compact group -/

section Unimodular
variable {G : Type*} [Group G] [TopologicalSpace G] [IsTopologicalGroup G] [CompactSpace G]
  [MeasurableSpace G] [BorelSpace G] [SecondCountableTopology G]

/-- On a compact group every left Haar measure is right-invariant: the right translate is again
left-invariant, hence a multiple of the Haar measure by uniqueness, and the total masses agree. -/
theorem isMulRightInvariant_of_compactSpace (μ : Measure G) [IsHaarMeasure μ] :
    μ.IsMulRightInvariant := by
  constructor
  intro g
  haveI : (μ.map (fun x => x * g)).IsMulLeftInvariant := by
    constructor
    intro h
    rw [Measure.map_map (measurable_const_mul h) (measurable_mul_const g)]
    have hcomp : (fun x : G => h * x) ∘ (fun x : G => x * g) = (fun x : G => x * g) ∘ (fun x : G => h * x) := by
      funext x; simp [mul_assoc]
    rw [hcomp, ← Measure.map_map (measurable_mul_const g) (measurable_const_mul h),
      map_mul_left_eq_self]
  haveI : IsFiniteMeasure (μ.map (fun x => x * g)) := isFiniteMeasure_map μ _
  have hK : (⊤ : TopologicalSpace.PositiveCompacts G) = ⊤ := rfl
  have h1 := haarMeasure_unique (μ.map (fun x => x * g)) ⊤
  have h2 := haarMeasure_unique μ ⊤
  have huniv : (μ.map (fun x => x * g)) ((⊤ : TopologicalSpace.PositiveCompacts G) : Set G)
      = μ ((⊤ : TopologicalSpace.PositiveCompacts G) : Set G) := by
    rw [TopologicalSpace.PositiveCompacts.coe_top,
      Measure.map_apply (measurable_mul_const g) MeasurableSet.univ, Set.preimage_univ]
  rw [h1, huniv, ← h2]

end Unimodular

namespace Orthogonal

variable (d : ℕ)

/-- The orthogonal group `O(d)`, as Mathlib's `Matrix.orthogonalGroup`. -/
abbrev Od := Matrix.orthogonalGroup (Fin d) ℝ

/-- `ℝ^d` with its Euclidean structure. -/
abbrev Ed := EuclideanSpace ℝ (Fin d)

instance : SecondCountableTopology (Matrix (Fin d) (Fin d) ℝ) :=
  inferInstanceAs (SecondCountableTopology (Fin d → Fin d → ℝ))

instance : SecondCountableTopology (Od d) :=
  inferInstanceAs (SecondCountableTopology
    ((Matrix.orthogonalGroup (Fin d) ℝ : Submonoid _) : Set (Matrix (Fin d) (Fin d) ℝ)))

instance : MeasurableSpace (Od d) := borel _
instance : BorelSpace (Od d) := ⟨rfl⟩

/-! ### Compactness -/

theorem entry_mem_Icc (g : Od d) (i j : Fin d) :
    (g : Matrix (Fin d) (Fin d) ℝ) i j ∈ Set.Icc (-1 : ℝ) 1 := by
  have h : star (g : Matrix (Fin d) (Fin d) ℝ) * g = 1 := Matrix.mem_unitaryGroup_iff'.mp g.2
  have hjj : (star (g : Matrix (Fin d) (Fin d) ℝ) * g) j j = 1 := by rw [h]; simp
  rw [Matrix.mul_apply] at hjj
  simp only [Matrix.star_apply, star_trivial] at hjj
  have hle : (g : Matrix (Fin d) (Fin d) ℝ) i j * (g : Matrix (Fin d) (Fin d) ℝ) i j ≤ 1 := by
    rw [← hjj]
    exact Finset.single_le_sum (f := fun k => (g : Matrix (Fin d) (Fin d) ℝ) k j *
      (g : Matrix (Fin d) (Fin d) ℝ) k j) (fun k _ => mul_self_nonneg _) (Finset.mem_univ i)
  constructor <;> nlinarith [hle]

theorem isClosed_orthogonalGroup :
    IsClosed ((Matrix.orthogonalGroup (Fin d) ℝ : Submonoid _) : Set (Matrix (Fin d) (Fin d) ℝ)) := by
  have : ((Matrix.orthogonalGroup (Fin d) ℝ : Submonoid _) : Set (Matrix (Fin d) (Fin d) ℝ))
      = {A | star A * A = 1} := by
    ext A; exact Matrix.mem_unitaryGroup_iff'
  rw [this]
  exact isClosed_eq (continuous_star.mul continuous_id) continuous_const

theorem isCompact_box :
    IsCompact (Set.pi Set.univ fun _ : Fin d => Set.pi Set.univ fun _ : Fin d => Set.Icc (-1 : ℝ) 1 :
      Set (Matrix (Fin d) (Fin d) ℝ)) :=
  isCompact_univ_pi fun _ => isCompact_univ_pi fun _ => isCompact_Icc

instance : CompactSpace (Od d) := by
  refine isCompact_iff_compactSpace.mp
    (IsCompact.of_isClosed_subset (isCompact_box d) (isClosed_orthogonalGroup d) ?_)
  intro A hA i _ j _
  exact entry_mem_Icc d ⟨A, hA⟩ i j

/-! ### Haar probability measure -/

/-- Normalised Haar measure on `O(d)`. -/
noncomputable def haarOd : Measure (Od d) := haarMeasure ⊤

instance : IsProbabilityMeasure (haarOd d) :=
  ⟨by rw [haarOd, ← TopologicalSpace.PositiveCompacts.coe_top]; exact haarMeasure_self⟩

instance : IsHaarMeasure (haarOd d) := by unfold haarOd; infer_instance

instance : (haarOd d).IsMulLeftInvariant := inferInstance

instance : (haarOd d).IsMulRightInvariant := isMulRightInvariant_of_compactSpace (haarOd d)

/-! ### The action on `ℝ^d` -/

/-- The action of `O(d)` on `ℝ^d` Mathlib already provides is matrix-vector multiplication. -/
theorem smul_def (g : Od d) (x : Ed d) :
    g • x = Matrix.toEuclideanLin (g : Matrix (Fin d) (Fin d) ℝ) x := by
  rw [Matrix.toEuclideanLin_apply]; rfl

instance : ContinuousSMul (Od d) (Ed d) := by
  constructor
  simp only [smul_def, Matrix.toEuclideanLin_apply]
  fun_prop

instance : MeasurableSMul₂ (Od d) (Ed d) := ContinuousSMul.measurableSMul₂

theorem inner_smul (g : Od d) (x y : Ed d) : ⟪g • x, g • y⟫_ℝ = ⟪x, y⟫_ℝ := by
  rw [smul_def, smul_def, ← LinearMap.adjoint_inner_right,
    ← Matrix.toEuclideanLin_conjTranspose_eq_adjoint, Matrix.toEuclideanLin_apply,
    Matrix.toEuclideanLin_apply, Matrix.mulVec_mulVec, ← Matrix.star_eq_conjTranspose,
    Matrix.mem_unitaryGroup_iff'.mp g.2, Matrix.one_mulVec, WithLp.toLp_ofLp]

theorem norm_smul' (g : Od d) (x : Ed d) : ‖g • x‖ = ‖x‖ := by
  rw [norm_eq_sqrt_real_inner, norm_eq_sqrt_real_inner, inner_smul]

/-- The rotation `g` as a linear isometry equivalence of `ℝ^d`. -/
noncomputable def iso (g : Od d) : Ed d ≃ₗᵢ[ℝ] Ed d :=
  LinearIsometry.toLinearIsometryEquiv
    ⟨Matrix.toEuclideanLin (g : Matrix (Fin d) (Fin d) ℝ), norm_smul' d g⟩ rfl

theorem iso_apply (g : Od d) (x : Ed d) : iso d g x = g • x := by
  rw [iso, LinearIsometry.coe_toLinearIsometryEquiv]; rfl

/-! ### Transitivity on the unit sphere -/

theorem transitive {x₀ : Ed d} (h₀ : ‖x₀‖ = 1) (x : Ed d) (hx : ‖x‖ = 1) :
    x ∈ MulAction.orbit (Od d) x₀ := by
  obtain ⟨L, hL⟩ := exists_isometry_map_eq h₀ hx
  let b := EuclideanSpace.basisFun (Fin d) ℝ
  refine ⟨⟨LinearMap.toMatrix b.toBasis b.toBasis L.toLinearEquiv,
    L.toMatrix_mem_unitaryGroup b b⟩, ?_⟩
  show Matrix.toEuclideanLin (LinearMap.toMatrix b.toBasis b.toBasis L.toLinearEquiv) x₀ = x
  rw [Matrix.toEuclideanLin_eq_toLin, ← EuclideanSpace.basisFun_toBasis, Matrix.toLin_toMatrix]
  exact hL

/-! ### The identification -/

/-- **Proposition 1 at `O(d)`, in full.**  The law of `g • x₀` for `g` Haar-distributed on
`O(d)` and `x₀` a unit vector is the uniform law on the sphere. -/
theorem orbitLaw_haar (x₀ : Ed d) (h₀ : ‖x₀‖ = 1) :
    orbitLaw (haarOd d) x₀ = sphereUniformAmbient (Ed d) := by
  haveI : Nontrivial (Ed d) := ⟨⟨x₀, 0, by rintro rfl; simp at h₀⟩⟩
  exact orbitLaw_eq_sphereUniformAmbient (ι := iso d) (fun g x => iso_apply d g x) (haarOd d)
    (fun x hx => transitive d h₀ x hx)

end Orthogonal
/-! ## §24  Proposition 2 as the paper states it, and a witness in every dimension

`integral_norm_resid_sq_paper` is Proposition 2 with nothing abstract left: Haar rotations on
`O(d)`, a scalar codebook applied coordinate-wise, and `ρ²` the quadrature `d·E_{f_d}[(t−q t)²]`
over the spherical marginal of Lemma 1.  Every hypothesis is about the inputs.

`exists_rotationStream_Od` shows the `RotationStream` bundle is satisfiable at `O(d)` for every
`d`: an i.i.d. Haar stream from `Measure.exists_iid`, the halving stage `x ↦ x/2`, and a unit
base point discharge every field.  `exists_witness_decay_Od` reads `err(P) = (1/2)^P` off the
general theorem, with `ρ² = 1/4` computed against the uniform law.  §12's `d = 1` witness is
now the special case it always was. -/

section Halving
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The halving stage in any space: `q(x) = x/2`, never exact on a nonzero residual. -/
noncomputable def halfE (x : E) : E := (2 : ℝ)⁻¹ • x

theorem norm_sub_halfE (u : E) : ‖u - halfE u‖ = ‖u‖ / 2 := by
  have : u - halfE u = (2 : ℝ)⁻¹ • u := by
    rw [halfE, sub_eq_iff_eq_add, ← add_smul]; norm_num
  rw [this, norm_smul, norm_inv, Real.norm_ofNat, div_eq_inv_mul]

theorem norm_resid_halfE (rot : ℕ → (E ≃ₗᵢ[ℝ] E)) {w : E} (hw : w ≠ 0) (p : ℕ) :
    ‖resid ⟨rot, halfE⟩ w p‖ = ‖w‖ / 2 ^ p := by
  induction p with
  | zero => simp
  | succ p ih =>
      have hpos : (0 : ℝ) < ‖w‖ / 2 ^ p := by
        have := norm_pos_iff.mpr hw
        positivity
      have hne : resid ⟨rot, halfE⟩ w p ≠ 0 := by
        rw [← norm_ne_zero_iff, ih]
        exact ne_of_gt hpos
      rw [norm_resid_succ]
      have hq : (⟨rot, halfE⟩ : Scheme E).q = halfE := rfl
      rw [hq, norm_sub_halfE, norm_unitVec _ _ _ hne, ih]
      ring

theorem resid_halfE_ne_zero (rot : ℕ → (E ≃ₗᵢ[ℝ] E)) {w : E} (hw : w ≠ 0) (p : ℕ) :
    resid ⟨rot, halfE⟩ w p ≠ 0 := by
  rw [← norm_ne_zero_iff, norm_resid_halfE rot hw p]
  have := norm_pos_iff.mpr hw
  positivity

theorem measurable_halfE [MeasurableSpace E] [BorelSpace E] : Measurable (halfE : E → E) :=
  (continuous_const_smul _).measurable

/-- The stage distortion of the halving stage against any probability law on the unit sphere
is exactly `1/4`. -/
theorem stageDistortion_halfE [MeasurableSpace E] {σ : Measure E} [IsProbabilityMeasure σ]
    (hσ : ∀ᵐ u ∂σ, ‖u‖ = 1) : stageDistortion halfE σ = 1 / 4 := by
  unfold stageDistortion
  have h : ∀ᵐ u ∂σ, ‖u - halfE u‖ ^ 2 = 1 / 4 := by
    filter_upwards [hσ] with u hu
    rw [norm_sub_halfE, hu]; norm_num
  rw [integral_congr_ae h, integral_const, probReal_univ, one_smul]

theorem integrable_halfE_err [MeasurableSpace E] [BorelSpace E] {σ : Measure E}
    [IsProbabilityMeasure σ] (hσ : ∀ᵐ u ∂σ, ‖u‖ = 1) :
    Integrable (fun u : E => ‖u - halfE u‖ ^ 2) σ := by
  have h : ∀ᵐ u ∂σ, ‖u - halfE u‖ ^ 2 = 1 / 4 := by
    filter_upwards [hσ] with u hu
    rw [norm_sub_halfE, hu]; norm_num
  exact (integrable_const (1 / 4 : ℝ)).congr (h.mono fun u hu => hu.symm)

end Halving

section PaperStatement
open Orthogonal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- **Proposition 2, exactly as the paper states it.**  Haar rotations on `O(d)`, a scalar
codebook `cb` applied coordinate-wise, and `ρ²` the quadrature `d·E_{f_d}[(t − cb t)²]` over the
spherical marginal: the expected squared residual after `P` passes is `ρ^{2P}` times the
expected squared weight.  Every hypothesis is about the inputs; nothing about any residual law
is assumed. -/
theorem integral_norm_resid_sq_paper {d : ℕ} (hd : 2 ≤ d) {cb : ℝ → ℝ} {W : Ω → Ed d}
    {R : Ω → ℕ → Od d} {x₀ : Ed d}
    (h : RotationStream (iso d) (blockStage cb) W R (haarOd d) x₀ μ) (hx₀ : ‖x₀‖ = 1)
    (hW : Integrable (fun ω => ‖W ω‖ ^ 2) μ)
    (hint : Integrable (fun t => (t - cb t) ^ 2) (sphericalCoordLaw d)) (P : ℕ) :
    ∫ ω, ‖resid (rotScheme (iso d) (blockStage cb) (R ω)) (W ω) P‖ ^ 2 ∂μ
      = rhoSq d cb (sphericalCoordLaw d) ^ P * ∫ ω, ‖W ω‖ ^ 2 ∂μ := by
  have hmarg : ∀ j : Fin d,
      (sphereUniformAmbient (Ed d)).map (fun v => v j) = sphericalCoordLaw d :=
    fun j => sphereUniformAmbient_map_coord hd j
  have hint' : Integrable (fun t : ℝ => ‖t - cb t‖ ^ 2) (sphericalCoordLaw d) := by
    simpa [Real.norm_eq_abs, sq_abs] using hint
  rw [integral_norm_resid_sq_uniform h hx₀ hW (integrable_blockStage_err cb hmarg hint') P,
    stageDistortion_blockStage_real cb hmarg hint]

end PaperStatement

section WitnessOd
open Orthogonal

variable (d : ℕ)

/-- **The hypotheses are satisfiable at `O(d)` for every `d`.**  An i.i.d. Haar rotation stream
on `O(d)` (from `Measure.exists_iid`), the halving stage, and a unit base point satisfy every
field of `RotationStream`.  This replaces §12's `d = 1` witness: the bundle is consistent in the
paper's own setting. -/
theorem exists_rotationStream_Od (x₀ : Ed d) (h₀ : ‖x₀‖ = 1) :
    ∃ (Ω : Type) (_ : MeasurableSpace Ω) (μ : Measure Ω) (R : Ω → ℕ → Od d),
      IsProbabilityMeasure μ ∧
        RotationStream (iso d) halfE (fun _ => x₀) R (haarOd d) x₀ μ := by
  obtain ⟨Ω, mΩ, P, X, hmeas, hlaw, hindep, hprob⟩ := ProbabilityTheory.exists_iid ℕ (haarOd d)
  refine ⟨Ω, mΩ, P, fun ω p => X p ω, hprob, ?_⟩
  have hx₀ : x₀ ≠ 0 := by rintro rfl; simp at h₀
  refine
    { implements := fun g x => iso_apply d g x
      meas_q := measurable_halfE
      meas_W := measurable_const
      meas_R := hmeas
      law_R := fun p => (hlaw p).map_eq
      fresh := ?_
      exact_never := fun p =>
        Filter.Eventually.of_forall fun ω => resid_halfE_ne_zero _ hx₀ p
      transitive := fun x hx => transitive d h₀ x hx }
  intro p
  have hdisj : Disjoint ({p} : Finset ℕ) (Finset.range p) := by
    simp [Finset.disjoint_singleton_left]
  have key := hindep.indepFun_finset ({p} : Finset ℕ) (Finset.range p) hdisj hmeas
  have hmemp : p ∈ ({p} : Finset ℕ) := Finset.mem_singleton_self p
  have hmemi : ∀ i : Fin p, (i : ℕ) ∈ Finset.range p := fun i => Finset.mem_range.mpr i.isLt
  have hφ : Measurable (fun v : (({p} : Finset ℕ) → Od d) => v ⟨p, hmemp⟩) :=
    measurable_pi_apply _
  have hψ : Measurable (fun v : (↥(Finset.range p) → Od d) =>
      (x₀, fun i : Fin p => v ⟨(i : ℕ), hmemi i⟩)) :=
    measurable_const.prodMk (measurable_pi_lambda _ fun _ => measurable_pi_apply _)
  exact key.comp hφ hψ

/-- **The witness, with its decay constant, in every dimension**: `err(P) = (1/2)^P`, read off
the general theorem with `ρ² = 1/4` computed against the uniform law. -/
theorem exists_witness_decay_Od (x₀ : Ed d) (h₀ : ‖x₀‖ = 1) :
    ∃ (Ω : Type) (_ : MeasurableSpace Ω) (μ : Measure Ω) (R : Ω → ℕ → Od d),
      IsProbabilityMeasure μ ∧
        RotationStream (iso d) halfE (fun _ => x₀) R (haarOd d) x₀ μ ∧
          ∀ P : ℕ,
            Real.sqrt (∫ ω, ‖resid (rotScheme (iso d) halfE (R ω)) x₀ P‖ ^ 2 ∂μ) / ‖x₀‖
              = (1 / 2 : ℝ) ^ P := by
  obtain ⟨Ω, mΩ, μ, R, hprob, hstream⟩ := exists_rotationStream_Od d x₀ h₀
  refine ⟨Ω, mΩ, μ, R, hprob, hstream, fun P => ?_⟩
  have hx₀ : x₀ ≠ 0 := by rintro rfl; simp at h₀
  haveI : Nontrivial (Ed d) := ⟨⟨x₀, 0, hx₀⟩⟩
  have hσ := sphereUniformAmbient_ae_norm (E := Ed d)
  have hint : Integrable (fun u : Ed d => ‖u - halfE u‖ ^ 2) (orbitLaw (haarOd d) x₀) := by
    rw [hstream.orbitLaw_eq h₀]; exact integrable_halfE_err hσ
  rw [rms_relative_error_rotationStream hx₀ hstream hint P, hstream.orbitLaw_eq h₀,
    stageDistortion_halfE hσ, show (1 / 4 : ℝ) = (1 / 2) ^ 2 by norm_num,
    Real.sqrt_sq (by norm_num)]

end WitnessOd
/-!
# §45  Uniqueness of the rotation-invariant law on the sphere

Header items 1 and 2 said that the uniform law on the sphere is rotation-invariant (§21) and
that the orbit law of a Haar rotation is the uniform law (§22, §23), but that the uniform law
being the *only* rotation-invariant probability measure on the sphere was not proved.  It is
the same argument as §22 run in the other direction: if `μ` is a `G`-invariant probability
measure carried by the unit sphere, then `μ(s) = ∫ μ((g•·)⁻¹ s) dκ(g)` for a right-invariant
probability `κ` on `G`; swapping the integrals turns the right side into
`∫ (orbitLaw κ x)(s) dμ(x)`, and the orbit law of every point of the sphere is the uniform law.

* `eq_sphereUniformAmbient_of_invariant`: the abstract statement, for a group acting by
  isometries and transitively on the unit sphere with a right-invariant probability measure.
* `Orthogonal.eq_sphereUniformAmbient_of_invariant`: at `O(d)` with its Haar measure, every
  `O(d)`-invariant probability measure on `ℝ^d` carried by the unit sphere is the uniform law.
-/

open scoped InnerProductSpace

section Uniqueness

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E]
  [BorelSpace E] [FiniteDimensional ℝ E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableMul G]

/-- **Uniqueness of the invariant law on the sphere.**  If `G` acts on `E` by isometries and
transitively on the unit sphere, and carries a right-invariant probability measure `κ`, then
every `G`-invariant probability measure on `E` carried by the unit sphere is the uniform law:
average the invariance `μ = (g • ·)_* μ` over `g ∼ κ`, swap the integrals, and use that the
orbit law of every point of the sphere is the uniform law (§22). -/
theorem eq_sphereUniformAmbient_of_invariant [Nontrivial E] {ι : G → (E ≃ₗᵢ[ℝ] E)}
    (hι : ∀ (g : G) (x : E), ι g x = g • x)
    (κ : Measure G) [IsProbabilityMeasure κ] [κ.IsMulRightInvariant]
    (htrans : ∀ x y : E, ‖x‖ = 1 → ‖y‖ = 1 → y ∈ MulAction.orbit G x)
    (μ : Measure E) [IsProbabilityMeasure μ] (hμ : ∀ᵐ x ∂μ, ‖x‖ = 1)
    (hinv : ∀ g : G, μ.map (fun x => g • x) = μ) :
    μ = sphereUniformAmbient E := by
  ext s hs
  have hmeas : Measurable fun p : G × E => s.indicator (1 : E → ℝ≥0∞) (p.1 • p.2) :=
    (measurable_const.indicator hs).comp measurable_smul
  have h1 : μ s = ∫⁻ g, ∫⁻ x, s.indicator (1 : E → ℝ≥0∞) (g • x) ∂μ ∂κ := by
    have : ∀ g : G, ∫⁻ x, s.indicator (1 : E → ℝ≥0∞) (g • x) ∂μ = μ s := by
      intro g
      have hg : Measurable fun x : E => g • x := measurable_const_smul g
      calc ∫⁻ x, s.indicator (1 : E → ℝ≥0∞) (g • x) ∂μ
          = ∫⁻ x, ((fun x : E => g • x) ⁻¹' s).indicator (1 : E → ℝ≥0∞) x ∂μ := by
            refine lintegral_congr fun x => ?_
            by_cases hx : g • x ∈ s
            · rw [Set.indicator_of_mem hx, Set.indicator_of_mem (by exact hx)]; rfl
            · rw [Set.indicator_of_notMem hx, Set.indicator_of_notMem (by exact hx)]
        _ = μ ((fun x : E => g • x) ⁻¹' s) := lintegral_indicator_one (hg hs)
        _ = (μ.map fun x : E => g • x) s := (Measure.map_apply hg hs).symm
        _ = μ s := by rw [hinv g]
    simp_rw [this]
    rw [lintegral_const, measure_univ, mul_one]
  have h2 : ∫⁻ g, ∫⁻ x, s.indicator (1 : E → ℝ≥0∞) (g • x) ∂μ ∂κ
      = ∫⁻ x, ∫⁻ g, s.indicator (1 : E → ℝ≥0∞) (g • x) ∂κ ∂μ :=
    lintegral_lintegral_swap (f := fun g x => s.indicator (1 : E → ℝ≥0∞) (g • x))
      hmeas.aemeasurable
  have h3 : ∀ x : E, ∫⁻ g, s.indicator (1 : E → ℝ≥0∞) (g • x) ∂κ = (orbitLaw κ x) s := by
    intro x
    have := lintegral_indicator_one (μ := orbitLaw κ x) hs
    rw [← this, lintegral_orbitLaw κ x (f := s.indicator 1) (measurable_const.indicator hs)]
  have h4 : ∫⁻ x, (orbitLaw κ x) s ∂μ = ∫⁻ x, (sphereUniformAmbient E) s ∂μ := by
    refine lintegral_congr_ae ?_
    filter_upwards [hμ] with x hx
    rw [orbitLaw_eq_sphereUniformAmbient hι κ (fun y hy => htrans x y hx hy)]
  rw [h1, h2]
  simp_rw [h3]
  rw [h4, lintegral_const, measure_univ, mul_one]

end Uniqueness

namespace Orthogonal
variable (d : ℕ)

/-- **Uniqueness at `O(d)`.**  Every `O(d)`-invariant probability measure on `ℝ^d` carried by
the unit sphere is the uniform law on the sphere.  With `orbitLaw_haar` this is Proposition 1
in the form the paper's proof invokes: the uniform law is *the* rotation-invariant law. -/
theorem eq_sphereUniformAmbient_of_invariant (μ : Measure (Ed d)) [IsProbabilityMeasure μ]
    (hμ : ∀ᵐ x ∂μ, ‖x‖ = 1) (hinv : ∀ g : Od d, μ.map (fun x => g • x) = μ) :
    μ = sphereUniformAmbient (Ed d) := by
  haveI : (ae μ).NeBot := ae_neBot.mpr (IsProbabilityMeasure.ne_zero μ)
  obtain ⟨x₀, h₀⟩ := hμ.exists
  haveI : Nontrivial (Ed d) := ⟨⟨x₀, 0, by rintro rfl; simp at h₀⟩⟩
  exact Progressive.eq_sphereUniformAmbient_of_invariant (ι := iso d) (fun g x => iso_apply d g x)
    (haarOd d) (fun x y hx hy => transitive d hx y hy) μ hμ hinv

end Orthogonal

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

/-- The closed-form normalising mass of the block density. -/
noncomputable def blockMass (B m : ℕ) : ℝ :=
  Real.sqrt π ^ B * Real.Gamma ((m : ℝ) / 2) / Real.Gamma (((B : ℝ) + m) / 2)

theorem blockMass_pos {B m : ℕ} (hm : 1 ≤ m) : 0 < blockMass B m := by
  unfold blockMass
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have h1 : 0 < Real.Gamma ((m : ℝ) / 2) := Real.Gamma_pos_of_pos (by positivity)
  have h2 : 0 < Real.Gamma (((B : ℝ) + m) / 2) := Real.Gamma_pos_of_pos (by positivity)
  positivity

/-- **The mass of the block density**: `∫_{ball} (1 − ‖x‖²)^{(m−2)/2} dx = √π^B Γ(m/2)/Γ((B+m)/2)`,
read off from the block marginal of the sphere measure (§20) applied to the whole space, with
the volumes of balls from Mathlib. -/
theorem blockMeasure_univ (m : ℕ) (hm : 1 ≤ m) :
    blockMeasure U m Set.univ = ENNReal.ofReal (blockMass (Module.finrank ℝ U) m) := by
  haveI : Nonempty (Fin m) := ⟨⟨0, by omega⟩⟩
  have hmV : Module.finrank ℝ (EuclideanSpace ℝ (Fin m)) = m := finrank_euclideanSpace_fin
  have h := congrArg (fun ν : Measure U => ν Set.univ)
    (toSphere_map_fst (U := U) (V := EuclideanSpace ℝ (Fin m)))
  rw [Measure.map_apply measurable_sphere_fst MeasurableSet.univ, Set.preimage_univ,
    Measure.smul_apply, smul_eq_mul, hmV, Measure.toSphere_apply_univ, Measure.toSphere_apply_univ,
    InnerProductSpace.volume_ball, InnerProductSpace.volume_ball, finrank_withLp_prod, hmV] at h
  set B := Module.finrank ℝ U with hB
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hB' : (0 : ℝ) < B := by exact_mod_cast Module.finrank_pos
  have hΓm : 0 < Real.Gamma ((m : ℝ) / 2) := Real.Gamma_pos_of_pos (by positivity)
  have hΓBm : 0 < Real.Gamma (((B : ℝ) + m) / 2) := Real.Gamma_pos_of_pos (by positivity)
  have hΓm1 : 0 < Real.Gamma ((m : ℝ) / 2 + 1) := Real.Gamma_pos_of_pos (by positivity)
  have hΓBm1 : 0 < Real.Gamma (((B + m : ℕ) : ℝ) / 2 + 1) := Real.Gamma_pos_of_pos (by positivity)
  have hc : (m : ℝ≥0∞) * (ENNReal.ofReal 1 ^ m
      * ENNReal.ofReal (Real.sqrt π ^ m / Real.Gamma ((m : ℝ) / 2 + 1)))
      = ENNReal.ofReal (m * (Real.sqrt π ^ m / Real.Gamma ((m : ℝ) / 2 + 1))) := by
    rw [ENNReal.ofReal_one, one_pow, one_mul, ENNReal.ofReal_mul (by positivity),
      ENNReal.ofReal_natCast]
  have ha : ((B + m : ℕ) : ℝ≥0∞) * (ENNReal.ofReal 1 ^ (B + m)
      * ENNReal.ofReal (Real.sqrt π ^ (B + m) / Real.Gamma (((B + m : ℕ) : ℝ) / 2 + 1)))
      = ENNReal.ofReal ((B + m : ℕ) * (Real.sqrt π ^ (B + m)
          / Real.Gamma (((B + m : ℕ) : ℝ) / 2 + 1))) := by
    rw [ENNReal.ofReal_one, one_pow, one_mul, ENNReal.ofReal_mul (by positivity),
      ENNReal.ofReal_natCast]
  rw [ha, hc] at h
  have hc0 : ENNReal.ofReal (m * (Real.sqrt π ^ m / Real.Gamma ((m : ℝ) / 2 + 1))) ≠ 0 := by
    rw [ne_eq, ENNReal.ofReal_eq_zero, not_le]; positivity
  refine (ENNReal.mul_right_inj hc0 ENNReal.ofReal_ne_top).mp ?_
  rw [← h, ← ENNReal.ofReal_mul (by positivity)]
  congr 1
  unfold blockMass
  rw [Real.Gamma_add_one (by positivity : ((m : ℝ) / 2) ≠ 0),
    Real.Gamma_add_one (by positivity : (((B + m : ℕ) : ℝ) / 2) ≠ 0)]
  push_cast
  field_simp
  ring

/-- **Lemma 1, block form, with its constant**: the law of a `B`-block of a uniform direction on
the sphere of dimension `B + m` has density `Γ((B+m)/2)/(√π^B Γ(m/2)) · (1 − ‖x‖²)^{(m−2)/2}` on
the unit ball. -/
theorem blockLaw_eq_withDensity (m : ℕ) (hm : 1 ≤ m) :
    blockLaw U m = (volume.restrict (ball (0 : U) 1)).withDensity fun x =>
      ENNReal.ofReal ((blockMass (Module.finrank ℝ U) m)⁻¹ * (1 - ‖x‖ ^ 2) ^ (((m : ℝ) - 2) / 2)) := by
  rw [blockLaw, blockMeasure_univ U m hm, ← ENNReal.ofReal_inv_of_pos (blockMass_pos hm), blockMeasure,
    ← withDensity_smul _ (measurable_blockDensity m)]
  congr 1
  funext x
  rw [Pi.smul_apply, smul_eq_mul, blockDensity, ← ENNReal.ofReal_mul (inv_nonneg.mpr (blockMass_pos hm).le)]

end BlockConstant

/-! ## §8  Scope: what is *not* claimed

The paper's Remark ("Scope") is careful that per-pass scalar optimality does **not** imply that
`P` passes at `b` bits beat one pass at `Pb` bits.  There is nothing to prove here — the point is
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
  §25–§35 prove the variance estimate the remark rests on — `variance_stageErr_le_bounded`,
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
