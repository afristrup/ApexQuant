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
