
/-! ## §48b  The coordinate law is one scalar law

§48 shows that every coordinate of `H D_ε v` gives the same *average* of every `f`.  That is an
identity between numbers, one for each `f`, and downstream arguments that want to name the
common object have to keep carrying `f` along.

This section upgrades it to an identity of measures.  `radLaw d v` is the law of the Rademacher
sum `∑ⱼ vⱼ εⱼ / √d` under the uniform law on the sign cube, so it is a law on `ℝ` carried by at
most `2 ^ d` points; `coordLaw_eq_radLaw` identifies the law of coordinate `i` of `H D_ε v` with
it for every flat `H` and every `i`, and `coordLaw_indep` is the resulting statement that the
coordinate law depends on `v` alone.

The route is a pushforward computation and nothing more: counting measure on the sign cube is
invariant under a relabelling of the cube (`map_count_equiv`), hence so is the uniform law
(`map_uniformSigns_equiv`), and the flip of the disagreeing bits is the relabelling that carries
one weight vector to another agreeing with it up to sign (`map_radSum_congr_abs`).  The averaged
form `signAvg_radSum_congr_abs` of §48 is the corresponding statement against a fixed test
function; it is proved on its own in §48 and is not rederived from the measure form here.

Nothing here is asymptotic, nothing is assumed about `v`, and no claim is made about how
`radLaw d v` relates to the spherical coordinate law `sphericalCoordLaw d`.  The only comparison
of those two anywhere in this development is §37's, between their fourth moments; §41 to §44
compare the coordinate law with a *Gaussian*, which is a different second law and does not
compose with §37 into a statement about `f_d`. -/

namespace CoordLaw

open Hadamard

variable {n d : ℕ}

/-- Counting measure on the sign cube is invariant under any relabelling of the cube. -/
theorem map_count_equiv (e : Signs n ≃ Signs n) :
    (Measure.count : Measure (Signs n)).map e = Measure.count := by
  refine Measure.ext fun s hs => ?_
  have himg : (e : Signs n → Signs n) ⁻¹' s = (e.symm : Signs n → Signs n) '' s := by
    rw [Equiv.image_eq_preimage_symm, Equiv.symm_symm]
  rw [Measure.map_apply (measurable_of_countable _) hs, himg,
    Measure.count_injective_image e.symm.injective]

/-- The uniform law on the sign cube is invariant under any relabelling of the cube. -/
theorem map_uniformSigns_equiv (e : Signs n ≃ Signs n) :
    (uniformSigns n).map e = uniformSigns n := by
  rw [uniformSigns, Measure.map_smul, map_count_equiv e]

/-- Two weight vectors agreeing up to sign are carried to one another by a relabelling of the
sign cube, namely the flip of the bits where they disagree.  This is the same pointwise identity
that `signAvg_radSum_congr_abs` establishes inline in §48, packaged here so that the flip itself
stays out of the statement. -/
theorem exists_equiv_radSum_comp {a b : Fin n → ℝ} (hab : ∀ j, |a j| = |b j|) :
    ∃ e : Signs n ≃ Signs n, ∀ s, radSum b (e s) = radSum a s := by
  refine ⟨flipEquiv (fun j => decide (a j ≠ b j)), fun s => ?_⟩
  unfold radSum
  refine Finset.sum_congr rfl fun j _ => ?_
  show b j * sgn (xor (decide (a j ≠ b j)) (s j)) = a j * sgn (s j)
  by_cases h : a j = b j
  · have hf : decide (a j ≠ b j) = false := by simp [h]
    rw [hf, Bool.false_xor, h]
  · have ht : decide (a j ≠ b j) = true := by simp [h]
    have hneg : a j = -b j := (abs_eq_abs.mp (hab j)).resolve_left h
    rw [ht, Bool.true_xor, sgn_not, hneg]
    ring

/-- **The law of a Rademacher sum depends only on `|aⱼ|`**, as an equality of measures and not
only under an average. -/
theorem map_radSum_congr_abs {a b : Fin n → ℝ} (hab : ∀ j, |a j| = |b j|) :
    (uniformSigns n).map (radSum a) = (uniformSigns n).map (radSum b) := by
  obtain ⟨e, he⟩ := exists_equiv_radSum_comp hab
  have hcomp : (radSum b) ∘ (e : Signs n → Signs n) = radSum a := funext he
  rw [← hcomp, ← Measure.map_map (measurable_of_countable _) (measurable_of_countable _),
    map_uniformSigns_equiv]

/-- The law of `∑ⱼ vⱼ εⱼ / √d` when `ε` is uniform on the sign cube: a law on `ℝ` carried by
the image of the cube, so by at most `2 ^ d` points. -/
noncomputable def radLaw (d : ℕ) (v : Ed d) : Measure ℝ :=
  (uniformSigns d).map (fun s => radSum (fun j => v j / Real.sqrt d) s)

/-- **Every coordinate of a flat randomised rotation carries the law of `∑ⱼ vⱼ εⱼ / √d`.**  The
right-hand side mentions neither `i` nor `H`.  This is `signAvg_coord_eq_signAvg_radSum` as an
equality of measures. -/
theorem coordLaw_eq_radLaw {H : Od d} (hH : IsFlat H) (v : Ed d) (i : Fin d) :
    coordLaw H v i = radLaw d v := by
  have habs : ∀ j, |(H : Matrix (Fin d) (Fin d) ℝ) i j * v j| = |v j / Real.sqrt d| := by
    intro j
    have h1 : |(H : Matrix (Fin d) (Fin d) ℝ) i j| = 1 / Real.sqrt d := by
      have h := hH i j
      have h2 : |(H : Matrix (Fin d) (Fin d) ℝ) i j| = Real.sqrt (1 / (d : ℝ)) := by
        rw [← h, Real.sqrt_sq_eq_abs]
      rw [h2, one_div, Real.sqrt_inv, one_div]
    rw [abs_mul, h1, abs_div, abs_of_nonneg (Real.sqrt_nonneg ((d : ℝ)))]
    ring
  have hcoord : (fun s => (hadRot H s • v) i)
      = radSum (fun j => (H : Matrix (Fin d) (Fin d) ℝ) i j * v j) :=
    funext fun s => hadRot_apply_eq_radSum H s v i
  rw [coordLaw, hcoord, radLaw]
  exact map_radSum_congr_abs habs

/-- **The coordinate law is the same for every flat rotation and every coordinate.**  Two flat
`H`, `H'` and any two coordinates give the same law, so the law depends on `v` alone. -/
theorem coordLaw_indep {H H' : Od d} (hH : IsFlat H) (hH' : IsFlat H') (v : Ed d)
    (i i' : Fin d) : coordLaw H v i = coordLaw H' v i' := by
  rw [coordLaw_eq_radLaw hH v i, coordLaw_eq_radLaw hH' v i']

instance (d : ℕ) (v : Ed d) : IsProbabilityMeasure (radLaw d v) := by
  constructor
  rw [radLaw, Measure.map_apply (measurable_of_countable _) MeasurableSet.univ,
    Set.preimage_univ, measure_univ]

/-- Integration against the Rademacher law is an average over the sign cube. -/
theorem integral_radLaw (d : ℕ) (v : Ed d) {g : ℝ → ℝ} (hg : Measurable g) :
    ∫ x, g x ∂(radLaw d v) = signAvg fun s => g (radSum (fun j => v j / Real.sqrt d) s) := by
  rw [radLaw, integral_map (measurable_of_countable _).aemeasurable hg.aestronglyMeasurable,
    integral_uniformSigns]

end CoordLaw
