/-!
# §47  The dimension restriction of Lemma 1 is necessary

Lemma 1 (`sphericalPdf_*`, §4) and the coordinate law it describes
(`sphereUniformAmbient_map_coord`, §22) both carry the hypothesis `2 ≤ d`.  The paper states
that this is necessary rather than cosmetic.  That claim is proved here rather than asserted.

At `d = 1` the unit sphere of `ℝ¹` is the two-point set `{-1, 1}`, so the law of the single
coordinate of a uniform direction puts all its mass on two points.  A measure carried by a
finite set is singular with respect to Lebesgue measure, so it admits no density: there is no
`f₁` for which Lemma 1 could hold, and weakening `2 ≤ d` to `1 ≤ d` would make the statement
false rather than merely unproved.  The corresponding restriction on §13's variance bound is of
a different kind, arithmetic rather than measure-theoretic, since that bound divides by `d - 2`.

* `norm_eq_abs_coord_one`: on `ℝ¹` the norm is the absolute value of the one coordinate.
* `coord_mem_pair_of_mem_sphere_one`: the unit sphere of `ℝ¹` lands in `{-1, 1}`.
* `coordLaw_one_pair`: that two-point set carries all of the coordinate law's mass.
* `not_absolutelyContinuous_coordLaw_one`: the law is therefore not absolutely continuous with
  respect to `volume`, hence has no density.
-/

open Metric

section DimensionNecessary

theorem norm_eq_abs_coord_one (x : EuclideanSpace ℝ (Fin 1)) : ‖x‖ = |x 0| := by
  rw [EuclideanSpace.norm_eq]
  simp [Real.sqrt_sq_eq_abs]

theorem coord_mem_pair_of_mem_sphere_one {x : EuclideanSpace ℝ (Fin 1)}
    (hx : x ∈ sphere (0 : EuclideanSpace ℝ (Fin 1)) 1) : x 0 ∈ ({-1, 1} : Set ℝ) := by
  have hx1 : ‖x‖ = 1 := by simpa using mem_sphere_iff_norm.mp hx
  have habs : |x 0| = 1 := by rw [← norm_eq_abs_coord_one]; exact hx1
  rcases (abs_eq (by norm_num : (0:ℝ) ≤ 1)).mp habs with h | h
  · simp [h]
  · simp [h]

theorem measurable_coord_one : Measurable (fun x : EuclideanSpace ℝ (Fin 1) => x 0) := by
  fun_prop

/-- At `d = 1` the whole coordinate law sits on `{-1, 1}`. -/
theorem coordLaw_one_pair :
    ((sphereUniformAmbient (EuclideanSpace ℝ (Fin 1))).map (fun x => x 0))
      ({-1, 1} : Set ℝ) = 1 := by
  have hpair : MeasurableSet ({-1, 1} : Set ℝ) :=
    (measurableSet_singleton _).union (measurableSet_singleton _)
  rw [Measure.map_apply measurable_coord_one hpair, sphereUniformAmbient,
    Measure.map_apply measurable_subtype_coe (measurable_coord_one hpair)]
  have huniv : (Subtype.val ⁻¹'
      ((fun x : EuclideanSpace ℝ (Fin 1) => x 0) ⁻¹' ({-1, 1} : Set ℝ)) :
      Set (sphere (0 : EuclideanSpace ℝ (Fin 1)) 1)) = Set.univ := by
    ext p
    simp only [Set.mem_preimage, Set.mem_univ, iff_true]
    exact coord_mem_pair_of_mem_sphere_one p.2
  rw [huniv, measure_univ]

/-- Lemma 1's `2 ≤ d` is necessary: at `d = 1` the coordinate law of a uniform direction is
carried by two points, so it is singular with respect to Lebesgue measure and has no density. -/
theorem not_absolutelyContinuous_coordLaw_one :
    ¬ ((sphereUniformAmbient (EuclideanSpace ℝ (Fin 1))).map (fun x => x 0)
        ≪ (volume : Measure ℝ)) := by
  intro hac
  have hfin : ({-1, 1} : Set ℝ).Finite := (Set.finite_singleton _).insert _
  have hzero := hac (hfin.measure_zero volume)
  rw [coordLaw_one_pair] at hzero
  exact one_ne_zero hzero

end DimensionNecessary
