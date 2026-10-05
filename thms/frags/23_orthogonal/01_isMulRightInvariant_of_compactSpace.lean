
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
