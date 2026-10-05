
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
