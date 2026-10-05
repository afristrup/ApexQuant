
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
