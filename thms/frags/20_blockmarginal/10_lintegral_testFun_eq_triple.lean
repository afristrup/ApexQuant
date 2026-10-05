
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
