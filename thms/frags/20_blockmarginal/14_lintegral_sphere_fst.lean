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
