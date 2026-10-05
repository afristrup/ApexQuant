
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
