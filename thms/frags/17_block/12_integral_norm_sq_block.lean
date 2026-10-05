
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
