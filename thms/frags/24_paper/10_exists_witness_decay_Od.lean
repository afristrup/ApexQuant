
/-- **The witness, with its decay constant, in every dimension**: `err(P) = (1/2)^P`, read off
the general theorem with `ρ² = 1/4` computed against the uniform law. -/
theorem exists_witness_decay_Od (x₀ : Ed d) (h₀ : ‖x₀‖ = 1) :
    ∃ (Ω : Type) (_ : MeasurableSpace Ω) (μ : Measure Ω) (R : Ω → ℕ → Od d),
      IsProbabilityMeasure μ ∧
        RotationStream (iso d) halfE (fun _ => x₀) R (haarOd d) x₀ μ ∧
          ∀ P : ℕ,
            Real.sqrt (∫ ω, ‖resid (rotScheme (iso d) halfE (R ω)) x₀ P‖ ^ 2 ∂μ) / ‖x₀‖
              = (1 / 2 : ℝ) ^ P := by
  obtain ⟨Ω, mΩ, μ, R, hprob, hstream⟩ := exists_rotationStream_Od d x₀ h₀
  refine ⟨Ω, mΩ, μ, R, hprob, hstream, fun P => ?_⟩
  have hx₀ : x₀ ≠ 0 := by rintro rfl; simp at h₀
  haveI : Nontrivial (Ed d) := ⟨⟨x₀, 0, hx₀⟩⟩
  have hσ := sphereUniformAmbient_ae_norm (E := Ed d)
  have hint : Integrable (fun u : Ed d => ‖u - halfE u‖ ^ 2) (orbitLaw (haarOd d) x₀) := by
    rw [hstream.orbitLaw_eq h₀]; exact integrable_halfE_err hσ
  rw [rms_relative_error_rotationStream hx₀ hstream hint P, hstream.orbitLaw_eq h₀,
    stageDistortion_halfE hσ, show (1 / 4 : ℝ) = (1 / 2) ^ 2 by norm_num,
    Real.sqrt_sq (by norm_num)]

end WitnessOd
