
/-- **The hypotheses are satisfiable at `O(d)` for every `d`.**  An i.i.d. Haar rotation stream
on `O(d)` (from `Measure.exists_iid`), the halving stage, and a unit base point satisfy every
field of `RotationStream`.  This replaces §12's `d = 1` witness: the bundle is consistent in the
paper's own setting. -/
theorem exists_rotationStream_Od (x₀ : Ed d) (h₀ : ‖x₀‖ = 1) :
    ∃ (Ω : Type) (_ : MeasurableSpace Ω) (μ : Measure Ω) (R : Ω → ℕ → Od d),
      IsProbabilityMeasure μ ∧
        RotationStream (iso d) halfE (fun _ => x₀) R (haarOd d) x₀ μ := by
  obtain ⟨Ω, mΩ, P, X, hmeas, hlaw, hindep, hprob⟩ := ProbabilityTheory.exists_iid ℕ (haarOd d)
  refine ⟨Ω, mΩ, P, fun ω p => X p ω, hprob, ?_⟩
  have hx₀ : x₀ ≠ 0 := by rintro rfl; simp at h₀
  refine
    { implements := fun g x => iso_apply d g x
      meas_q := measurable_halfE
      meas_W := measurable_const
      meas_R := hmeas
      law_R := fun p => (hlaw p).map_eq
      fresh := ?_
      exact_never := fun p =>
        Filter.Eventually.of_forall fun ω => resid_halfE_ne_zero _ hx₀ p
      transitive := fun x hx => transitive d h₀ x hx }
  intro p
  have hdisj : Disjoint ({p} : Finset ℕ) (Finset.range p) := by
    simp [Finset.disjoint_singleton_left]
  have key := hindep.indepFun_finset ({p} : Finset ℕ) (Finset.range p) hdisj hmeas
  have hmemp : p ∈ ({p} : Finset ℕ) := Finset.mem_singleton_self p
  have hmemi : ∀ i : Fin p, (i : ℕ) ∈ Finset.range p := fun i => Finset.mem_range.mpr i.isLt
  have hφ : Measurable (fun v : (({p} : Finset ℕ) → Od d) => v ⟨p, hmemp⟩) :=
    measurable_pi_apply _
  have hψ : Measurable (fun v : (↥(Finset.range p) → Od d) =>
      (x₀, fun i : Fin p => v ⟨(i : ℕ), hmemi i⟩)) :=
    measurable_const.prodMk (measurable_pi_lambda _ fun _ => measurable_pi_apply _)
  exact key.comp hφ hψ
