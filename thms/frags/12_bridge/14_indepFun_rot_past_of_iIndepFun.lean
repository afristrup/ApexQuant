
/-- **The fresh-rotation lemma.**  If the weight row and the whole rotation stream are mutually
independent, then the pass-`p` rotation is independent of the pair *(weight row, all earlier
rotations)*, which by `resid_rotScheme_eq_residPast` is everything `r_p` depends on. -/
theorem indepFun_rot_past_of_iIndepFun {W : Ω → E} {R : Ω → ℕ → G}
    (hW : Measurable W) (hR : ∀ p, Measurable (fun ω => R ω p))
    (h : iIndepFun (streamVar W R) μ) (p : ℕ) :
    IndepFun (fun ω => R ω p) (fun ω => (W ω, fun i : Fin p => R ω i)) μ := by
  have hmeas : ∀ i, Measurable (streamVar W R i) := by
    rintro (_ | q)
    · exact hW
    · exact hR q
  have hdisj : Disjoint ({some p} : Finset (Option ℕ))
      (insert none ((Finset.range p).image some)) := by
    simp [Finset.disjoint_singleton_left]
  have key := h.indepFun_finset {some p} (insert none ((Finset.range p).image some)) hdisj hmeas
  have hmemp : some p ∈ ({some p} : Finset (Option ℕ)) := Finset.mem_singleton_self _
  have hmemn : none ∈ insert none ((Finset.range p).image some) := Finset.mem_insert_self _ _
  have hmemi : ∀ i : Fin p, some (i : ℕ) ∈ insert none ((Finset.range p).image some) := fun i =>
    Finset.mem_insert_of_mem (Finset.mem_image_of_mem some (Finset.mem_range.mpr i.isLt))
  have hφ : Measurable (fun v : ((i : ({some p} : Finset (Option ℕ))) → streamSpace E G i) =>
      v ⟨some p, hmemp⟩) := measurable_pi_apply _
  have hψ : Measurable
      (fun v : ((i : (insert none ((Finset.range p).image some) : Finset (Option ℕ))) →
          streamSpace E G i) =>
        ((v ⟨none, hmemn⟩ : E), fun i : Fin p => (v ⟨some (i : ℕ), hmemi i⟩ : G))) :=
    (measurable_pi_apply _).prodMk
      (measurable_pi_lambda _ fun _ => measurable_pi_apply _)
  exact key.comp hφ hψ

end BridgeStream

/-! ### §11c  Assembling `IsotropicPasses` -/

section BridgeAssembly

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
