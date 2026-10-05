
/-- At `F = ℝ` the block quadrature is §3b's `rhoSq`: the scalar quadrature is the block
quadrature with one-dimensional blocks. -/
theorem stageDistortion_blockStage_real {σ : Measure (EuclideanSpace ℝ (Fin m))} {ν : Measure ℝ}
    (cb : ℝ → ℝ) (hmarg : ∀ j, σ.map (fun v => v j) = ν)
    (hint : Integrable (fun t => (t - cb t) ^ 2) ν) :
    stageDistortion (blockStage cb) σ = rhoSq m cb ν := by
  have hint' : Integrable (fun t : ℝ => ‖t - cb t‖ ^ 2) ν := by
    simpa [Real.norm_eq_abs, sq_abs] using hint
  rw [stageDistortion_blockStage cb hmarg hint', rhoSq]
  simp [Real.norm_eq_abs, sq_abs]

end Block

/-! ### §17b  Identically distributed blocks, from invariance of the direction law

`stageDistortion_blockStage` assumes every block has the same law.  For the paper's uniform law
that is permutation invariance; here it is derived from the rotation stream alone: the orbit law
is invariant under the group when `κ` is left-invariant (Haar on a compact group is), so if the
group contains an element that moves block `j'` to block `j`, the two block marginals agree. -/

section BlockExchange

variable {G : Type*} [Group G] [MeasurableSpace G] [MeasurableMul G]
variable {X : Type*} [MeasurableSpace X] [MulAction G X] [MeasurableSMul G X]
