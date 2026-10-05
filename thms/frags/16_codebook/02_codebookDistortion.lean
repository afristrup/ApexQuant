
/-- The distortion of a codebook against the direction law: the expected squared distance from
a direction to its nearest codeword.  This is `ρ²` as a functional of the codebook alone. -/
noncomputable def codebookDistortion (C : Set E) (σ : Measure E) : ℝ :=
  ∫ u, Metric.infDist u C ^ 2 ∂σ

omit [MeasurableSpace E] in
