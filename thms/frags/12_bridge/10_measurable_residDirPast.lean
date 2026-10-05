
theorem measurable_residDirPast {q : E → E} (hq : Measurable q) (p : ℕ) :
    Measurable (residDirPast (G := G) q p) :=
  ((measurable_residPast hq p).norm.inv).smul (measurable_residPast hq p)

end BridgeMeasurable

/-! ### §11b  Mutual independence of the stream makes each rotation fresh

The remaining structural fact is that the pass-`p` rotation is independent of *everything the
recursion has already used*.  Stated for the family `(W, R_0, R_1, …)` indexed by `Option ℕ`
(`none` = the weight row), this is a disjoint-index-sets consequence of mutual independence.

The `Option ℕ`-indexed family is heterogeneous (`E` at `none`, `G` elsewhere), which forces `E`
and `G` into a common universe here; nothing downstream of this subsection needs that, because
`indepFun_rot_past_of_iIndepFun`'s *conclusion* is universe-polymorphic and is the only thing
the assembly consumes. -/

section BridgeStream

universe u

variable {E : Type u} [MeasurableSpace E] {G : Type u} [MeasurableSpace G]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
