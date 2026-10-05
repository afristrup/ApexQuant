
/-! ### §4b  The coordinate law, and the link to `ρ`

The audit of the first draft observed, correctly, that `ν` in §3b/§6 and `sphericalPdf` in §4
were formally unrelated symbols that the prose silently identifies.  `sphericalCoordLaw d` is
that identification made explicit: the probability measure on `ℝ` whose density is `f_d`.  With
it, `rhoSq_eq_ratio` applies to the paper's actual `ρ` with no side hypothesis. -/

/-- The coordinate law claimed by Lemma 1: the measure on `ℝ` with density `f_d` on `[-1,1]`. -/
noncomputable def sphericalCoordLaw (d : ℕ) : Measure ℝ :=
  (volume.restrict (Set.Icc (-1 : ℝ) 1)).withDensity fun t => ENNReal.ofReal (sphericalPdf d t)
