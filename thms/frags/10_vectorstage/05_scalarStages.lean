
/-- The coordinate-wise scalar quantisers built from a set `C` of scalar codebooks.  This is the
class Corollary 1 optimises over; `sInf_stageDistortion_le_of_subset` applies to any class
containing it. -/
def scalarStages (d : ℕ) (C : Set (ℝ → ℝ)) :
    Set (EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d)) :=
  {Q | ∃ cb ∈ C, Q = fun v => WithLp.toLp 2 fun i => cb (v i)}
