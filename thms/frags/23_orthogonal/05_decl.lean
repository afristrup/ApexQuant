
instance : SecondCountableTopology (Od d) :=
  inferInstanceAs (SecondCountableTopology
    ((Matrix.orthogonalGroup (Fin d) ℝ : Submonoid _) : Set (Matrix (Fin d) (Fin d) ℝ)))
