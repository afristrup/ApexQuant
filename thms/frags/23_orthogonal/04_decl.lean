
instance : SecondCountableTopology (Matrix (Fin d) (Fin d) ℝ) :=
  inferInstanceAs (SecondCountableTopology (Fin d → Fin d → ℝ))
