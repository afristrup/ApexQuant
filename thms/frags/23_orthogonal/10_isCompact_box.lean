
theorem isCompact_box :
    IsCompact (Set.pi Set.univ fun _ : Fin d => Set.pi Set.univ fun _ : Fin d => Set.Icc (-1 : ℝ) 1 :
      Set (Matrix (Fin d) (Fin d) ℝ)) :=
  isCompact_univ_pi fun _ => isCompact_univ_pi fun _ => isCompact_Icc
