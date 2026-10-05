
instance : CompactSpace (Od d) := by
  refine isCompact_iff_compactSpace.mp
    (IsCompact.of_isClosed_subset (isCompact_box d) (isClosed_orthogonalGroup d) ?_)
  intro A hA i _ j _
  exact entry_mem_Icc d ⟨A, hA⟩ i j
