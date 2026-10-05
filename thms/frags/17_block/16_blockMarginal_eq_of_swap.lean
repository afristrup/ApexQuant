/-- If some rotation carries block `j'` to block `j`, the two block marginals of the direction
law coincide.  With all block swaps available (they are in `O(d)`), every block has the same
law, which is the `hmarg` hypothesis of `stageDistortion_blockStage`. -/
theorem blockMarginal_eq_of_swap (κ : Measure G) [κ.IsMulLeftInvariant] (x₀ : BlockSpace m F)
    {j j' : Fin m} {h : G} (hswap : ∀ v : BlockSpace m F, (h • v) j = v j') :
    (orbitLaw κ x₀).map (fun v => v j) = (orbitLaw κ x₀).map (fun v => v j') := by
  conv_lhs => rw [← orbitLaw_map_smul κ h x₀]
  rw [Measure.map_map (measurable_block j) (measurable_const_smul h)]
  congr 1
  funext v
  exact hswap v

end BlockExchange
