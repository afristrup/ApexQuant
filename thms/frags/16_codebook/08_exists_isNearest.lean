/-- Every finite nonempty codebook has a nearest-codeword quantiser. -/
theorem exists_isNearest (hC : C.Finite) (hne : C.Nonempty) : ∃ Q : E → E, IsNearest C Q := by
  choose Q hQ using fun x => Set.exists_min_image C (fun c => ‖x - c‖) hC hne
  exact ⟨Q, hQ⟩

end Codebook
