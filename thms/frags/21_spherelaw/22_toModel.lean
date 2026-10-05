
/-- An isometry from `E` onto the model space of the same dimension. -/
noncomputable def toModel (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [FiniteDimensional ℝ E] (hd : 1 ≤ Module.finrank ℝ E) :
    E ≃ₗᵢ[ℝ] modelSpace (Module.finrank ℝ E - 1) :=
  (stdOrthonormalBasis ℝ E).repr.trans
    ((LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
      (finCongr (by rw [finrank_modelSpace]; omega))).trans
        (stdOrthonormalBasis ℝ (modelSpace (Module.finrank ℝ E - 1))).repr.symm)
