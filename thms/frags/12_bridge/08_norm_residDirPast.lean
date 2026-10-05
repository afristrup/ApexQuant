
/-- The normalised residual is a unit vector whenever the pass is not already exact. -/
theorem norm_residDirPast (hι : ∀ (g : G) (x : E), ι g x = g • x) {q : E → E} {w : E}
    {g : ℕ → G} {p : ℕ} (h : resid (rotScheme ι q g) w p ≠ 0) :
    ‖residDirPast q p (w, fun i : Fin p => g i)‖ = 1 := by
  rw [resid_rotScheme_eq_residPast hι] at h
  rw [residDirPast, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ (norm_ne_zero_iff.mpr h)]

end BridgeAlgebra

section BridgeMeasurable

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
variable {G : Type*} [Group G] [MeasurableSpace G] [MulAction G E] [MeasurableSMul₂ G E]
  [MeasurableInv G]
