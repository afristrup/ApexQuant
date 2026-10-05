
/-- The distinguished unit vector `(1, 0)` of the model space. -/
noncomputable def modelVec (n : ℕ) : modelSpace n := WithLp.toLp 2 (1, 0)
