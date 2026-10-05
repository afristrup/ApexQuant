
/-- The model space `ℝ × ℝⁿ` with the Euclidean norm. -/
abbrev modelSpace (n : ℕ) := WithLp 2 (ℝ × EuclideanSpace ℝ (Fin n))
