
theorem measurable_blockDensity (m : ℕ) : Measurable (blockDensity (U := U) m) := by
  unfold blockDensity; fun_prop
