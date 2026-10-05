
/-- `m` blocks of `F` with the Euclidean norm across blocks.  `BlockSpace m ℝ` is
`EuclideanSpace ℝ (Fin m)` definitionally. -/
abbrev BlockSpace (m : ℕ) (F : Type*) := PiLp 2 (fun _ : Fin m => F)
