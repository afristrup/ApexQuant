
/-- The law of `g • x` when `g ∼ κ`. -/
noncomputable def orbitLaw (κ : Measure G) (x : X) : Measure X := κ.map (fun g => g • x)
