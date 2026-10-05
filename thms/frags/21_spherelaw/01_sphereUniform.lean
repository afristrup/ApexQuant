
/-- The uniform probability measure on the unit sphere of `E`: normalised surface measure. -/
noncomputable def sphereUniform : Measure (sphere (0 : E) 1) :=
  (volume.toSphere (Set.univ : Set (sphere (0 : E) 1)))⁻¹ • volume.toSphere
