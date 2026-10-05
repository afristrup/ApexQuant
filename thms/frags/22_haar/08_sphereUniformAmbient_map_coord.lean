
/-- Lemma 1 for the coordinates of `ℝ^d`: every coordinate of the uniform direction has law
`sphericalCoordLaw d`. -/
theorem sphereUniformAmbient_map_coord {d : ℕ} (hd : 2 ≤ d) (i : Fin d) :
    (sphereUniformAmbient (EuclideanSpace ℝ (Fin d))).map (fun x => x i) = sphericalCoordLaw d := by
  have h := sphereUniformAmbient_map_inner (E := EuclideanSpace ℝ (Fin d))
    (by rw [finrank_euclideanSpace_fin]; exact hd) (e := EuclideanSpace.single i (1 : ℝ))
    (by rw [PiLp.norm_single 2 _, norm_one])
  rw [finrank_euclideanSpace_fin] at h
  convert h using 2
  funext x
  rw [EuclideanSpace.inner_single_left]
  simp

end Coordinates
