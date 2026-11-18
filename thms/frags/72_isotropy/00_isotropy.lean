
/-! ## §72  Isotropy of the uniform direction: second moments and quadratic forms

Under `sphereUniformAmbient (EuclideanSpace ℝ (Fin d))` with `d ≥ 2` the coordinates satisfy
`E[xᵢ xⱼ] = δᵢⱼ / d`.  The diagonal is the second moment of the coordinate marginal
(`sphereUniformAmbient_map_coord`, `integral_sq_sphericalCoordLaw`); the off-diagonal vanishes
because the law is invariant under the reflection flipping coordinate `i`
(`sphereUniform_map_sphereMap`).  Consequently `E[xᵀ H x] = tr(H) / d` for every matrix `H`,
with no hypothesis on second moments. -/

section Isotropy

/-- The ambient uniform law is invariant under every linear isometry equivalence. -/
theorem sphereUniformAmbient_map_isometry {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [MeasurableSpace E] [BorelSpace E] [FiniteDimensional ℝ E]
    (L : E ≃ₗᵢ[ℝ] E) :
    (sphereUniformAmbient E).map L = sphereUniformAmbient E := by
  have h1 : (L : E → E) ∘ (Subtype.val : Metric.sphere (0 : E) 1 → E)
      = Subtype.val ∘ sphereMap L := by
    funext u; simp
  rw [sphereUniformAmbient, Measure.map_map L.continuous.measurable measurable_subtype_coe, h1,
    ← Measure.map_map measurable_subtype_coe (measurable_sphereMap _), sphereUniform_map_sphereMap]

/-- The reflection of `ℝ^d` that flips the sign of coordinate `i`. -/
noncomputable def coordFlip {d : ℕ} (i : Fin d) :
    EuclideanSpace ℝ (Fin d) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin d) :=
  LinearIsometryEquiv.piLpCongrRight 2 fun k =>
    if k = i then LinearIsometryEquiv.neg ℝ else LinearIsometryEquiv.refl ℝ ℝ

theorem coordFlip_apply_self {d : ℕ} (i : Fin d) (x : EuclideanSpace ℝ (Fin d)) :
    coordFlip i x i = -x i := by
  simp [coordFlip]

theorem coordFlip_apply_of_ne {d : ℕ} {i j : Fin d} (h : j ≠ i)
    (x : EuclideanSpace ℝ (Fin d)) : coordFlip i x j = x j := by
  simp [coordFlip, h]

/-- Products of coordinates are integrable against the uniform direction law. -/
theorem integrable_coord_mul_coord_sphereUniformAmbient {d : ℕ} (hd : 2 ≤ d) (i j : Fin d) :
    Integrable (fun x : EuclideanSpace ℝ (Fin d) => x i * x j)
      (sphereUniformAmbient (EuclideanSpace ℝ (Fin d))) := by
  haveI : Nontrivial (EuclideanSpace ℝ (Fin d)) :=
    Module.nontrivial_of_finrank_pos (R := ℝ) (by rw [finrank_euclideanSpace_fin]; omega)
  have hmeas : Measurable fun x : EuclideanSpace ℝ (Fin d) => x i * x j := by fun_prop
  refine Integrable.of_bound hmeas.aestronglyMeasurable 1 ?_
  filter_upwards [sphereUniformAmbient_ae_norm (E := EuclideanSpace ℝ (Fin d))] with x hx
  have hi : ‖x i‖ ≤ 1 := hx ▸ PiLp.norm_apply_le x i
  have hj : ‖x j‖ ≤ 1 := hx ▸ PiLp.norm_apply_le x j
  rw [norm_mul]
  exact mul_le_one₀ hi (norm_nonneg _) hj

/-- **Isotropy of the uniform direction.**  `E[xᵢ xⱼ] = δᵢⱼ / d`. -/
theorem integral_coord_mul_coord_sphereUniformAmbient {d : ℕ} (hd : 2 ≤ d) (i j : Fin d) :
    ∫ x : EuclideanSpace ℝ (Fin d), x i * x j
        ∂(sphereUniformAmbient (EuclideanSpace ℝ (Fin d)))
      = if i = j then 1 / (d : ℝ) else 0 := by
  split_ifs with hij
  · subst hij
    have hmeas : Measurable fun x : EuclideanSpace ℝ (Fin d) => x i := by fun_prop
    have h := integral_sq_sphericalCoordLaw hd
    rw [← sphereUniformAmbient_map_coord hd i,
      integral_map hmeas.aemeasurable (by fun_prop)] at h
    rw [← h]
    exact integral_congr_ae (Filter.Eventually.of_forall fun x => by ring)
  · have hji : j ≠ i := fun h => hij h.symm
    have hinv := sphereUniformAmbient_map_isometry (coordFlip i)
    have hmeas : Measurable fun x : EuclideanSpace ℝ (Fin d) => x i * x j := by fun_prop
    have h : ∫ x : EuclideanSpace ℝ (Fin d), x i * x j
          ∂(sphereUniformAmbient (EuclideanSpace ℝ (Fin d)))
        = ∫ x : EuclideanSpace ℝ (Fin d), coordFlip i x i * coordFlip i x j
          ∂(sphereUniformAmbient (EuclideanSpace ℝ (Fin d))) := by
      conv_lhs => rw [← hinv]
      exact integral_map (coordFlip i).continuous.measurable.aemeasurable
        hmeas.aestronglyMeasurable
    simp_rw [coordFlip_apply_self, coordFlip_apply_of_ne hji, neg_mul] at h
    rw [integral_neg] at h
    linarith

/-- **The expected quadratic form of a uniform direction is `tr(H) / d`**, for every matrix
`H`, with no hypothesis on second moments. -/
theorem integral_quadraticForm_sphereUniformAmbient {d : ℕ} (hd : 2 ≤ d)
    (H : Matrix (Fin d) (Fin d) ℝ) :
    ∫ x : EuclideanSpace ℝ (Fin d), ∑ i, ∑ j, x i * H i j * x j
        ∂(sphereUniformAmbient (EuclideanSpace ℝ (Fin d)))
      = Matrix.trace H / (d : ℝ) := by
  have hint := integrable_coord_mul_coord_sphereUniformAmbient hd
  have hq : (fun x : EuclideanSpace ℝ (Fin d) => ∑ i, ∑ j, x i * H i j * x j) =
      fun x => ∑ i, ∑ j, H i j * (x i * x j) := by
    funext x
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
  rw [hq, integral_finsetSum _ fun i _ =>
    integrable_finsetSum _ fun j _ => (hint i j).const_mul (H i j)]
  simp_rw [integral_finsetSum _ fun j _ => (hint _ j).const_mul (H _ j),
    integral_const_mul, integral_coord_mul_coord_sphereUniformAmbient hd]
  simp [Matrix.trace, Finset.sum_mul, div_eq_mul_inv]

/-- The same statement in `Matrix.dotProduct` form: `E[x ⬝ᵥ H *ᵥ x] = tr(H) / d`. -/
theorem integral_dotProduct_mulVec_sphereUniformAmbient {d : ℕ} (hd : 2 ≤ d)
    (H : Matrix (Fin d) (Fin d) ℝ) :
    ∫ x : EuclideanSpace ℝ (Fin d), (WithLp.ofLp x) ⬝ᵥ (H.mulVec (WithLp.ofLp x))
        ∂(sphereUniformAmbient (EuclideanSpace ℝ (Fin d)))
      = Matrix.trace H / (d : ℝ) := by
  rw [← integral_quadraticForm_sphereUniformAmbient hd H]
  refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
  simp only [dotProduct, Matrix.mulVec, Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

end Isotropy
