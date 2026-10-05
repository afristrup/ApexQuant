
/-- The second moment of the coordinate law is `1/d`, matching what `second_moment_eq_inv_card`
of §3b derives from `‖u‖ = 1` alone.  That these two routes agree is the only place where
Lemma 1 and Proposition 2 make contact. -/
theorem integral_sq_sphericalCoordLaw (hd : 2 ≤ d) :
    ∫ t, t ^ 2 ∂(sphericalCoordLaw d) = 1 / (d : ℝ) := by
  rw [sphericalCoordLaw,
    integral_withDensity_eq_integral_toReal_smul
      (f := fun t => ENNReal.ofReal (sphericalPdf d t))
      (ENNReal.measurable_ofReal.comp (aux_meas_pdf d))
      (Filter.Eventually.of_forall fun t => ENNReal.ofReal_lt_top)]
  rw [show (∫ t, (ENNReal.ofReal (sphericalPdf d t)).toReal • t ^ 2
        ∂(volume.restrict (Set.Icc (-1:ℝ) 1)))
      = ∫ t, t ^ 2 * sphericalPdf d t ∂(volume.restrict (Set.Icc (-1:ℝ) 1)) from
    setIntegral_congr_fun measurableSet_Icc fun t ht => by
      rw [ENNReal.toReal_ofReal (sphericalPdf_nonneg ht hd), smul_eq_mul, mul_comm]]
  exact aux_integral_sq_pdf hd

end Marginal

