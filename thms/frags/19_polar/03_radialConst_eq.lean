
theorem radialConst_eq (n : ℕ) : radialConst n = ENNReal.ofReal (1 / (n + 1)) := by
  have h1 : radialConst n = ∫⁻ r in Set.Ioo (0 : ℝ) 1, ENNReal.ofReal (r ^ n) := by
    unfold radialConst
    rw [setLIntegral_congr_fun measurableSet_Ioi (g := (Set.Iio (1 : ℝ)).indicator
        fun r => ENNReal.ofReal (r ^ n)) (fun r _ => by
          by_cases h : r < 1 <;> simp [Set.indicator, h]),
      lintegral_indicator measurableSet_Iio, Measure.restrict_restrict measurableSet_Iio,
      Set.inter_comm, Set.Ioi_inter_Iio]
  have h2 : ∫⁻ r in Set.Ioo (0 : ℝ) 1, ENNReal.ofReal (r ^ n) = ENNReal.ofReal (1 / (n + 1)) := by
    rw [← ofReal_integral_eq_lintegral_ofReal]
    · rw [← integral_Ioc_eq_integral_Ioo, ← intervalIntegral.integral_of_le zero_le_one,
        integral_pow]
      simp
    · exact (intervalIntegral.intervalIntegrable_pow n).1.mono_set Ioo_subset_Ioc_self
    · exact (ae_restrict_mem measurableSet_Ioo).mono fun r hr => pow_nonneg hr.1.le n
  rw [h1, h2]

end Radial

section Polar2D
