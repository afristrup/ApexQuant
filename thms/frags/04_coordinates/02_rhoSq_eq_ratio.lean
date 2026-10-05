
/-- The second equality of eq. (2.3): `d · E[(t-q t)²] = E[(t-q t)²] / E[t²]` once `E[t²] = 1/d`.
The paper derives `E_{f_d}[t²] = 1/d` from `∑ᵢ E[u_i²] = 1`, i.e. from `‖u‖ = 1` and
exchangeability — not from the closed form of `f_d`. -/
theorem rhoSq_eq_ratio {cb : ℝ → ℝ} {ν : Measure ℝ} (hd : d ≠ 0)
    (hν : ∫ t, t ^ 2 ∂ν = 1 / (d : ℝ)) :
    rhoSq d cb ν = (∫ t, (t - cb t) ^ 2 ∂ν) / (∫ t, t ^ 2 ∂ν) := by
  have hd' : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hd
  rw [rhoSq, hν]
  field_simp
