/-! ## §28  Right derivatives of finite minima

The squared distance to a finite codebook is a minimum of finitely many smooth functions, so it
is not differentiable but has a right derivative everywhere, equal to the derivative of an
active branch (`exists_hasDerivWithinAt_finset_inf'`).  The fundamental theorem of calculus for
right derivatives (`intervalIntegral.integral_eq_sub_of_hasDeriv_right_of_le`) then gives the
path formula `g(b) − g(a) = ∫ₐᵇ g'₊` (`finset_inf'_sub_eq_integral`). -/

section MinDeriv

/-- The right derivative of a minimum of two right-differentiable functions at a tie point is
the minimum of the derivatives. -/
theorem hasDerivWithinAt_min_Ioi_of_eq {g₁ g₂ : ℝ → ℝ} {θ d₁ d₂ : ℝ}
    (h₁ : HasDerivWithinAt g₁ d₁ (Ioi θ) θ) (h₂ : HasDerivWithinAt g₂ d₂ (Ioi θ) θ)
    (htie : g₁ θ = g₂ θ) :
    HasDerivWithinAt (fun t => min (g₁ t) (g₂ t)) (min d₁ d₂) (Ioi θ) θ := by
  rw [hasDerivWithinAt_iff_tendsto_slope] at h₁ h₂ ⊢
  refine (h₁.min h₂).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with t ht
  have hpos : 0 < t - θ := by
    have : t ∈ Ioi θ := ht.1
    exact sub_pos.mpr this
  simp only [slope_def_field]
  rw [min_def (g₁ θ) (g₂ θ), if_pos htie.le, min_div_div_right hpos.le, ← htie, min_sub_sub_right]

/-- If `g₁ θ < g₂ θ` and both are continuous from the right, the minimum agrees with `g₁` near
`θ` on the right. -/
theorem hasDerivWithinAt_min_Ioi_of_lt {g₁ g₂ : ℝ → ℝ} {θ d₁ d₂ : ℝ}
    (h₁ : HasDerivWithinAt g₁ d₁ (Ioi θ) θ) (h₂ : HasDerivWithinAt g₂ d₂ (Ioi θ) θ)
    (hlt : g₁ θ < g₂ θ) :
    HasDerivWithinAt (fun t => min (g₁ t) (g₂ t)) d₁ (Ioi θ) θ := by
  have hc : ∀ᶠ t in 𝓝[Ioi θ] θ, g₁ t < g₂ t :=
    (h₁.continuousWithinAt.eventually_lt h₂.continuousWithinAt hlt)
  refine h₁.congr_of_eventuallyEq ?_ (by simp [min_eq_left hlt.le])
  filter_upwards [hc] with t ht
  simp [min_eq_left ht.le]

/-- **Right derivative of a finite minimum.**  If each `φ i` has right derivative `φ' i` at `θ`,
the minimum over a nonempty finite set has a right derivative, equal to `φ' i` for some active
index `i` (one attaining the minimum at `θ`). -/
theorem exists_hasDerivWithinAt_finset_inf' {ι : Type*} {s : Finset ι}
    (hs : s.Nonempty) {φ : ι → ℝ → ℝ} {φ' : ι → ℝ} {θ : ℝ}
    (h : ∀ i ∈ s, HasDerivWithinAt (φ i) (φ' i) (Ioi θ) θ) :
    ∃ i ∈ s, φ i θ = s.inf' hs (fun j => φ j θ) ∧
      HasDerivWithinAt (fun t => s.inf' hs (fun j => φ j t)) (φ' i) (Ioi θ) θ := by
  revert h
  induction hs using Finset.Nonempty.cons_induction with
  | singleton a =>
      intro h
      refine ⟨a, Finset.mem_singleton_self a, by simp, ?_⟩
      simpa using h a (Finset.mem_singleton_self a)
  | cons a s ha hs' ih =>
      intro h
      obtain ⟨j, hj, hjeq, hjder⟩ := ih fun i hi => h i (Finset.mem_cons_of_mem hi)
      have hfun : (fun t => (Finset.cons a s ha).inf' ⟨a, Finset.mem_cons_self a s⟩ (fun k => φ k t))
          = fun t => min (φ a t) (s.inf' hs' (fun k => φ k t)) := by
        funext t; rw [Finset.inf'_cons hs']
      have ha' := h a (Finset.mem_cons_self a s)
      rw [hfun]
      rcases lt_trichotomy (φ a θ) (s.inf' hs' (fun k => φ k θ)) with hlt | heq | hgt
      · refine ⟨a, Finset.mem_cons_self a s, ?_, hasDerivWithinAt_min_Ioi_of_lt ha' hjder hlt⟩
        rw [Finset.inf'_cons hs', min_eq_left hlt.le]
      · -- tie: derivative is `min (φ' a) (φ' j)`, which is `φ' a` or `φ' j`
        have hmin := hasDerivWithinAt_min_Ioi_of_eq ha' hjder heq
        rcases le_total (φ' a) (φ' j) with hle | hle
        · refine ⟨a, Finset.mem_cons_self a s, ?_, by rwa [min_eq_left hle] at hmin⟩
          rw [Finset.inf'_cons hs', min_eq_left heq.le]
        · refine ⟨j, Finset.mem_cons_of_mem hj, ?_, by rwa [min_eq_right hle] at hmin⟩
          rw [Finset.inf'_cons hs', min_eq_right heq.ge, hjeq]
      · refine ⟨j, Finset.mem_cons_of_mem hj, ?_, ?_⟩
        · rw [Finset.inf'_cons hs', min_eq_right hgt.le, hjeq]
        · have := hasDerivWithinAt_min_Ioi_of_lt hjder ha' hgt
          exact this.congr_of_eventuallyEq (Filter.Eventually.of_forall fun t => min_comm _ _)
            (min_comm _ _)

/-- A finite minimum of continuous functions is continuous. -/
theorem continuous_finset_inf'_fun {ι : Type*} {s : Finset ι} (hs : s.Nonempty) {φ : ι → ℝ → ℝ}
    (h : ∀ i ∈ s, Continuous (φ i)) : Continuous fun t => s.inf' hs (fun i => φ i t) := by
  revert h
  induction hs using Finset.Nonempty.cons_induction with
  | singleton a => intro h; simpa using h a (Finset.mem_singleton_self a)
  | cons a s ha hs' ih =>
      intro h
      have hfun : (fun t => (Finset.cons a s ha).inf' ⟨a, Finset.mem_cons_self a s⟩ (fun k => φ k t))
          = fun t => min (φ a t) (s.inf' hs' (fun k => φ k t)) := by
        funext t; rw [Finset.inf'_cons hs']
      rw [hfun]
      exact (h a (Finset.mem_cons_self a s)).min (ih fun i hi => h i (Finset.mem_cons_of_mem hi))

/-- The right derivative of a finite minimum of differentiable functions, as a function:
`derivWithin g (Ioi t) t` equals `φ' i t` for some index `i` active at `t`. -/
theorem exists_derivWithin_Ioi_finset_inf' {ι : Type*} {s : Finset ι}
    (hs : s.Nonempty) {φ : ι → ℝ → ℝ} {φ' : ι → ℝ → ℝ}
    (hφ : ∀ i ∈ s, ∀ t, HasDerivAt (φ i) (φ' i t) t) (t : ℝ) :
    ∃ i ∈ s, φ i t = s.inf' hs (fun j => φ j t) ∧
      derivWithin (fun u => s.inf' hs (fun j => φ j u)) (Ioi t) t = φ' i t ∧
      HasDerivWithinAt (fun u => s.inf' hs (fun j => φ j u))
        (derivWithin (fun u => s.inf' hs (fun j => φ j u)) (Ioi t) t) (Ioi t) t := by
  obtain ⟨i, hi, heq, hder⟩ := exists_hasDerivWithinAt_finset_inf' hs
    (φ' := fun j => φ' j t) (θ := t) fun j hj => (hφ j hj t).hasDerivWithinAt
  have hd := hder.derivWithin (uniqueDiffWithinAt_Ioi t)
  exact ⟨i, hi, heq, hd, hd ▸ hder⟩

/-- **The path formula for a finite minimum of `C¹` functions**: `g b − g a = ∫ₐᵇ D` with `D`
the right derivative of `g`, and `|D t| ≤ ∑ᵢ |φ' i t|` pointwise. -/
theorem finset_inf'_sub_eq_integral {ι : Type*} {s : Finset ι} (hs : s.Nonempty)
    {φ : ι → ℝ → ℝ} {φ' : ι → ℝ → ℝ} {a b : ℝ} (hab : a ≤ b)
    (hφ : ∀ i ∈ s, ∀ t, HasDerivAt (φ i) (φ' i t) t) (hφ' : ∀ i ∈ s, Continuous (φ' i)) :
    s.inf' hs (fun j => φ j b) - s.inf' hs (fun j => φ j a)
      = ∫ t in a..b, derivWithin (fun u => s.inf' hs (fun j => φ j u)) (Ioi t) t := by
  have hcont : Continuous fun u => s.inf' hs (fun j => φ j u) :=
    continuous_finset_inf'_fun hs fun i hi =>
      continuous_iff_continuousAt.mpr fun t => (hφ i hi t).continuousAt
  have hbound : ∀ t, |derivWithin (fun u => s.inf' hs (fun j => φ j u)) (Ioi t) t|
      ≤ ∑ i ∈ s, |φ' i t| := by
    intro t
    obtain ⟨i, hi, -, hd, -⟩ := exists_derivWithin_Ioi_finset_inf' hs hφ t
    rw [hd]
    exact Finset.single_le_sum (f := fun i => |φ' i t|) (fun i _ => abs_nonneg _) hi
  have hmeas : Measurable fun t => derivWithin (fun u => s.inf' hs (fun j => φ j u)) (Ioi t) t :=
    measurable_derivWithin_Ioi _
  have hint : IntervalIntegrable
      (fun t => derivWithin (fun u => s.inf' hs (fun j => φ j u)) (Ioi t) t) volume a b := by
    have hg : IntervalIntegrable (fun t => ∑ i ∈ s, |φ' i t|) volume a b :=
      (continuous_finsetSum s fun i hi => (hφ' i hi).abs).intervalIntegrable a b
    refine hg.mono_fun hmeas.aestronglyMeasurable (Filter.Eventually.of_forall fun t => ?_)
    show ‖derivWithin (fun u => s.inf' hs (fun j => φ j u)) (Ioi t) t‖ ≤ ‖∑ i ∈ s, |φ' i t|‖
    rw [Real.norm_eq_abs, Real.norm_eq_abs,
      abs_of_nonneg (Finset.sum_nonneg fun i _ => abs_nonneg (φ' i t))]
    exact hbound t
  symm
  exact intervalIntegral.integral_eq_sub_of_hasDeriv_right_of_le hab hcont.continuousOn
    (fun t _ => (exists_derivWithin_Ioi_finset_inf' hs hφ t).choose_spec.2.2.2) hint

end MinDeriv
