
/-- **`w - ŵ_P = r_P`**: the residual after `P` passes *is* the reconstruction error, so the
error of the `P`-pass model is exactly the object Proposition 2 bounds. -/
theorem sub_recon (P : ℕ) : w - recon S w P = resid S w P := aux_sub_recon S w P

/-! ### §2  Nested decoding: the any-prefix property

The paper (§7) claims: *"pass `p` depends only on the residual of passes `< p`, so the first `P'`
passes of a `P`-pass artifact are, bit for bit, the `P'`-pass quantisation."*

The substantive half is `code_stable_of_rot_eq`: the pass-`p` code is unchanged by any
modification of the rotation stream at indices `≥ p`.  `encode_take` is then the literal
prefix statement. -/

/-- The `P`-pass artifact: the list of codes actually written to storage. -/
noncomputable def encode (P : ℕ) : List (ℝ × E) := (List.range P).map (code S w)

/-- Passes are determined by the rotations that precede them: if two schemes share the
quantiser and agree on the first `P'` rotations, their residuals agree up to pass `P'`. -/
theorem resid_stable_of_rot_eq {S₁ S₂ : Scheme E} (hq : S₁.q = S₂.q) {P' : ℕ}
    (hrot : ∀ p < P', S₁.rot p = S₂.rot p) (w : E) :
    ∀ p ≤ P', resid S₁ w p = resid S₂ w p := by
  intro p
  induction p with
  | zero => intro _; rfl
  | succ n ih =>
    intro hn
    have hn' : n < P' := hn
    have ihn := ih (Nat.le_of_lt hn')
    simp only [resid_succ, unitVec, ihn, hrot n hn', hq]

/-- …and therefore emit the same codes on the shared prefix. -/
theorem code_stable_of_rot_eq {S₁ S₂ : Scheme E} (hq : S₁.q = S₂.q) {P' : ℕ}
    (hrot : ∀ p < P', S₁.rot p = S₂.rot p) (w : E) :
    ∀ p < P', code S₁ w p = code S₂ w p := by
  intro p hp
  have hr := resid_stable_of_rot_eq hq hrot w p (Nat.le_of_lt hp)
  simp only [code, unitVec, hr, hrot p hp, hq]

/-- **Any-prefix property.**  The first `P'` entries of the `P`-pass artifact *are* the
`P'`-pass artifact, for every `P' ≤ P`.  One stored object serves every operating point
`P' ≤ P` with no re-quantisation. -/
theorem encode_take {P' P : ℕ} (h : P' ≤ P) :
    (encode S w P).take P' = encode S w P' := by
  simp only [encode, ← List.map_take, List.take_range, Nat.min_eq_left h]

/-- The prefix reconstruction is likewise the `P'`-pass reconstruction, with reconstruction
error exactly `r_{P'}`, so every prefix is itself an honest `P'`-pass quantised model
(the sense in which Corollary 1 applies to each tier). -/
theorem recon_prefix (P' : ℕ) : w - recon S w P' = resid S w P' := sub_recon S w P'

/-- **The paper's nesting claim, composed.**  The first `P'` codes of a `P`-pass artifact are
identical to the artifact of an *independently run* `P'`-pass quantisation under any scheme with
the same codebook and the same first `P'` rotations.  This is the "bit for bit, the `P'`-pass
quantisation" statement the paper verifies empirically; `encode_take` alone only re-reads one
scheme's own list and is pure list algebra. -/
theorem encode_take_eq {S₁ S₂ : Scheme E} (hq : S₁.q = S₂.q) {P' P : ℕ} (hP : P' ≤ P)
    (hrot : ∀ p < P', S₁.rot p = S₂.rot p) (w : E) :
    (encode S₁ w P).take P' = encode S₂ w P' := by
  rw [encode_take S₁ w hP]
  simp only [encode]
  refine List.map_congr_left ?_
  intro p hp
  exact code_stable_of_rot_eq hq hrot w p (List.mem_range.mp hp)

end Deterministic

