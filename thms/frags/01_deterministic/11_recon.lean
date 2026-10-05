
/-! ### Reconstruction and the telescoping identity -/

/-- `recon S w P = ŵ_P = ∑_{p<P} s_p R_pᵀ q(u_p)`, the `P`-pass reconstruction. -/
noncomputable def recon (P : ℕ) : E :=
  ∑ p ∈ range P, ‖resid S w p‖ • (S.rot p).symm (S.q (unitVec S w p))
