/-! ## §6  Corollary 1 and the design rule

Corollary 1: since every pass sees the same marginal, the codebook minimising the per-coordinate
MSE minimises it at *every* pass, hence minimises `ρ`, hence minimises the whole envelope `ρ^P`.

The formal content is a monotonicity statement, trivial, and worth stating precisely for that
reason: the corollary is a consequence of Theorem 1, not an independent optimality theorem, and
its scope is *scalar quantisers applied coordinate-wise*. -/

section Optimality

variable {d : ℕ} {ν : Measure ℝ}
