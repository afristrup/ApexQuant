/-! ## §4  Lemma 1 (spherical marginal) and the data-free evaluation of `ρ`

Lemma 1 states that a coordinate of a uniform point on `S^{d-1}` has density

  `f_d(t) = Γ(d/2) / (√π Γ((d-1)/2)) · (1 - t²)^((d-3)/2)`  on `[-1,1]`,

equivalently `(u_i + 1)/2 ∼ Beta((d-1)/2, (d-1)/2)`, with mean `0` and variance `1/d`.

What §3 needs from this is only `∫ t² f_d = 1/d`, which §3b already derives for free.  What
Lemma 1 is genuinely *for* is the paper's data-free design chart: it makes `ρ` a
one-dimensional quadrature that can be evaluated before any weight is touched.  The statements
below certify that `f_d` really is a probability density with the advertised moments. -/

section Marginal

variable {d : ℕ}
