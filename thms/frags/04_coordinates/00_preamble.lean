/-! ### §3b  Where `ρ²` comes from: the coordinate layer

`distortion` above is an assumption about vectors.  Here it is discharged from the paper's
actual hypothesis — that the `d` coordinates of `u_p` share a common marginal law `ν` — by
linearity of expectation.  The paper is careful on this point and the formalisation confirms
it: *the coordinates need not be independent, only identically distributed*. -/

section Coordinates

variable {d : ℕ} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
