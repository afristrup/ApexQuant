
/-- The *code* emitted at pass `p`: the scale `s_p` (stored as one fp16 number per group) and
the codeword `q(u_p)` (stored as `d` `b`-bit indices).  This pair is the entire artifact of
pass `p`. -/
noncomputable def code (p : ℕ) : ℝ × E := (‖resid S w p‖, S.q (unitVec S w p))
