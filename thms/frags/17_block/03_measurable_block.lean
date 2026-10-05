theorem measurable_block (j : Fin m) : Measurable (fun v : BlockSpace m F => v j) :=
  (measurable_pi_apply j).comp (WithLp.measurable_ofLp 2 _)

omit [NormedAddCommGroup F] in
