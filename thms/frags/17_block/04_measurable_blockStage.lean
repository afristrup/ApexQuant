theorem measurable_blockStage {qB : F → F} (hq : Measurable qB) :
    Measurable (blockStage (m := m) qB) :=
  (WithLp.measurable_toLp 2 _).comp (measurable_pi_lambda _ fun j => hq.comp (measurable_block j))
