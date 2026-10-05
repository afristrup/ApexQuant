
/-- The stage that applies one block quantiser `qB` to every block. -/
def blockStage (qB : F → F) (v : BlockSpace m F) : BlockSpace m F :=
  WithLp.toLp 2 fun j => qB (v j)

omit [NormedAddCommGroup F] in
