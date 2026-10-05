
theorem measurable_sphere_fst :
    Measurable fun u : sphere (0 : WithLp 2 (U × V)) 1 => (WithLp.ofLp (u : WithLp 2 (U × V))).1 :=
  measurable_fst.comp ((WithLp.measurable_ofLp 2 (U × V)).comp measurable_subtype_coe)
