from apexquant import ApexQuantConfig, linear_is_fusable, resolve_group_size


def test_resolve_group_size_shrinks_narrow_layers():
    assert resolve_group_size(256, 128) == 128   # wide: unchanged
    assert resolve_group_size(128, 128) == 128
    assert resolve_group_size(64, 128) == 64      # narrow: largest pow2 <= in
    assert resolve_group_size(9, 128) == 8
    assert resolve_group_size(3, 128) == 2        # floor is 2


def test_linear_is_fusable_policies():
    cfg = ApexQuantConfig()  # group_size=128, bits=4
    # default policy: only >= group_size
    assert linear_is_fusable(128, cfg) and not linear_is_fusable(64, cfg)
    # full coverage: narrow layers become fusable...
    assert linear_is_fusable(64, cfg, fuse_narrow=True)
    assert linear_is_fusable(9, cfg, fuse_narrow=True)
    # ...but a 2-coord row cannot pack 4 two-bit indices per byte.
    cfg2 = ApexQuantConfig(bits=2)
    assert not linear_is_fusable(2, cfg2, fuse_narrow=True)
    assert linear_is_fusable(64, cfg2, fuse_narrow=True)
