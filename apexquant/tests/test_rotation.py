import torch

from apexquant.rotation import (
    inverse_random_rotation,
    random_rotation,
    walsh_hadamard_transform,
)
from apexquant.rotation.random_sign_vector import random_sign_vector


def test_wht_is_orthonormal_self_inverse():
    x = torch.randn(3, 128)
    y = walsh_hadamard_transform(x)
    # Orthonormal: preserves L2 norm.
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    # Self-inverse.
    assert torch.allclose(walsh_hadamard_transform(y), x, atol=1e-4)


def test_random_rotation_preserves_norm_and_inverts():
    x = torch.randn(4, 5, 64)
    r = random_rotation(x, seed=7)
    assert torch.allclose(r.norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    assert torch.allclose(inverse_random_rotation(r, seed=7), x, atol=1e-4)


def test_different_seeds_give_different_rotations():
    x = torch.randn(64)
    assert not torch.allclose(random_rotation(x, 1), random_rotation(x, 2), atol=1e-3)


def test_matmul_rotation_matches_butterfly():
    # random_rotation uses a cached matmul for small groups; it must equal the
    # explicit butterfly (sign flip + WHT) it replaces, up to floating point.
    for d in (64, 128, 512):
        x = torch.randn(9, 3, d)
        signs = random_sign_vector(d, 7, device=x.device)
        butterfly = walsh_hadamard_transform(x * signs)
        assert torch.allclose(random_rotation(x, 7), butterfly, atol=1e-5)
