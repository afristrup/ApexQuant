"""group_tau: the per-group mean input variance, tr(Cov x)/d and not tr(E[x x^T])/d."""

import pytest
import torch

from apexquant.bn import group_tau

CHANNELS = 8
KERNEL_AREA = 9                 # a 3x3 convolution, so in(j) = j // 9
GROUP_SIZE = 8
COLUMNS = CHANNELS * KERNEL_AREA
SAMPLE_DRAWS = 500000


def _conv_column_map(channels=CHANNELS, kernel_area=KERNEL_AREA):
    return torch.arange(channels * kernel_area) // kernel_area


def test_matches_an_explicit_per_group_mean():
    torch.manual_seed(0)
    e = torch.randn(CHANNELS)
    s = e.pow(2) + torch.rand(CHANNELS) + 0.1
    in_index = _conv_column_map()

    tau = group_tau(e, s, in_index, GROUP_SIZE)

    variance = s - e.pow(2)
    expected = torch.tensor([
        float(variance[in_index[b * GROUP_SIZE:(b + 1) * GROUP_SIZE]].mean())
        for b in range(COLUMNS // GROUP_SIZE)
    ])
    assert tau.shape == (COLUMNS // GROUP_SIZE,)
    assert torch.allclose(tau, expected, rtol=1e-6, atol=1e-7)


def test_against_monte_carlo_samples_of_a_known_input_distribution():
    """Feed empirical moments in, get the empirical tr(Cov x)/d of that group out."""
    torch.manual_seed(1)
    # Exponential channels: E[x] = 1/rate, E[x^2] = 2/rate^2, Var = 1/rate^2.
    rate = 0.5 + torch.rand(CHANNELS)
    x = torch.distributions.Exponential(rate).sample((SAMPLE_DRAWS,))

    e = x.mean(dim=0)
    s = x.pow(2).mean(dim=0)
    in_index = _conv_column_map()
    tau = group_tau(e, s, in_index, GROUP_SIZE)

    population = 1.0 / rate.pow(2)
    expected = torch.tensor([
        float(population[in_index[b * GROUP_SIZE:(b + 1) * GROUP_SIZE]].mean())
        for b in range(COLUMNS // GROUP_SIZE)
    ])
    assert torch.allclose(tau, expected, rtol=0.02, atol=1e-3)

    # Same thing said the other way: it is the sample variance of the columns.
    sample_variance = x.var(dim=0, unbiased=False)
    per_column = sample_variance[in_index].view(-1, GROUP_SIZE).mean(dim=1)
    assert torch.allclose(tau, per_column, rtol=1e-4, atol=1e-5)


def test_second_moment_would_double_count_the_mean():
    """The document's ResNet-50 warning, reproduced on a rectified Gaussian."""
    torch.manual_seed(2)
    m, sigma = 0.3, 1.0
    x = torch.relu(m + sigma * torch.randn(SAMPLE_DRAWS, CHANNELS))
    e = x.mean(dim=0)
    s = x.pow(2).mean(dim=0)
    in_index = _conv_column_map()

    tau = group_tau(e, s, in_index, GROUP_SIZE)
    inflated = s[in_index].view(-1, GROUP_SIZE).mean(dim=1)   # tr(E[x x^T])/d

    assert float(tau.max()) < float(inflated.min())
    assert torch.allclose(inflated - tau, e.pow(2)[in_index].view(-1, GROUP_SIZE).mean(dim=1),
                          rtol=1e-4, atol=1e-6)


def test_zero_mean_channels_make_the_two_forms_agree():
    torch.manual_seed(3)
    s = torch.rand(CHANNELS) + 0.1
    e = torch.zeros(CHANNELS)
    in_index = _conv_column_map()

    tau = group_tau(e, s, in_index, GROUP_SIZE)
    assert torch.allclose(tau, s[in_index].view(-1, GROUP_SIZE).mean(dim=1), rtol=1e-6, atol=1e-7)


def test_rejects_a_variance_passed_where_the_second_moment_belongs():
    torch.manual_seed(4)
    e = 2.0 + torch.rand(CHANNELS)
    variance = torch.rand(CHANNELS) * 0.1
    with pytest.raises(ValueError, match="second moment"):
        group_tau(e, variance, _conv_column_map(), GROUP_SIZE)


def test_fails_fast_on_a_bad_call():
    e = torch.zeros(CHANNELS)
    s = torch.ones(CHANNELS)
    in_index = _conv_column_map()

    with pytest.raises(ValueError, match="1-D and equal"):
        group_tau(e, s[:-1], in_index, GROUP_SIZE)
    with pytest.raises(ValueError, match="multiple of group_size"):
        group_tau(e, s, in_index[:-1], GROUP_SIZE)
    with pytest.raises(ValueError, match="outside e"):
        group_tau(e, s, in_index + CHANNELS, GROUP_SIZE)
    with pytest.raises(ValueError, match="integer tensor"):
        group_tau(e, s, in_index.float(), GROUP_SIZE)
