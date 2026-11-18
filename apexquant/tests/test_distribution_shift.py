import torch

from apexquant.analysis import distribution_shift


def test_identical_logits_have_zero_shift():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    metrics = distribution_shift(logits, logits, logits=True)

    assert metrics["js_divergence"] == 0.0
    assert metrics["total_variation"] == 0.0
    assert metrics["top1_agreement"] == 1.0


def test_logit_shift_reports_distribution_change():
    reference = torch.tensor([[8.0, 0.0], [8.0, 0.0]])
    candidate = torch.tensor([[0.0, 8.0], [8.0, 0.0]])
    metrics = distribution_shift(reference, candidate, logits=True)

    assert metrics["js_divergence"] > 0.0
    assert 0.0 < metrics["top1_agreement"] < 1.0


def test_activation_shift_reports_moments():
    reference = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    candidate = reference + 0.25
    metrics = distribution_shift(reference, candidate)

    assert metrics["mean_abs_delta"] == 0.25
    assert metrics["relative_mean_shift"] > 0.0


def test_domain_protocol_uses_measurement_batches_only():
    class Model(torch.nn.Module):
        def __init__(self, delta):
            super().__init__()
            self.delta = delta

        def forward(self, batch):
            return batch + self.delta

    batches = [torch.zeros(2, 3), torch.ones(1, 3)]
    metrics = __import__("apexquant").measure_domain_shift(
        Model(0.0), Model(0.25), batches,
        forward=lambda model, batch: model(batch), logits=False)

    assert metrics["mean_abs_delta"] == 0.25
