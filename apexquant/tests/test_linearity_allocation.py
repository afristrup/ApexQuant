"""Tests for the linearity-theorem coefficients and the pass allocator."""

import itertools

import torch
from torch import nn

from apexquant import (
    knapsack_pass_allocation,
    linear_error_coefficients,
    random_token_divergence,
)


class _TinyLM(nn.Module):
    """Small stand-in with an embedding, two linears and a tied-size head."""

    def __init__(self, vocab=64, dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.a = nn.Linear(dim, dim, bias=False)
        self.b = nn.Linear(dim, dim, bias=False)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids):
        h = self.embed(ids)
        return self.head(torch.relu(self.b(torch.relu(self.a(h)))))


# --------------------------------------------------------------------------
# The data-free divergence probe.
# --------------------------------------------------------------------------

def test_divergence_is_zero_against_itself_and_grows_with_perturbation():
    torch.manual_seed(0)
    model = _TinyLM().eval()
    assert random_token_divergence(model, model, 64) < 1e-6

    previous = 0.0
    for scale in (0.05, 0.15, 0.45):
        perturbed = _TinyLM().eval()
        perturbed.load_state_dict(model.state_dict())
        with torch.no_grad():
            perturbed.a.weight.add_(torch.randn_like(perturbed.a.weight) * scale)
        d = random_token_divergence(model, perturbed, 64, seed=1)
        assert d > previous
        previous = d


def test_divergence_is_deterministic_for_a_fixed_seed():
    # The coefficient fit is a least squares over many calls; sampling jitter
    # between them would be fitted as signal.
    torch.manual_seed(0)
    ref, cand = _TinyLM().eval(), _TinyLM().eval()
    a = random_token_divergence(ref, cand, 64, seed=7)
    b = random_token_divergence(ref, cand, 64, seed=7)
    assert a == b


# --------------------------------------------------------------------------
# Coefficient estimation by noise insertion.
# --------------------------------------------------------------------------

def test_coefficients_rank_layers_and_leave_the_model_untouched():
    torch.manual_seed(0)
    reference = _TinyLM().eval()
    model = _TinyLM().eval()
    model.load_state_dict(reference.state_dict())
    before = {k: v.clone() for k, v in model.state_dict().items()}

    def measure(m):
        return random_token_divergence(reference, m, 64, seed=3)

    alphas = linear_error_coefficients(model, measure, levels=(0.05, 0.10, 0.15))

    assert set(alphas) == {"a", "b", "head"}
    assert all(v >= 0.0 for v in alphas.values())
    # Every perturbation must be undone, or later layers are measured against a
    # silently damaged model and the whole sweep is wrong.
    after = model.state_dict()
    for k, v in before.items():
        assert torch.equal(after[k], v)


def test_coefficients_detect_a_layer_that_does_not_matter():
    # A layer whose output is discarded cannot affect the model, so its
    # coefficient must come out at (clamped) zero while a live layer's does not.
    torch.manual_seed(0)

    class _DeadBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 32)
            self.live = nn.Linear(32, 32, bias=False)
            self.dead = nn.Linear(32, 32, bias=False)
            self.head = nn.Linear(32, 64, bias=False)

        def forward(self, ids):
            h = self.embed(ids)
            self.dead(h)  # computed and thrown away
            return self.head(torch.relu(self.live(h)))

    reference = _DeadBranch().eval()
    model = _DeadBranch().eval()
    model.load_state_dict(reference.state_dict())

    alphas = linear_error_coefficients(
        model, lambda m: random_token_divergence(reference, m, 64, seed=5),
        levels=(0.05, 0.10, 0.15),
    )
    assert alphas["dead"] == 0.0
    assert alphas["live"] > 0.0


# --------------------------------------------------------------------------
# The allocator.
# --------------------------------------------------------------------------

def _predicted(alphas, sizes, rho, passes):
    def r(n):
        return rho[n] if isinstance(rho, dict) else rho
    return sum(alphas[n] * r(n) ** (2 * passes[n]) for n in passes)


def _bits(sizes, passes, bits_per_pass):
    total = sum(sizes[n] for n in passes)
    return sum(sizes[n] * passes[n] * bits_per_pass for n in passes) / total


def test_allocation_respects_the_budget_and_the_floor():
    alphas = {"x": 10.0, "y": 1.0, "z": 0.1}
    sizes = {"x": 1000, "y": 1000, "z": 1000}
    passes = knapsack_pass_allocation(alphas, sizes, 0.34, bit_budget=4.2,
                                      bits_per_pass=2.03, max_passes=4)
    assert all(v >= 1 for v in passes.values())
    assert _bits(sizes, passes, 2.03) <= 4.2 + 1e-9


def test_allocation_prefers_the_sensitive_layer():
    alphas = {"sensitive": 100.0, "inert": 0.01}
    sizes = {"sensitive": 1000, "inert": 1000}
    passes = knapsack_pass_allocation(alphas, sizes, 0.34, bit_budget=3.1,
                                      bits_per_pass=2.03, max_passes=3)
    assert passes["sensitive"] > passes["inert"]


def test_allocation_is_optimal_by_exhaustive_search():
    # The greedy is exact because the objective is separable and convex in the
    # pass count. Small enough instances can be checked against brute force, and
    # this is the property the whole design rests on.
    alphas = {"a": 7.3, "b": 2.1, "c": 0.4, "d": 11.9}
    sizes = {"a": 900, "b": 1500, "c": 700, "d": 400}
    bpp, max_passes = 2.03, 4
    per_layer = {"a": 0.3422, "b": 0.2650, "c": 0.3019, "d": 0.3422}

    for rho, budget in itertools.product((0.3422, per_layer),
                                         (2.5, 3.0, 4.0, 5.0, 6.5)):
        got = knapsack_pass_allocation(alphas, sizes, rho, bit_budget=budget,
                                       bits_per_pass=bpp, max_passes=max_passes)
        best, best_val = None, float("inf")
        for combo in itertools.product(range(1, max_passes + 1), repeat=len(sizes)):
            cand = dict(zip(sizes, combo))
            if _bits(sizes, cand, bpp) > budget + 1e-9:
                continue
            val = _predicted(alphas, sizes, rho, cand)
            if val < best_val:
                best, best_val = cand, val
        assert abs(_predicted(alphas, sizes, rho, got) - best_val) < 1e-12, (
            f"rho={rho} budget={budget}: greedy {got} != optimal {best}")


def test_allocation_accepts_per_layer_rho():
    alphas = {"fast": 1.0, "slow": 1.0}
    sizes = {"fast": 1000, "slow": 1000}
    # At equal alpha the extra pass is worth more to the layer with the *larger*
    # residual, i.e. the slower decay: a pass removes a fixed fraction of what is
    # left, so the layer that has more left saves more. The faster-decaying layer
    # is already closer to lossless and buys less by going deeper.
    passes = knapsack_pass_allocation(alphas, sizes, {"fast": 0.26, "slow": 0.34},
                                      bit_budget=3.1, bits_per_pass=2.03, max_passes=3)
    assert passes["slow"] >= passes["fast"]


def test_allocation_refuses_an_impossible_floor():
    try:
        knapsack_pass_allocation({"a": 1.0}, {"a": 100}, 0.34, bit_budget=2.0,
                                 bits_per_pass=2.03, max_passes=3, min_passes=2)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError when min_passes busts the budget")


def test_allocation_degenerates_to_uniform_when_layers_are_identical():
    alphas = {n: 1.0 for n in "abcd"}
    sizes = {n: 1000 for n in "abcd"}
    passes = knapsack_pass_allocation(alphas, sizes, 0.34, bit_budget=4.06,
                                      bits_per_pass=2.03, max_passes=3)
    assert set(passes.values()) == {2}


# --------------------------------------------------------------------------
# The validation gate.
# --------------------------------------------------------------------------

def test_validation_keeps_a_candidate_that_wins():
    from apexquant import validated_pass_allocation
    cand, unif = {"a": 3, "b": 1}, {"a": 2, "b": 2}
    got = validated_pass_allocation(cand, unif, lambda p: 1.0 if p == cand else 2.0)
    assert got == cand


def test_validation_falls_back_when_the_candidate_loses():
    from apexquant import validated_pass_allocation
    cand, unif = {"a": 3, "b": 1}, {"a": 2, "b": 2}
    got = validated_pass_allocation(cand, unif, lambda p: 2.0 if p == cand else 1.0)
    assert got == unif


def test_validation_margin_demands_a_real_improvement():
    from apexquant import validated_pass_allocation
    cand, unif = {"a": 3, "b": 1}, {"a": 2, "b": 2}
    score = lambda p: 0.99 if p == cand else 1.0  # noqa: E731
    assert validated_pass_allocation(cand, unif, score) == cand
    assert validated_pass_allocation(cand, unif, score, margin=1.05) == unif


def test_validation_requires_matching_layer_sets():
    from apexquant import validated_pass_allocation
    try:
        validated_pass_allocation({"a": 1}, {"b": 1}, lambda p: 0.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on mismatched layer sets")


def test_validation_calls_score_exactly_twice():
    from apexquant import validated_pass_allocation
    calls = []
    validated_pass_allocation({"a": 3}, {"a": 2},
                              lambda p: calls.append(dict(p)) or 1.0)
    assert len(calls) == 2
