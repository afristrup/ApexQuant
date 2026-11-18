"""Packed nested trellis decoding for routed mixture-of-experts layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..fuse.knapsack_pass_allocation import knapsack_pass_allocation
from ..quantize.pack_bits import pack_bits
from ..vq.progressive_artifact import dequantize


class ProgressiveMoE(nn.Module):
    """Apply a packed stacked-expert artifact to top-k routed tokens."""

    def __init__(self, codes, scale, table, codec, in_features, group_size, seed, routing_weights=None):
        super().__init__()
        if codes.ndim != 5 or scale.ndim != 4 or codes.shape[:4] != scale.shape:
            raise ValueError("codes must be (passes, experts, out, groups, values)")
        self.passes, self.experts, self.out_features = codes.shape[:3]
        self.in_features, self.group_size, self.seed, self.codec = in_features, group_size, seed, codec
        self.register_buffer("codes", codes)
        self.register_buffer("scale", scale)
        self.register_buffer("table", table, persistent=False)
        self.register_buffer("routing_weights", routing_weights, persistent=False)

    def _dense(self, passes):
        return dequantize(self.codes[:passes].flatten(1, 2), self.scale[:passes].flatten(1, 2), self.codec,
                          self.in_features, self.group_size, self.seed, passes=passes).view(self.experts, self.out_features, -1)

    def router_passes(self, routing_weights, rho, min_passes=1, max_passes=None):
        """Choose a nested depth per active route from its gate weight."""
        if max_passes is None:
            max_passes = self.passes
        if not 0 < rho < 1:
            raise ValueError("rho must be between zero and one")
        if not 1 <= min_passes <= max_passes <= self.passes:
            raise ValueError("pass bounds must lie within the stored artifact")
        if routing_weights.numel() == 0:
            return routing_weights.to(dtype=torch.long)
        if (routing_weights < 0).any() or not torch.isfinite(routing_weights).all():
            raise ValueError("routing_weights must be finite and non-negative")
        tiny = torch.finfo(routing_weights.dtype).tiny
        target = routing_weights.max().clamp_min(tiny) * rho ** max_passes
        depths = torch.ceil(torch.log(target / routing_weights.clamp_min(tiny)) /
                            torch.log(torch.as_tensor(rho, device=routing_weights.device,
                                                       dtype=routing_weights.dtype)))
        return depths.clamp(min=min_passes, max=max_passes).to(torch.long)

    def allocate_expert_passes(self, expert_indices, routing_weights, rho,
                               min_passes=1, max_passes=None, reduction="sum"):
        """Allocate one depth per expert from observed router traffic."""
        importance = self._expert_importance(expert_indices, routing_weights, reduction)
        return self.router_passes(importance, rho, min_passes, max_passes)

    def _expert_importance(self, expert_indices, routing_weights, reduction):
        """Aggregate nonnegative router evidence into one score per expert."""
        if expert_indices.ndim != 2 or routing_weights.shape != expert_indices.shape:
            raise ValueError("expert_indices and routing_weights must be (tokens, top_k)")
        if reduction not in ("sum", "max"):
            raise ValueError("reduction must be 'sum' or 'max'")
        if (expert_indices < 0).any() or (expert_indices >= self.experts).any():
            raise ValueError("expert index is outside the packed expert range")
        if (routing_weights < 0).any() or not torch.isfinite(routing_weights).all():
            raise ValueError("routing_weights must be finite and non-negative")
        importance = torch.zeros(self.experts, device=expert_indices.device,
                                 dtype=routing_weights.dtype)
        flat_ids = expert_indices.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        if reduction == "sum":
            importance.scatter_add_(0, flat_ids, flat_weights)
        else:
            importance.scatter_reduce_(0, flat_ids, flat_weights, reduce="amax")
        return importance

    def threshold_passes(self, gate, rho, tolerance, initial=None,
                         min_passes=1, max_passes=None):
        """Choose the least prefix satisfying ``gate * initial * rho**P <= tolerance``."""
        if max_passes is None:
            max_passes = self.passes
        if not 0 < rho < 1 or not 1 <= min_passes <= max_passes <= self.passes:
            raise ValueError("invalid rho or pass bounds")
        if initial is None:
            initial = torch.ones_like(gate)
        tolerance = torch.as_tensor(tolerance, device=gate.device, dtype=gate.dtype)
        if initial.shape != gate.shape or (gate < 0).any() or (initial < 0).any():
            raise ValueError("gate and initial must be nonnegative tensors with matching shapes")
        if (tolerance <= 0).any() or not torch.isfinite(tolerance).all():
            raise ValueError("tolerance must be finite and positive")
        denominator = gate * initial
        tiny = torch.finfo(gate.dtype).tiny
        ratio = tolerance / denominator.clamp_min(tiny)
        raw = torch.ceil(torch.log(ratio.clamp_min(tiny)) /
                         torch.log(torch.as_tensor(rho, device=gate.device, dtype=gate.dtype)))
        raw = torch.where(denominator > 0, raw, torch.full_like(raw, min_passes))
        return raw.clamp(min=min_passes, max=max_passes).to(torch.long)

    def allocate_expert_passes_budget(self, expert_indices, routing_weights, rho,
                                      bit_budget, bits_per_pass, initial=None,
                                      min_passes=1, max_passes=None,
                                      reduction="sum", resolution=200_000):
        """Minimize predicted squared router error under an average bit budget."""
        if max_passes is None:
            max_passes = self.passes
        importance = self._expert_importance(expert_indices, routing_weights.square(), reduction)
        if initial is None:
            initial = torch.ones(self.experts, device=importance.device, dtype=importance.dtype)
        if initial.shape != (self.experts,) or (initial < 0).any():
            raise ValueError("initial must be a nonnegative vector with one value per expert")
        coefficients = {
            expert: float(importance[expert] * initial[expert].square())
            for expert in range(self.experts)
        }
        size = self.out_features * self.in_features
        allocation = knapsack_pass_allocation(
            coefficients, {expert: size for expert in range(self.experts)}, rho,
            bit_budget, bits_per_pass=bits_per_pass, min_passes=min_passes,
            max_passes=max_passes, resolution=resolution)
        return torch.tensor([allocation[expert] for expert in range(self.experts)],
                            device=importance.device, dtype=torch.long)

    def _forward_fixed(self, x, expert_indices, weights, passes, use_triton):
        """Decode one nested prefix for a dense route batch."""
        if use_triton and x.is_cuda and hasattr(self.codec, "trellis_bits"):
            from ..rotation.triton_wht_rotate import triton_wht_rotate
            from .triton_moe_trellis import fused_moe_trellis
            routes = expert_indices.shape[1]
            grouped = x.float()
            pad = (-grouped.shape[-1]) % self.group_size
            if pad:
                grouped = F.pad(grouped, (0, pad))
            groups = grouped.shape[-1] // self.group_size
            rotated = triton_wht_rotate(
                grouped[:, None, :].expand(-1, routes, -1).reshape(-1, self.group_size),
                self.group_size, self.seed, passes)
            rx = rotated.view(passes, x.shape[0] * routes, groups * self.group_size)
            packed = pack_bits(self.codes[:passes], self.codec.bits)
            codes = packed.reshape(passes, self.experts, self.out_features, -1).contiguous().view(torch.int32)
            parts = fused_moe_trellis(
                rx, expert_indices.reshape(-1), codes, self.scale[:passes], self.table,
                self.group_size, self.codec.bits, self.codec.trellis_bits)
            return (parts.sum(0).view(x.shape[0], routes, -1) *
                    weights.float().unsqueeze(-1)).sum(1)

        if use_triton and x.is_cuda and hasattr(self.codec, "levels"):
            from ..kernels.fused_dequant_matmul import fused_dequant_matmul
            from ..quantize.quantized_weight import QuantizedWeight
            out = torch.zeros(x.shape[0], self.out_features, device=x.device, dtype=torch.float32)
            for expert in expert_indices.unique().tolist():
                qw = QuantizedWeight(
                    packed=self.codes[:passes, expert].contiguous(),
                    scale=self.scale[:passes, expert].float().contiguous(),
                    levels=self.codec.levels, in_features=self.in_features,
                    out_features=self.out_features, group_size=self.group_size,
                    seed=self.seed, bits=self.codec.bits)
                for route in range(expert_indices.shape[1]):
                    selected = expert_indices[:, route] == expert
                    if selected.any():
                        out[selected] += fused_dequant_matmul(
                            x[selected].float(), qw) * weights[selected, route].float().unsqueeze(-1)
            return out

        dense = self._dense(passes)
        out = torch.zeros(x.shape[0], self.out_features, device=x.device, dtype=torch.float32)
        for expert in expert_indices.unique().tolist():
            for route in range(expert_indices.shape[1]):
                selected = expert_indices[:, route] == expert
                if selected.any():
                    out[selected] += F.linear(x[selected].float(), dense[expert]) * weights[selected, route].float().unsqueeze(-1)
        return out

    def forward(self, x, expert_indices, routing_weights=None, passes=None, use_triton=True):
        """Return routed outputs, optionally with one nested depth per route."""
        if x.ndim != 2 or expert_indices.ndim != 2 or x.shape[0] != expert_indices.shape[0]:
            raise ValueError("x must be (tokens, in_features), indices (tokens, top_k)")
        weights = routing_weights if routing_weights is not None else self.routing_weights
        if weights is None:
            weights = torch.ones_like(expert_indices, dtype=x.dtype)
        if weights.shape != expert_indices.shape:
            raise ValueError("routing_weights and expert_indices must have the same shape")
        if expert_indices.numel() == 0:
            raise ValueError("at least one route is required")
        expert_indices = expert_indices.to(x.device)
        weights = weights.to(device=x.device)
        if expert_indices.min() < 0 or expert_indices.max() >= self.experts:
            raise ValueError("expert index is outside the packed expert range")

        if passes is None:
            passes = self.passes
        if isinstance(passes, int):
            if not 0 < passes <= self.passes:
                raise ValueError(f"passes must be in 1..{self.passes}")
            return self._forward_fixed(x, expert_indices, weights, passes, use_triton).to(x.dtype)
        if not torch.is_tensor(passes):
            raise ValueError("passes must be an integer or tensor")
        if passes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("passes must be an integer tensor")
        passes = passes.to(x.device)
        if passes.ndim == 1 and passes.shape == (self.experts,):
            passes = passes[expert_indices]
        elif passes.shape != expert_indices.shape:
            raise ValueError("passes must be per-expert or per-route")
        if passes.numel() and (passes.min() < 1 or passes.max() > self.passes):
            raise ValueError(f"passes must be in 1..{self.passes}")
        output = torch.zeros(x.shape[0], self.out_features, device=x.device, dtype=torch.float32)
        for depth in passes.unique(sorted=True).tolist():
            tokens, routes = torch.nonzero(passes == depth, as_tuple=True)
            parts = self._forward_fixed(
                x[tokens], expert_indices[tokens, routes, None],
                weights[tokens, routes, None], int(depth), use_triton)
            output.index_add_(0, tokens, parts)
        return output.to(x.dtype)


class ProgressiveTrellisMoE(ProgressiveMoE):
    """Backward-compatible name for a trellis-backed progressive MoE."""


class ProgressiveScalarMoE(ProgressiveMoE):
    """Progressive MoE backed by packed scalar codes."""


class ProgressiveE8MoE(ProgressiveMoE):
    """Progressive MoE backed by E8P codes."""
