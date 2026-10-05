"""Scalar (apex Lloyd-Max) codec with dense bit packing.

This is the published stage quantiser, wrapped in the same encode/decode
interface as :class:`~apexquant.vq.e8p_codec.E8PCodec` so the progressive
recursion and the checkpoint writer are stage-agnostic.

Packing matters for the bit accounting. ``QuantizedWeight`` stores indices
nibble-packed at 4 bits each, which is right for ``b=4`` and wastes half the
space at ``b=2`` -- the headline setting. An artifact that claims 2.03 bits per
weight has to actually store 2, so this module packs densely at the true width:
``8 // b`` indices per byte.
"""

from __future__ import annotations

import os

import torch

from ..codebook.apex_beta_levels import apex_beta_levels
from ..quantize.quantize_to_indices import quantize_to_indices

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                  # CPU-only or no triton
    HAVE_TRITON = False

# Escape hatch, and the only supported way to compare the two paths on one
# machine. The fallback is the reference implementation the kernel is checked
# against, so this also answers "is the kernel the thing that changed this
# number" without editing code.
if os.environ.get("APEXQUANT_NO_TRITON") == "1":
    HAVE_TRITON = False

# Output values per program, matching the other two fused decoders.
_DECODE_BLOCK = 1024


if HAVE_TRITON:

    @triton.jit
    def _decode_kernel(codes_ptr, lvl_ptr, out_ptr, total, dd, code_stride,
                       BITS: tl.constexpr, PER: tl.constexpr,
                       BLOCK: tl.constexpr):
        """Packed indices straight to levels, unpack fused into the gather.

        Flat over output elements for the same reason the trellis kernel is: a
        grid shaped per row makes program size a function of the group size,
        which is not what should decide occupancy.
        """
        e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = e < total
        row, pos = e // dd, e % dd
        byte = tl.load(codes_ptr + row.to(tl.int64) * code_stride + pos // PER,
                       mask=live, other=0)
        idx = (byte.to(tl.int32) >> ((pos % PER) * BITS)) & ((1 << BITS) - 1)
        tl.store(out_ptr + e, tl.load(lvl_ptr + idx, mask=live, other=0.0),
                 mask=live)


def pack_indices(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Densely pack ``bits``-wide indices along the last axis into uint8.

    Args:
        idx: Uint8 indices in ``[0, 2**bits)``, last axis a multiple of ``8 //
            bits``.
        bits: Index width; must divide 8.

    Returns:
        Uint8 tensor whose last axis is ``idx.shape[-1] * bits // 8``.
    """
    if 8 % bits:
        raise ValueError(f"bits must divide 8, got {bits}")
    per = 8 // bits
    n = idx.shape[-1]
    if n % per:
        raise ValueError(f"last axis {n} not a multiple of {per}")
    v = idx.reshape(*idx.shape[:-1], n // per, per).to(torch.uint8)
    out = torch.zeros(v.shape[:-1], dtype=torch.uint8, device=idx.device)
    for i in range(per):
        out |= v[..., i] << (i * bits)
    return out


def unpack_indices(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of :func:`pack_indices`."""
    per = 8 // bits
    mask = (1 << bits) - 1
    outs = [(packed >> (i * bits)) & mask for i in range(per)]
    return torch.stack(outs, dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * per)


class ScalarCodec:
    """Coordinate-wise nearest-apex-level quantiser, densely packed.

    The codebook is the Lloyd-Max solution for the *known* post-rotation
    marginal ``f_d``, solved on a grid, so it is fitted to a density rather than
    to data: the zero-data constraint holds.
    """

    def __init__(self, d: int, bits: int = 2,
                 device: torch.device | str = "cpu") -> None:
        """
        Args:
            d: Group size (fan-in of the marginal ``f_d``).
            bits: Index width in bits.
            device: Device to build the codebook on.
        """
        self.device = torch.device(device)
        self.bits = bits
        self.d = d
        self.levels = apex_beta_levels(bits, d, device=self.device)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """Encode ``(n, gs)`` values to densely packed uint8 codes."""
        return pack_indices(quantize_to_indices(u, self.levels), self.bits)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode packed uint8 codes back to values.

        The torch path unpacks to an int64 index tensor first, which is eight
        bytes per coordinate written to DRAM and read back purely to be thrown
        away. Fusing the unpack into the gather removes it.
        """
        if HAVE_TRITON and codes.is_cuda:
            flat = codes.reshape(-1, codes.shape[-1]).contiguous()
            dd = flat.shape[-1] * (8 // self.bits)
            out = torch.empty((flat.shape[0], dd), dtype=self.levels.dtype,
                              device=codes.device)
            total = out.numel()
            _decode_kernel[(triton.cdiv(total, _DECODE_BLOCK),)](
                flat, self.levels, out, total, dd, flat.shape[1],
                BITS=self.bits, PER=8 // self.bits, BLOCK=_DECODE_BLOCK,
                num_warps=4)
            return out.reshape(*codes.shape[:-1], dd)
        return self.levels[unpack_indices(codes, self.bits).long()]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(u)).reshape(u.shape)
