"""Linear layer that runs a progressive trellis checkpoint through the kernel.

`load_progressive` reconstructs dense weights and hands them to cuBLAS, which is
how every perplexity in the paper was measured and is why a trellis checkpoint
costs as much memory to serve as the fp16 model it replaces. This module is the
packed-form path: the codes stay packed, :func:`triton_gemv_trellis` decodes them
inside the mat-vec, and a dense weight is never formed at batch one.

Three things are worth stating plainly about what this does and does not cover.

**Three paths, by row count.** One row goes to the mat-vec and a few hundred to
the tile GEMM, both keeping the weight packed. Past
``TRELLIS_GEMM_MAX_ROWS`` the layer is reconstructed and handed to cuBLAS, and
that is not a gap waiting to be closed: the fused form is ``P`` dots of ``2MNK``
where reconstruction is a fixed cost plus one, so beyond some M the arithmetic
the fusion adds exceeds the reconstruction it avoids. See the GEMM's module
docstring. Reconstruction is also the fallback for a group too narrow to tile,
which no real layer is.

What survives at every M is the thing that matters most here: resident memory
stays at the size of the codes. A reconstruction is one layer's worth and
transient, so a 12B checkpoint still fits on a card that could not hold it dense.

**One table, shared.** The permuted codeword table depends only on the codec, and
Proposition 2 is what says the codec does not change across passes, so every
layer of a model can hold the same tensor. Pass the same one to every layer and
they share storage; it is registered non-persistently because it is derived from
the artifact's config rather than stored in it.

**Float32 inside.** The kernel accumulates in fp32 and the table is fp32, so an
fp16 activation is cast on the way in and the result is cast back. That makes
this path slightly *more* accurate than the dense fp16 one it replaces, not less,
which is worth remembering when comparing logits: they will not be bit-identical
to the reconstruct-and-cuBLAS path, and the difference runs in this direction.
"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn.functional as F
from torch import nn

from ..kernels.triton_gemm_trellis import (
    TRELLIS_GEMM_MAX_ROWS,
    trellis_gemm_block_k,
    triton_gemm_trellis,
)
from ..kernels.triton_gemv_trellis import triton_gemv_trellis
from ..rotation.random_rotation import random_rotation
from ..rotation.rotation_matrix import MATMUL_ROTATION_MAX_DIM, rotation_matrix
from ..rotation.triton_wht_rotate import MAX_FUSED_DIM, triton_wht_rotate
from .progressive_artifact import dequantize


@functools.lru_cache(maxsize=None)
def _stacked_rotations(gs: int, seed: int, passes: int,
                      device: str) -> torch.Tensor:
    """``(P, gs, gs)`` rotations for passes ``seed..seed+P-1``, built once.

    The same four lines as ``kernels.fused_dequant_matmul._stacked_rotations``,
    for the same reason -- re-stacking per call is an alloc and a copy that shows
    up at decode batch one. Left duplicated rather than lifted to a shared module
    so that this path adds no edit to the scalar one; collapsing the two is a
    tidy-up, not a fix.
    """
    return torch.stack([rotation_matrix(gs, seed + p, device)
                        for p in range(passes)]).contiguous()


class ProgressiveTrellisLinear(nn.Module):
    """``nn.Linear`` whose weight stays a progressive trellis code.

    Attributes are buffers, so ``.to(device)`` and ``state_dict`` behave. The
    codec is held as a plain attribute rather than a buffer: it is a rebuildable
    description of the format, and every layer shares one.
    """

    #: The GEMM can add the bias and apply GELU in its epilogue, so a caller
    #: that wants ``gelu(layer(x))`` should ask for it rather than run a second
    #: pass over the output. A dense ``nn.Linear`` has no such option, which is
    #: why this is advertised rather than assumed.
    fuses_gelu = True

    def __init__(self, words: torch.Tensor, scale: torch.Tensor,
                 table: torch.Tensor, codec, in_features: int,
                 group_size: int, seed: int,
                 bias: torch.Tensor | None) -> None:
        """
        Args:
            words: ``(P, out, K*bits//32)`` int32, the artifact's packed codes
                viewed as int32 words.
            scale: ``(P, out, groups)`` float32 per-group scales.
            table: ``(2**L,)`` from ``trellis_window_table``, shared by layers.
            codec: The ``TrellisCodec`` the artifact was written with; used only
                by the multi-row fallback.
            in_features: True input width, before the group padding.
            group_size: Scale group size, and the length of one trellis run.
            seed: Base rotation seed; pass ``p`` used ``seed + p``.
            bias: Full-precision bias, or None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = words.shape[1]
        self.passes = words.shape[0]
        self.group_size = group_size
        self.num_groups = scale.shape[-1]
        self.seed = seed
        self.codec = codec

        self.register_buffer("words", words)
        self.register_buffer("scale", scale)
        self.register_buffer("table", table, persistent=False)
        self.register_buffer("bias", bias if bias is not None else None)
        self._dense_cache: torch.Tensor | None = None
        self._resident_dense = os.environ.get("APEXQUANT_TRELLIS_DENSE_CACHE", "0") == "1"

    def _rotate(self, xg: torch.Tensor) -> torch.Tensor:
        """Grouped activations ``(M, groups, gs)`` to ``(P, M, K)``.

        The butterfly arm computes the same rotation as the matmul at
        ``g log g`` work instead of ``g**2``, in one launch, and is the default on
        CUDA. It agrees with the matmul to about 5e-07 relative, which is
        rounding: the two are the same orthogonal transform, and unlike the
        encode path (see ``rotation.hadamard_backend``) nothing here re-decides a
        codeword, so the difference cannot compound.

        The matmul arm, kept for the non-CUDA and oversized-group cases,
        broadcasts one batched gemm rather than contracting with
        ``einsum``. Both compute the same thing, but ``"mng,pgh->pmnh"`` lands
        the pass axis outermost only after a permute, so the ``reshape`` that
        follows cannot be a view and the ``contiguous`` is a full copy of the
        rotated activations. Batching over the pass axis instead writes them in
        that order to begin with. Worth the rewrite because it runs per layer per
        forward: on Clay at P=1 the rotation and its copy were 1.4 ms of a 17 ms
        forward.
        """
        gs, p, m = self.group_size, self.passes, xg.shape[0]
        if xg.is_cuda and gs <= MAX_FUSED_DIM and not gs & (gs - 1):
            return triton_wht_rotate(xg.reshape(-1, gs), gs, self.seed,
                                     p).view(p, m, -1)
        if gs <= MATMUL_ROTATION_MAX_DIM:
            r = _stacked_rotations(gs, self.seed, p, str(xg.device))
            return torch.matmul(xg.reshape(1, -1, gs), r).view(p, m, -1)
        return torch.stack([
            random_rotation(xg, self.seed + i).reshape(m, -1) for i in range(p)
        ]).contiguous()

    def _dense(self) -> torch.Tensor:
        """Reconstruct the layer. Only the multi-row fallback calls this."""
        codes = self.words.view(torch.uint8).reshape(
            self.passes, self.out_features, self.num_groups, -1)
        return dequantize(codes, self.scale, self.codec, self.in_features,
                          self.group_size, self.seed, passes=self.passes)

    def _cached_dense(self, dtype: torch.dtype) -> torch.Tensor:
        if self._dense_cache is None:
            self._dense_cache = self._dense().to(dtype)
            self.words = None
            self.scale = None
            self.table = None
        return self._dense_cache

    def forward(self, x: torch.Tensor, gelu: bool = False) -> torch.Tensor:
        """Apply the layer, optionally with GELU fused into the matmul.

        Args:
            x: Input ``(..., in_features)``.
            gelu: Apply the exact (erf) GELU to the result. Free on the packed
                GEMM path, where it costs one instruction on a tile already in
                registers; applied separately otherwise.
        """
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        m, gs = x2.shape[0], self.group_size
        packed_ok = m == 1 or (m <= TRELLIS_GEMM_MAX_ROWS
                               and trellis_gemm_block_k(gs, self.codec.bits))
        epilogue_done = False
        if self._resident_dense:
            dense = self._cached_dense(x2.dtype)
            out = F.linear(x2.to(dense.dtype), dense)
        elif not packed_ok:
            out = F.linear(x2.float(), self._dense())
        else:
            pad = self.num_groups * gs - x2.shape[-1]
            xg = x2.float()
            if pad:
                xg = F.pad(xg, (0, pad))
            rx = self._rotate(xg.view(m, self.num_groups, gs))
            args = (self.words, self.scale, self.table, gs, self.codec.bits,
                    self.codec.trellis_bits)
            if m == 1:
                out = triton_gemv_trellis(rx[:, 0], *args)
            else:
                out = triton_gemm_trellis(rx, *args, bias=self.bias, gelu=gelu)
                epilogue_done = True
        out = out.to(x.dtype).view(*lead, self.out_features)
        if epilogue_done:
            return out
        if self.bias is not None:
            out = out + self.bias
        return F.gelu(out) if gelu else out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, passes={self.passes}, "
                f"group_size={self.group_size}, bits={self.codec.bits}, "
                f"trellis_bits={self.codec.trellis_bits}")
