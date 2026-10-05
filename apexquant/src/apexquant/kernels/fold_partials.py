"""Launch the band-folding kernel."""

from __future__ import annotations

import torch
import triton

from .fold_partials_kernel import fold_partials_kernel

FOLD_BLOCK_M = 32
FOLD_BLOCK_N = 128


def fold_partials(
    parts: torch.Tensor,
    bands: int,
    rows: int,
    cols: int,
    bias: torch.Tensor | None = None,
    gelu: bool = False,
) -> torch.Tensor:
    """Sum ``bands`` bands of ``parts`` into one ``(rows, cols)`` result.

    Args:
        parts: Partial results, ``(bands * rows, cols)``.
        bands: Number of bands to reduce, ``SPLIT_K * P``.
        rows: Output rows.
        cols: Output columns.
        bias: Optional bias of shape ``(cols,)``, added after the reduction.
        gelu: Apply the exact (erf) GELU after the bias.

    Returns:
        Tensor of shape ``(rows, cols)``.
    """
    out = torch.empty(rows, cols, device=parts.device, dtype=parts.dtype)
    grid = (triton.cdiv(rows, FOLD_BLOCK_M), triton.cdiv(cols, FOLD_BLOCK_N))
    fold_partials_kernel[grid](
        parts,
        bias if bias is not None else parts,
        out,
        rows,
        cols,
        bands,
        rows * cols,
        HAS_BIAS=bias is not None,
        ACT_GELU=gelu,
        BLOCK_M=FOLD_BLOCK_M,
        BLOCK_N=FOLD_BLOCK_N,
    )
    return out
