"""Fused GEMV against a QTIP-style *computed* trellis code: no table at all.

The other three arms bracket a table from 1 MiB down to 4 KiB. This one removes
it. QTIP's bitshift trellis makes the trellis state a pure function of bit
position -- with L=16, K=2, V=1 the t-th weight is decoded from the 16-bit window
of the packed bitstream starting at bit 2t -- so there is no state recurrence to
walk, and the codeword for that window is *computed* by integer arithmetic rather
than looked up. This kernel implements QTIP's 1MAD code function, verbatim from
the reference implementation:

    x <- (34038481 * x + 76625530) mod 2^32            one 32-bit multiply-add
    y <- (x&255) + ((x>>8)&255) + ((x>>16)&255) + ((x>>24)&255)
    w <- (y - 510) / 147.800537109375

The byte-sum of a uniform 32-bit word is a sum of four iid uniforms, hence
approximately Gaussian; the affine tail rescales it to unit variance. Nothing is
loaded. That is the property being measured, and it is why this arm is the only
one whose decode cost is independent of cache state.

Bandwidth is matched to the other three arms exactly. Weights are the same
``(N, K/16)`` int32 words, the same 16 coordinates per word, the same one scale
read per word, the same 2 bits per coordinate. The inner loop, blocking
(BLOCK_N=64, BLOCK_W=32) and split-K policy are copied from triton_gemv_mp, so
against the scalar arm the *only* difference is the arithmetic between loading a
word and multiplying by an activation.

What is faithful and what is not
--------------------------------
Faithful: the 1MAD arithmetic (constants and divisor are the repo's, not the
paper's rounded 147.8); the bitshift-trellis property that weight t reads the
L-bit window at bit offset K*V*t; the tail-biting wrap, which is what QTIP's own
``unpack_trellis`` does when it re-appends the leading L-K*V bits.

Not faithful, and deliberately so: our tail-biting block is one 32-bit word, i.e.
T=16 weights, because that is what pins bandwidth to the other arms. QTIP uses
much longer blocks. Block length changes the *quantiser* -- a short tail-biting
block is a slightly worse code -- but not the decode: every weight is still one
16-bit window plus one 1MAD evaluation, which is the thing being timed. We also
index bit windows LSB-first where QTIP packs MSB-first; that is a relabelling of
the codebook, not a different code. The final rescale is a multiply by the fp32
reciprocal rather than an fp32 divide, which is the form that makes 1MAD four
instructions rather than a MUFU refinement sequence; over the code function's
entire domain, all 2^16 states, that multiply is *bit-identical* to the
reference division, so this costs nothing in fidelity.

Scope. This measures DECODE COST at a fixed rate and matched weight bandwidth.
It is not a reimplementation of QTIP's quantiser: there is no Viterbi search, no
incoherence processing, no fine-tuning, and consequently no claim about
quantisation quality -- the codebook here is whatever 1MAD emits on random words.
Nor is it a reproduction of any published QTIP throughput number. The often-cited
"3.4x FP16" is a derived ratio (188/55.9) from one line of the camera-ready's
Table 6, specific to Llama-2-7B at 2 bits, batch size 1, on an RTX 6000 Ada with
q/k/v and up/gate fusion, and it was achieved by the HYB code, which *does* carry
a 2 KiB lookup table. No throughput number for the lookup-free 1MAD or 3INST
codes is published anywhere, which is precisely why we measure one here rather
than cite one.
"""

from __future__ import annotations

import functools

import torch

# QTIP's 1MAD constants, from lib/codebook/bitshift.py::decode_1mad. The paper
# rounds the divisor to 147.8; the reference implementation does not, and we
# follow the reference so our decode is bit-comparable to theirs.
TCQ_MAD_A = 34038481
TCQ_MAD_B = 76625530
TCQ_MAD_DIV = 147.800537109375
TCQ_MAD_RECIP = 1.0 / TCQ_MAD_DIV

# L=16 bit trellis state, K=2 bits per weight, V=1 weight per group: one int32
# word is a tail-biting block of exactly 16 weights.
TCQ_L = 16
TCQ_KV = 2
TCQ_VALS_PER_WORD = 32 // TCQ_KV

try:
    import triton
    import triton.language as tl

    GEMV_TCQ_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEMV_TCQ_AVAILABLE = False


if GEMV_TCQ_AVAILABLE:

    @triton.jit
    def _decode_1mad(state, A: tl.constexpr, B: tl.constexpr,
                     RECIP: tl.constexpr):
        """QTIP 1MAD: 16-bit trellis state -> pseudo-Gaussian fp32 weight.

        Pure arithmetic, no memory operand. int32 multiply-add wraps mod 2^32,
        which is the LCG step; the four byte extractions are shifts and masks
        (a real kernel folds them into one ``dp4a`` against 0x01010101).
        """
        x = state * A + B                       # LCG, low 32 bits
        y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
        return (y - 510).to(tl.float32) * RECIP

    @triton.jit
    def _gemv_tcq_kernel(
        rx_ptr, words_ptr, scale_ptr, out_ptr,
        N, K, words_per_row, words_per_chunk,
        stride_wn, stride_sn, stride_sg,
        GROUP_SIZE: tl.constexpr, VPW: tl.constexpr,
        A: tl.constexpr, B: tl.constexpr, RECIP: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        chunk = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        w_lo = chunk * words_per_chunk
        w_hi = tl.minimum(w_lo + words_per_chunk, words_per_row)

        wrd_base = words_ptr + offs_n[:, None] * stride_wn
        sc_base = scale_ptr + offs_n[:, None] * stride_sn

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for w0 in range(w_lo, w_hi, BLOCK_W):
            offs_w = w0 + tl.arange(0, BLOCK_W)
            w_mask = offs_w < w_hi
            word = tl.load(wrd_base + offs_w[None, :],
                           mask=n_mask[:, None] & w_mask[None, :], other=0)
            wsum = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.float32)
            # The word is a tail-biting block of VPW weights: weight v reads the
            # 16-bit window at bit offset 2v of the 32-bit word, wrapping round.
            for v in tl.static_range(VPW):
                s = 2 * v
                if s + 16 <= 32:
                    state = (word >> s) & 0xFFFF
                else:
                    top = 32 - s                       # bits taken from the top
                    state = (((word >> s) & ((1 << top) - 1))
                             | ((word & ((1 << (s - 16)) - 1)) << top))
                code = _decode_1mad(state, A, B, RECIP)
                k = offs_w * VPW + v
                rxv = tl.load(rx_ptr + k, mask=k < K, other=0.0)
                wsum += code * rxv[None, :]
            g = (offs_w * VPW) // GROUP_SIZE  # a word lies in one scale group
            sc = tl.load(sc_base + g[None, :] * stride_sg,
                         mask=n_mask[:, None] & w_mask[None, :], other=0.0)
            acc += tl.sum(wsum * sc, axis=1)
        tl.atomic_add(out_ptr + offs_n, acc, mask=n_mask)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def tcq_decode_reference(words: torch.Tensor) -> torch.Tensor:
    """Torch definition of the code the kernel decodes: ``(N, W)`` -> ``(N, 16W)``.

    This is the specification, not a fast path -- it materialises every weight.
    It exists so the fused kernel can be checked against something that computes
    the trellis window and the 1MAD codeword independently, in int64 where no
    wraparound question arises.
    """
    n = words.shape[0]
    w64 = words.to(torch.int64) & 0xFFFFFFFF
    # Tail-biting: the 32-bit block, repeated, so a window may wrap the end.
    doubled = w64 | (w64 << 32)
    off = TCQ_KV * torch.arange(TCQ_VALS_PER_WORD, device=words.device,
                                dtype=torch.int64)
    state = (doubled.unsqueeze(-1) >> off) & ((1 << TCQ_L) - 1)
    x = (state * TCQ_MAD_A + TCQ_MAD_B) & 0xFFFFFFFF
    y = ((x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255))
    return ((y - 510).to(torch.float32) * TCQ_MAD_RECIP).reshape(n, -1)


def triton_gemv_tcq(rx: torch.Tensor, words: torch.Tensor, scale: torch.Tensor,
                    group_size: int) -> torch.Tensor:
    """Fused mat-vec against a computed (table-free) trellis code.

    Note the missing argument: unlike the stored-table and generated-table arms
    there is no codebook tensor, because a computed code has nothing to store.

    Args:
        rx: Rotated activations ``(K,)`` float32.
        words: Packed trellis bits as int32 words ``(N, K/16)``; each word is a
            tail-biting block covering 16 input coordinates, identical bandwidth
            to the scalar, stored-table and generated-table paths.
        scale: Per-group scales ``(N, K/group_size)`` float32.
        group_size: Scale group size (multiple of 16).

    Returns:
        Output ``(1, N)`` float32.
    """
    if not GEMV_TCQ_AVAILABLE:
        raise RuntimeError("triton is not available")
    K = rx.shape[-1]
    N, words_per_row = words.shape
    vpw = TCQ_VALS_PER_WORD
    wpg = group_size // vpw  # words per scale group

    block_n, block_w = 64, 32
    n_blocks = triton.cdiv(N, block_n)
    sms = _sm_count(rx.device.index or 0)
    max_chunks = max(1, words_per_row // max(1, wpg))
    split_k = min(max(1, (2 * sms) // max(1, n_blocks)), max_chunks)
    words_per_chunk = triton.cdiv(triton.cdiv(words_per_row, split_k),
                                  max(1, wpg)) * max(1, wpg)

    out = torch.zeros((1, N), dtype=torch.float32, device=rx.device)
    _gemv_tcq_kernel[(n_blocks, split_k)](
        rx, words, scale, out,
        N, K, words_per_row, words_per_chunk,
        words.stride(0), scale.stride(0), scale.stride(1),
        GROUP_SIZE=group_size, VPW=vpw,
        A=TCQ_MAD_A, B=TCQ_MAD_B, RECIP=TCQ_MAD_RECIP,
        BLOCK_N=block_n, BLOCK_W=block_w,
        num_warps=4, num_stages=3,
    )
    return out
