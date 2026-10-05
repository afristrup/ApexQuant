"""Multi-pass split-K GEMV with the LOP3 packed-dot decode (2-bit only).

Same launch structure as :mod:`triton_gemv_mp` (pass index and K-split in the
grid, group-aligned chunks, fp32 atomics), but the per-value decode loop --
shift, mask, codebook gather, fp32 fma, the measured 2.49x instruction-issue
tax of lessons N -- is replaced by a PTX block that decodes one int32 word (16
values) and computes its entire inner-product contribution in packed fp16
(lessons R, prototype ``experiments/prototypes/lop3_gemv_proto.py``):

* ``u = q ^ (q >> 1)`` puts the inner/outer magnitude flag at each 2-bit
  field's low bit; eight ``lop3`` splice the flag fields into fp16 lanes
  against the ``0x6400`` magic (fields above mantissa position 8 come from
  ``u >> 8``); an exact ``sub.f16x2`` removes the 1024 bias and one
  ``fma.rn.f16x2`` maps the flag to the magnitude (multipliers are exact
  powers of two, so no rounding beyond fp16 of the levels themselves).
* The sign is spliced from ``NOT(bit1)`` pre-shifted to lane bit 15
  (``lop3`` immLut ``0xd8``: sign-bit mux).
* Eight ``mul/fma.f16x2`` against activation pairs, an ``add.f16x2`` tree and
  ONE fp32 convert close the word. fp16 accumulation spans a single word (16
  values), so the group scale and the fp32 block accumulator see ~1e-3
  relative rounding at most, three orders below the 2-bit quantisation noise.

The activation pairs are the Marlin move: the packed words are re-permuted
ONCE (cached view, like ``decode_views``) so that the two fp16 lanes of asm
pair ``p`` decode the *adjacent* original values ``(2p, 2p+1)`` of the word's
K-range. The kernel then reads each activation pair as one int32 from the
fp16 activations -- no per-value pairing arithmetic in kernel and no host-side
re-layout beyond a single fp32->fp16 cast per forward. (The first attempt
paired in-kernel from fp32 with uint16 bitcast chains; it compiled so poorly
it was 4x slower than the decode it replaced -- re-measured, not assumed.)
"""

from __future__ import annotations

import functools

import torch

try:
    import triton
    import triton.language as tl

    GEMV_MP_LOP3_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEMV_MP_LOP3_AVAILABLE = False


if GEMV_MP_LOP3_AVAILABLE:

    # One int32 word ($1) + 8 activation half2 pairs ($2-$9) + 5 half2 decode
    # constants ($10: (b-a), $11: a, $12-$14: (b-a)/4^p) -> fp32 dot ($0).
    _DOT_ASM = tl.constexpr(r"""
{
.reg .b32 q, u, qh, uh, qn, qnh, t, e, s, bias;
.reg .b32 v0, v1, v2, v3, v4, v5, v6, v7;
.reg .b16 hlo, hhi;
.reg .f32 flo, fhi;
mov.b32 bias, 0x64006400;
mov.b32 q, $1;
shr.b32 t, q, 1;
xor.b32 u, q, t;
shr.b32 qh, q, 8;
shr.b32 uh, u, 8;
not.b32 qn, q;
shr.b32 qnh, qn, 8;
lop3.b32 e, u, 0x00010001, 0x64006400, 0xea; sub.f16x2 e, e, bias; fma.rn.f16x2 e, e, $10, $11; shl.b32 s, qn, 14; lop3.b32 v0, e, s, 0x80008000, 0xd8;
lop3.b32 e, u, 0x00040004, 0x64006400, 0xea; sub.f16x2 e, e, bias; fma.rn.f16x2 e, e, $12, $11; shl.b32 s, qn, 12; lop3.b32 v1, e, s, 0x80008000, 0xd8;
lop3.b32 e, u, 0x00100010, 0x64006400, 0xea; sub.f16x2 e, e, bias; fma.rn.f16x2 e, e, $13, $11; shl.b32 s, qn, 10; lop3.b32 v2, e, s, 0x80008000, 0xd8;
lop3.b32 e, u, 0x00400040, 0x64006400, 0xea; sub.f16x2 e, e, bias; fma.rn.f16x2 e, e, $14, $11; shl.b32 s, qn, 8;  lop3.b32 v3, e, s, 0x80008000, 0xd8;
lop3.b32 e, uh, 0x00010001, 0x64006400, 0xea; sub.f16x2 e, e, bias; fma.rn.f16x2 e, e, $10, $11; shl.b32 s, qnh, 14; lop3.b32 v4, e, s, 0x80008000, 0xd8;
lop3.b32 e, uh, 0x00040004, 0x64006400, 0xea; sub.f16x2 e, e, bias; fma.rn.f16x2 e, e, $12, $11; shl.b32 s, qnh, 12; lop3.b32 v5, e, s, 0x80008000, 0xd8;
lop3.b32 e, uh, 0x00100010, 0x64006400, 0xea; sub.f16x2 e, e, bias; fma.rn.f16x2 e, e, $13, $11; shl.b32 s, qnh, 10; lop3.b32 v6, e, s, 0x80008000, 0xd8;
lop3.b32 e, uh, 0x00400040, 0x64006400, 0xea; sub.f16x2 e, e, bias; fma.rn.f16x2 e, e, $14, $11; shl.b32 s, qnh, 8;  lop3.b32 v7, e, s, 0x80008000, 0xd8;
mul.f16x2 v0, v0, $2;
fma.rn.f16x2 v0, v1, $3, v0;
mul.f16x2 v1, v2, $4;
fma.rn.f16x2 v1, v3, $5, v1;
mul.f16x2 v2, v4, $6;
fma.rn.f16x2 v2, v5, $7, v2;
mul.f16x2 v3, v6, $8;
fma.rn.f16x2 v3, v7, $9, v3;
add.f16x2 v0, v0, v1;
add.f16x2 v2, v2, v3;
add.f16x2 v0, v0, v2;
mov.b32 {hlo, hhi}, v0;
cvt.f32.f16 flo, hlo;
cvt.f32.f16 fhi, hhi;
add.f32 $0, flo, fhi;
}
""")

    @triton.jit
    def _rxi(rxi_base, i_base, p, IW, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
        """int32 activation pair p of each word: fp16 lanes (rx_{2p}, rx_{2p+1})."""
        v = tl.load(rxi_base + i_base + p, mask=i_base + p < IW, other=0)
        return v[None, :] + tl.zeros((BLOCK_N, BLOCK_W), tl.int32)

    @triton.jit
    def _gemv_mp_lop3_kernel(
        rxi_ptr, words_ptr, scale_ptr, out_ptr,
        N, IW, words_per_row, words_per_chunk,  # IW = K // 2 int32 activation pairs per pass
        stride_rp,                        # rxi (P, K//2): pass stride
        stride_wp, stride_wn,             # words (P, N, W), lane-permuted
        stride_sp, stride_sn, stride_sg,  # scale (P, N, G)
        k1_0, k2c, k1_2, k1_4, k1_6,      # half2 decode constants as int32
        GROUP_SIZE: tl.constexpr, VPW: tl.constexpr,
        SPLIT_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_z = tl.program_id(1)
        p = pid_z // SPLIT_K
        chunk = pid_z % SPLIT_K
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        w_lo = chunk * words_per_chunk
        w_hi = tl.minimum(w_lo + words_per_chunk, words_per_row)

        rxi_base = rxi_ptr + p * stride_rp
        wrd_base = words_ptr + p * stride_wp + offs_n[:, None] * stride_wn
        sc_base = scale_ptr + p * stride_sp + offs_n[:, None] * stride_sn

        c1 = tl.full((BLOCK_N, BLOCK_W), k1_0, tl.int32)
        c2 = tl.full((BLOCK_N, BLOCK_W), k2c, tl.int32)
        c3 = tl.full((BLOCK_N, BLOCK_W), k1_2, tl.int32)
        c4 = tl.full((BLOCK_N, BLOCK_W), k1_4, tl.int32)
        c5 = tl.full((BLOCK_N, BLOCK_W), k1_6, tl.int32)

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for w0 in range(w_lo, w_hi, BLOCK_W):
            offs_w = w0 + tl.arange(0, BLOCK_W)
            w_mask = offs_w < w_hi
            word = tl.load(wrd_base + offs_w[None, :],
                           mask=n_mask[:, None] & w_mask[None, :], other=0)
            i_base = offs_w * (VPW // 2)  # 8 int32 activation pairs per word
            r0 = _rxi(rxi_base, i_base, 0, IW, BLOCK_N, BLOCK_W)
            r1 = _rxi(rxi_base, i_base, 1, IW, BLOCK_N, BLOCK_W)
            r2 = _rxi(rxi_base, i_base, 2, IW, BLOCK_N, BLOCK_W)
            r3 = _rxi(rxi_base, i_base, 3, IW, BLOCK_N, BLOCK_W)
            r4 = _rxi(rxi_base, i_base, 4, IW, BLOCK_N, BLOCK_W)
            r5 = _rxi(rxi_base, i_base, 5, IW, BLOCK_N, BLOCK_W)
            r6 = _rxi(rxi_base, i_base, 6, IW, BLOCK_N, BLOCK_W)
            r7 = _rxi(rxi_base, i_base, 7, IW, BLOCK_N, BLOCK_W)
            wsum = tl.inline_asm_elementwise(
                _DOT_ASM, "=f,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
                [word, r0, r1, r2, r3, r4, r5, r6, r7, c1, c2, c3, c4, c5],
                dtype=tl.float32, is_pure=True, pack=1)
            g = (offs_w * VPW) // GROUP_SIZE  # a word lies in one scale group
            sc = tl.load(sc_base + g[None, :] * stride_sg,
                         mask=n_mask[:, None] & w_mask[None, :], other=0.0)
            acc += tl.sum(wsum * sc, axis=1)
        tl.atomic_add(out_ptr + offs_n, acc, mask=n_mask)


# asm pair p decodes word fields (p, p+8); permute so those fields hold the
# original values (2p, 2p+1), making the activation pair one contiguous load.
_FIELD_TO_VALUE = [2 * f if f < 8 else 2 * (f - 8) + 1 for f in range(16)]


def lop3_views(qw) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
    """Cached ``(permuted words, half2 decode constants)`` for the asm decode.

    Field ``f`` of each permuted word holds original value ``2f`` (``f < 8``)
    or ``2(f-8)+1``: the asm's half2 pair ``p`` then decodes original values
    ``(2p, 2p+1)``, adjacent in K. Pure bit shuffling, done once per weight.
    The constants ((b-a), a, (b-a)/4, /16, /64 as half2 int32 bit patterns)
    live here too: reading ``levels`` per call is a GPU->CPU sync that costs
    more than the kernel itself (~15 us each, measured).
    """
    cached = getattr(qw, "_lop3_views", None)
    if cached is not None and cached[0].device == qw.packed.device:
        return cached
    words, _, levels = qw.decode_views()
    perm = torch.zeros_like(words)
    for f, v in enumerate(_FIELD_TO_VALUE):
        perm |= ((words >> (2 * v)) & 3) << (2 * f)
    perm = perm.contiguous()

    def half2(x: float) -> int:
        h = torch.tensor([x], dtype=torch.float16).view(torch.int16)[0].item() & 0xFFFF
        v = (h << 16) | h
        return v - (1 << 32) if v >= (1 << 31) else v

    lvl_a, lvl_b = float(levels[3]), float(levels[2])
    d = lvl_b - lvl_a
    consts = (half2(d), half2(lvl_a), half2(d / 4.0), half2(d / 16.0), half2(d / 64.0))
    views = (perm, consts)
    object.__setattr__(qw, "_lop3_views", views)
    return views


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def triton_gemv_mp_lop3(rx: torch.Tensor, qw) -> torch.Tensor:
    """All-passes 2-bit fused mat-vec with the packed-dot LOP3 decode.

    ``rx`` is the per-pass rotated activations ``(P, K)`` float32 (or float16);
    weights come from the cached lane-permuted view. Output ``(1, N)`` float32.
    Requires ``qw.bits == 2`` (symmetric 4-level codebook ``[-a, -b, b, a]``).
    """
    if not GEMV_MP_LOP3_AVAILABLE:
        raise RuntimeError("triton is not available")
    if qw.bits != 2:
        raise ValueError("the LOP3 decode is 2-bit only")
    words, (k1_0, k2c, k1_2, k1_4, k1_6) = lop3_views(qw)
    _, scale, _ = qw.decode_views()
    P, K = rx.shape
    N = words.shape[1]
    words_per_row = words.shape[2]
    group_size = qw.group_size
    vpw = 16
    wpg = group_size // vpw

    rxi = rx.to(torch.float16).view(torch.int32)  # (P, K//2) fp16 lane pairs

    block_n, block_w = 64, 32
    n_blocks = triton.cdiv(N, block_n)
    sms = _sm_count(rx.device.index or 0)
    max_chunks = max(1, words_per_row // wpg)
    split_k = min(max(1, (2 * sms) // max(1, n_blocks * P)), max_chunks)
    words_per_chunk = triton.cdiv(triton.cdiv(words_per_row, split_k), wpg) * wpg

    out = torch.zeros((1, N), dtype=torch.float32, device=rx.device)
    grid = (n_blocks, split_k * P)
    _gemv_mp_lop3_kernel[grid](
        rxi, words, scale, out,
        N, K // 2, words_per_row, words_per_chunk,
        rxi.stride(0),
        words.stride(0), words.stride(1),
        scale.stride(0), scale.stride(1), scale.stride(2),
        k1_0, k2c, k1_2, k1_4, k1_6,
        GROUP_SIZE=group_size, VPW=vpw,
        SPLIT_K=split_k, BLOCK_N=block_n, BLOCK_W=block_w,
        num_warps=4, num_stages=3,
    )
    return out
