# apexquant

Calibration-free elastic quantization by residual re-isotropization.

Reproduction package for *ApexQuant: Data-Free Elastic Quantization by Residual
Re-Isotropization*. See `Makefile` for every reproduction target (`make help`).

## The method

Quantizing a weight row is a sequence of passes. Each pass draws a fresh
randomized Hadamard rotation, normalizes the rotated residual onto the unit
sphere, rounds it with one of three interchangeable stages, and subtracts the
reconstruction before the next pass. Because the rotation is redrawn fresh
every time, the residual's marginal returns exactly to the density each
stage's codebook was built for (proposition 1), which gives a closed-form
geometric decay law for the error and makes every prefix of P passes a valid,
independently usable lower-rate model ("nested decoding").

The three stages, in increasing order of strength: scalar Lloyd-Max, an E8
lattice, and a Viterbi-coded trellis with register width L=14 (2^14 states,
accelerated by a fused Triton encoder). No stage needs calibration data; one
optional pass (`apexquant.fuse`) spends a leave-one-out sensitivity probe to
allocate extra passes where they help most.

## Quick start

```python
import torch
from apexquant import ApexQuantConfig, turboquantize_weight, dequantize_weight

weight = torch.randn(4096, 4096)
qw = turboquantize_weight(weight, ApexQuantConfig(bits=2, passes=2))
recon = dequantize_weight(qw)
```

Or turn a real Hugging Face checkpoint into a progressive artifact:

```bash
make convert MODEL=mistralai/Mistral-7B-v0.1 OUT=out/mistral7b-trellis-p3
```

## Layout

- `src/apexquant/` — the package:
  - `rotation/` — the randomized Hadamard transform (fused Triton kernel +
    torch fallback) and its inverse.
  - `codebook/` — the scalar reconstruction levels: the density-matched
    (`apex_beta_levels`), Lloyd-Max, uniform, l_p-fitted, and heavy-tailed
    variants, plus a reference trellis quantizer.
  - `quantize/` — the base scalar-quantization primitives (indices, packing).
  - `vq/` — the three stages proper (`scalar_codec`, `e8p_codec`,
    `trellis_codec`) and the progressive/nested-pass recursion
    (`progressive_artifact`, `progressive_trellis_linear`) that proposition 2
    is about.
  - `kernels/` — the fused dequant+matmul GEMM/GEMV Triton kernels, including
    the multi-pass and gemlite-forked mid-M paths nested decoding uses in
    service.
  - `bn/` — the data-free batch-norm correction (table 10, figure 5): the
    largest subsystem here, with its own conv/BN-site adapters, moment
    oracle, and correction driver.
  - `fuse/` — the optional calibration-pass allocators (appendix I): a
    leave-one-out sensitivity probe plus a greedy and a knapsack-DP allocator,
    gated by a validation pass that only keeps the allocation if it wins.
  - `analysis/`, `nn/`, `preprocess/`, `throughput/` — diagnostics, `nn.Linear`
    replacement, the SINQ/A-SINQ Sinkhorn scaling baseline, and the analytical
    roofline / CUDA-graph throughput measurement.
- `experiments/` — the paper's own measurement scripts, kept in scope where
  they target one of its tables or figures (see `docs`/plan notes for the
  full include/exclude accounting).
- `tests/` — the package's test suite, including a synthetic bit-identity
  check of the nested-decoding claim that needs no checkpoint download.

## Reproducing the paper

Every `make` target below is CPU-honest or fails fast: it either needs nothing
but this package, or it says exactly what's missing (CUDA, the `gemlite`
extra, a checkpoint) rather than silently producing a smaller number.

| target | what it reproduces |
|---|---|
| `make stages` | Table 2: the three stages' closed-form rho^2, ~5s, no download |
| `make rotation-check` | Appendix B: Hadamard vs. a true Haar draw, on a real LM |
| `make llm-table` | Table 3/7/8: LLM KL sweep across seven data-free arms + two calibrated baselines |
| `make vision-table` | Table 4/5: Clay v1.5 embedding error at P=1,2,3 |
| `make kernel-bench` | Table 6: the Triton Viterbi encoder's throughput |
| `make serving-bench` | Table 13: tiered-decode latency, five serving scenarios |
| `make serving-sweep` | the mix/batch-size sweep behind table 13's prose |
| `make convert` | turn any HF causal LM into a progressive artifact at a chosen stage/depth |

`make setup` installs everything (`triton`, `gemlite`, and the experiment
dependencies); `make preflight` reports what's actually available without
installing anything, so you know before a long run what will and won't work.

`make serving-bench`/`serving-sweep` are the one place this package still
depends on a forked third-party kernel (gemlite's GEMM_SPLITK, pinned in the
`gemlite` extra along with the exact Triton version it needs — a newer Triton
breaks its autotuner). Nothing else here needs it.

## Testing

```bash
uv sync --all-extras
uv run pytest
```

491 tests, CPU- and CUDA-covered, 0 skips on a machine with CUDA/Triton and
the `timm`/`pillow` extras installed.

## Scope

This package is a curated subset of a much larger research tree: it excludes
a separate, broader "unifying framework" thread (SpectralQuant, rate-distortion
bit allocation, spectrum-aware routing) that treats "ApexQuant" as an external
baseline rather than this paper's own construction, along with a synthetic
Earth-observation stand-in superseded here by the real `clay-model` sibling
package. See `../../.claude/plans/async-brewing-ember.md` for the full
inclusion/exclusion accounting and rationale.
