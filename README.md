# ApexQuant

Reproduction package for *ApexQuant: Data-Free Elastic Quantization by
Residual Re-Isotropization*.

Three directories:

- **`apexquant/`** — the method itself: the three quantization stages
  (scalar, E8, trellis), the progressive/nested-pass recursion, the fused
  Triton kernels, the batch-norm correction, and the paper's own experiment
  scripts and Makefile targets. Start here: `cd apexquant && make help`.
- **`clay-model/`** — a de-Lightning'd, Triton-fused port of the Clay v1.5
  Earth-observation encoder, the real frozen model `apexquant`'s vision
  table evaluates. `apexquant` depends on it as an editable local package.
- **`thms/`** — the Lean 4 formalisation of the paper's theory section
  (1057 machine-checked theorems, no `sorry`, no custom axioms). Not code
  `apexquant` runs; it's the proof behind proposition 1/2's decay law.

See `apexquant/README.md`, `clay-model/README.md`, and `thms/README.md` for
each directory's own detail.
