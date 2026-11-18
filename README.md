# ApexQuant

Code for *ApexQuant: Data-Free Elastic Quantization by Residual Re-Isotropization*.

The package contains three directories:

- `apexquant/` holds the method itself: the three quantization stages (scalar, E8, trellis), the nested-pass recursion, the fused Triton kernels, the vLLM plugin, the batch-norm correction, and the scripts underlying every table and figure. Run `cd apexquant && make help` to list the available targets.
- `clay-model/` is a port of the Clay v1.5 Earth-observation encoder with Lightning removed and Triton kernels fused in. The vision results quantize this frozen model; `apexquant` installs it as an editable local package.
- `thms/` is the Lean 4 formalization of the theory section: 1057 machine-checked theorems, with no `sorry` and no custom axioms. `apexquant` does not import it. It proves the decay law underlying Propositions 1 and 2. Build it with `lake build` inside `thms/`.

Install the Python side with `uv sync` in `apexquant/`. Model weights, datasets, and result files are not included; the scripts download public checkpoints from Hugging Face on first use.
