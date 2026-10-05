# Lean 4 formalisation

Machine-checked rendering of the named results in the ApexQuant paper, containing
every theorem referenced in the paper's theory section and proofs appendix.

## Build

```bash
lake exe cache get && lake build
```

Lean 4.33.0-rc1, Mathlib pinned at `v4.33.0-rc1` (see `lean-toolchain`, `lakefile.toml`).
A clean `lake build` prints no errors and no `declaration uses 'sorry'`. All public
theorems `#print axioms` to `propext, Classical.choice, Quot.sound` only.

## Layout

| file | role |
| --- | --- |
| `contributions.lean` | the development — see the header comment for what is proved |
| `frags/` | the same file split one declaration per file; `contributions.lean` is their concatenation |
| `assemble.sh` | `find frags -name '*.lean' | sort | xargs cat > out.lean` |

## Scope

The development covers the core theory (re-isotropisation, geometric decay, optimality,
block marginals, the bridge from abstract hypotheses to O(d)) and two auxiliary results
(uniqueness of the rotation-invariant measure and the block-form constant). Results not
referenced in the paper — including the per-pass concentration bound, the deployed-rotation
formalisation, and the detailed Haar-versus-Hadamard gap analysis — are not included.
