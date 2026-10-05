#!/usr/bin/env bash
# Type-check a standalone Lean file against this project's Mathlib, without Lake.
#   ./check.sh <file.lean>
# LEAN_PATH is built from this script's own directory, not hardcoded, so a
# copy of thms/ (e.g. ApexQuant/thms/) points at its own .lake once
# `lake exe cache get && lake build` has been run there -- see README.md.
ROOT="$(cd "$(dirname "$0")" && pwd)"
PKG="$ROOT/.lake/packages"
export LEAN_PATH="$PKG/Cli/.lake/build/lib/lean:$PKG/batteries/.lake/build/lib/lean:$PKG/Qq/.lake/build/lib/lean:$PKG/aesop/.lake/build/lib/lean:$PKG/proofwidgets/.lake/build/lib/lean:$PKG/importGraph/.lake/build/lib/lean:$PKG/LeanSearchClient/.lake/build/lib/lean:$PKG/plausible/.lake/build/lib/lean:$PKG/mathlib/.lake/build/lib/lean:$ROOT/.lake/build/lib/lean"
exec ~/.elan/toolchains/leanprover--lean4---v4.33.0-rc1/bin/lean "$@"
