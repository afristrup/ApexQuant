#!/usr/bin/env bash
# verify.sh <slot>
#   Assembles work/<slot>/frags/*.lean into work/<slot>/asm.lean and type-checks it
#   against this project's Mathlib.  Prints errors and remaining sorries.
set -e
cd "$(dirname "$0")"
SLOT="$1"
[ -d "work/$SLOT/frags" ] || { echo "no work/$SLOT/frags -- run: mkdir -p work/$SLOT && cp -r frags work/$SLOT/"; exit 2; }
LC_ALL=C; export LC_ALL   # see assemble.sh
find "work/$SLOT/frags" -name '*.lean' -type f | sort | xargs cat > "work/$SLOT/asm.lean"
./check.sh work/$SLOT/asm.lean 2>&1 | sed "s|$PWD/work/$SLOT/asm.lean|ASM|"
echo "--- (no output above = clean; line numbers are into work/$SLOT/asm.lean) ---"
