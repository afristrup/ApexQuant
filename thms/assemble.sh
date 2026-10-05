#!/usr/bin/env bash
# assemble.sh <fragdir> <out.lean>   -- concatenate fragments in name order
#
# Fragments nest one directory deep (NN_section/KK_declname.lean, one
# declaration per file); 00_header.lean and 99_footer.lean stay flat since
# they carry no declarations to split. find+sort reconstructs the same
# order a flat glob gave before nesting: "01_x/00_....lean" sorts right
# after "00_header.lean" and right before "02_y/...", exactly as
# "01_x.lean" did.
set -e
LC_ALL=C; export LC_ALL   # sort order must be byte order, not locale collation
find "$1" -name '*.lean' -type f | sort | xargs cat > "$2"
