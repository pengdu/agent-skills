#!/usr/bin/env bash
# Static evidence for a CUDA source file: registers, spills, shared memory, and
# the load/store instruction mix from the generated SASS.
#
# This replaces the NCU metrics for registers-per-thread, instruction mix and
# (indirectly) coalescing. It is where hypotheses come from -- it is NOT where
# they are confirmed. A byte-wise gather looks catastrophic in SASS and
# frequently costs nothing, because L1 absorbs the redundancy and the DRAM
# traffic is identical. Confirm with an ablation or A/B before editing.
#
# Usage:
#   static_evidence.sh <file.cu> [arch] [extra nvcc flags...]
#
# Examples:
#   static_evidence.sh src/cuda/kernel.cu sm_75
#   static_evidence.sh src/cuda/kernel.cu sm_80 -I include -std=c++14 -DUSE_CUDA

set -u

if [ $# -lt 1 ]; then
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

SRC=$1
shift
ARCH=${1:-sm_75}
if [ $# -gt 0 ]; then shift; fi
EXTRA=("$@")

if [ ! -f "$SRC" ]; then
    echo "no such file: $SRC" >&2
    exit 1
fi

NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null || { echo "nvcc not found; set NVCC=" >&2; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
OBJ="$TMP/obj.o"

echo "file : $SRC"
echo "arch : $ARCH"
echo "flags: ${EXTRA[*]:-<none>}"
echo

echo "== registers / shared memory / spills (ptxas -v)"
echo "   spills are the red flag: any 'spill stores' means the kernel lost"
echo "   registers to local memory, which is DRAM traffic the source does not show."
echo
"$NVCC" -O3 -arch="$ARCH" -Xptxas -v -c "$SRC" -o "$OBJ" "${EXTRA[@]}" 2>&1 \
    | grep -E "Compiling entry|Used .* registers|spill" \
    | sed -e 's/^ptxas info *: *//' -e 's/^Compiling entry function/\nkernel: /' \
    | sed 's/^/   /'

if [ ! -f "$OBJ" ]; then
    echo
    echo "compile failed -- add the include paths and defines the project uses as extra flags" >&2
    exit 1
fi

echo
echo
echo "== global memory instruction mix (cuobjdump -sass)"
echo "   LDG/STG suffixes are the access width per instruction:"
echo "     .U8 / .S8   1 byte     scalar byte access"
echo "     .U16/.S16   2 bytes"
echo "     (none)      4 bytes"
echo "     .64         8 bytes    float2 / uint2"
echo "     .128       16 bytes    float4 / uint4  <- widest, fewest instructions"
echo "   Many narrow accesses per thread is a vectorisation *candidate*, not a"
echo "   diagnosis. Also watch for LDL/STL, which are register spills."
echo

command -v cuobjdump >/dev/null || { echo "   cuobjdump not found" >&2; exit 0; }

cuobjdump -sass "$OBJ" > "$TMP/sass.txt" 2>/dev/null

awk '
  /Function : / { fn = $NF; next }
  # Only real instruction lines carry an /*offset*/ prefix, so mangled names in
  # headers cannot be mistaken for opcodes.
  /\/\*[0-9a-f][0-9a-f]*\*\// {
    if (match($0, /(LDG|STG|LDL|STL|LDS|STS)\.?[.A-Za-z0-9_]*/)) {
      op = substr($0, RSTART, RLENGTH)
      sub(/\.E\./, ".", op)
      sub(/\.E$/, "", op)
      sub(/\.(SYS|STRONG|CONSTANT|GPU|CTA)$/, "", op)
      count[fn "|" op]++
    }
  }
  END { for (k in count) print count[k] "|" k }
' "$TMP/sass.txt" | sort -t'|' -k2,2 -k1,1nr | awk -F'|' '
  { if ($2 != last) { printf "\n   kernel: %s\n", Short($2); last = $2 }
    printf "     %6d  %s\n", $1, $3 }
  function Short(s) {
    # Template instantiations of the same kernel differ only in the tail, so
    # keep enough of it to tell them apart.
    if (length(s) > 72) return "..." substr(s, length(s) - 68)
    return s
  }
'

echo
echo "   read/write instruction ratio per kernel (LDG total vs STG total):"
awk '
  /Function : / { fn = $NF; next }
  /\/\*[0-9a-f][0-9a-f]*\*\// {
    if (match($0, /LDG[.A-Za-z0-9_]*/)) ld[fn]++
    else if (match($0, /STG[.A-Za-z0-9_]*/)) st[fn]++
  }
  END {
    for (f in ld) printf "     LDG=%-5d STG=%-5d  %s\n", ld[f], st[f] + 0, \
      (length(f) > 72 ? "..." substr(f, length(f) - 68) : f)
  }
' "$TMP/sass.txt" | sort -k1,1

echo
echo "   total instruction count per kernel:"
awk '
  /Function : / { fn = $NF; next }
  /\/\*[0-9a-f][0-9a-f]*\*\// { n[fn]++ }
  END {
    for (f in n) printf "     %6d  %s\n", n[f], \
      (length(f) > 72 ? "..." substr(f, length(f) - 68) : f)
  }
' "$TMP/sass.txt" | sort -k1,1nr
