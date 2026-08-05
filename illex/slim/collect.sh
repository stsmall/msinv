#!/bin/bash
# Concatenate per-task TSVs into one table and report the failure breakdown.
#
#   bash illex/slim/collect.sh results/abc results/abc/sims_all.tsv
#
# Keeps the header from the first file only. Reports status counts, because a
# high failure rate in a particular region of parameter space is a result, not
# just noise -- e.g. many "lost_too_often" rows at low p_start is direct evidence
# about how hard it is for a single-founder inversion to reach p = 0.626.

set -euo pipefail

DIR=${1:-results/abc}
OUT=${2:-$DIR/sims_all.tsv}

shopt -s nullglob
files=("$DIR"/sims_task*.tsv)
if [[ ${#files[@]} -eq 0 ]]; then
    echo "no sims_task*.tsv in $DIR" >&2
    exit 1
fi

head -1 "${files[0]}" > "$OUT"
for f in "${files[@]}"; do
    tail -n +2 "$f" >> "$OUT"
done

n=$(( $(wc -l < "$OUT") - 1 ))
echo "combined ${#files[@]} task files -> $OUT  ($n simulations)"
echo
echo "status breakdown:"
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="status") c=i; next} {print $c}' "$OUT" \
    | sort | uniq -c | sort -rn | sed 's/^/  /'
echo
echo "wall time (ok rows only): "
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++){if($i=="status")c=i; if($i=="slim_wall_s")w=i}; next}
            $c=="ok"{s+=$w; n++; if($w>mx)mx=$w}
            END{if(n) printf "  mean %.1fs  max %.1fs  total %.1f core-hours\n", s/n, mx, s/3600}' "$OUT"
