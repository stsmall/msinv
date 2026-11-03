#!/usr/bin/env bash
# realSFS inputs for illex/abc/observed.py (the observed ABC statistic vector).
#
#   bash illex/scripts/run_abc_observed_angsd.sh        # from the worktree root
#
# 1D (within-class, full n: AA 254 / BB 95 diploids) from the existing chr2 SAFs of steps/04_angsd_chr2:
#   body     2:BP_LEFT+1-BP_RIGHT  (60,550,001-79,520,000; 1-based, = [BP_LEFT, BP_RIGHT) 0-based)
#   blocks   19 x 1 Mb from BP_LEFT, the last truncated at BP_RIGHT (block jackknife)
#   control  2:10,000,001-30,000,000 (the empirical_windowed.py REGIONS control)
# 2D (AA x BB, 40 diploids per class = 80 chromosomes) from the AA40/BB40 SAFs of sfs2d_karyotype.py's run.sh.
# Those SAFs cover only 2:60.5-79.5 Mb, so the 2D body is 60,550,001-79,500,000 (20 kb short at the right).
#
# -fold 0 throughout: SAFs were built with -anc REF, so the spectra are REF-polarised ALT-count spectra; the
# projection is exact on unfolded counts and folding happens at the target size (sfs_shape.project_unfolded).
#
# Cores (shared box, <100): 1D xargs -P 6 x -P 6 = 36; 2D blocks xargs -P 4 x -P 8 = 32; 2D body -P 12.
set -uo pipefail
D=/sietch_colab/data_share/illex/popgen_data/analysis/steps/04_angsd_chr2
S2=/sietch_colab/ssmall/projects/msinv_dir/inversion_sims/files/.tmp/sfs2d
R=/home/ssmall/programs/angsd/misc/realSFS
O=${O:-$PWD/.tmp/abc_observed}
mkdir -p "$O"
BPL=60550000; BPR=79520000; BPR2=79500000; BLK=1000000

one1 () {  # class label region
  local g=$1 lab=$2 reg=$3
  local out="$O/${g}.${lab}.sfs"
  [ -s "$out" ] && { echo "[skip] $out"; return 0; }
  "$R" "$D/saf/${g}.2.saf.idx" -r "$reg" -fold 0 -P 6 -tole 1e-8 -maxIter 200 > "$out.part" 2> "$O/${g}.${lab}.log" \
    && mv "$out.part" "$out"
  echo "[$(date +%T)] 1D $g $lab entries=$(wc -w < "$out" 2>/dev/null)"
}
one2 () {  # label region threads
  local lab=$1 reg=$2 th=$3
  local out="$O/AA_BB.${lab}.2dSFS"
  [ -s "$out" ] && { echo "[skip] $out"; return 0; }
  "$R" "$S2/AA40.body.saf.idx" "$S2/BB40.body.saf.idx" -r "$reg" -fold 0 -P "$th" -tole 1e-8 -maxIter 200 \
    > "$out.part" 2> "$O/AA_BB.${lab}.log" && mv "$out.part" "$out"
  echo "[$(date +%T)] 2D $lab entries=$(wc -w < "$out" 2>/dev/null)"
}
export -f one1 one2; export D S2 R O

echo "[$(date +%T)] START  O=$O"
# 2D body (long, ~2 h) in the background
one2 body "2:$((BPL + 1))-$BPR2" 12 &
p2body=$!
# 2D blocks
{
  for ((lo = BPL; lo < BPR2; lo += BLK)); do
    hi=$((lo + BLK)); ((hi > BPR2)) && hi=$BPR2
    echo "b$lo 2:$((lo + 1))-$hi 8"
  done
} | xargs -P 4 -L 1 bash -c 'one2 "$@"' _ &
p2blk=$!
# 1D: body, control, blocks
{
  for g in AA BB; do
    echo "$g body 2:$((BPL + 1))-$BPR"
    echo "$g control 2:10000001-30000000"
    for ((lo = BPL; lo < BPR; lo += BLK)); do
      hi=$((lo + BLK)); ((hi > BPR)) && hi=$BPR
      echo "$g b$lo 2:$((lo + 1))-$hi"
    done
  done
} | xargs -P 6 -L 1 bash -c 'one1 "$@"' _
echo "[$(date +%T)] 1D ALL DONE"
wait $p2blk; echo "[$(date +%T)] 2D blocks ALL DONE"
wait $p2body; echo "[$(date +%T)] ALL DONE"
