#!/usr/bin/env bash
# Per-karyotype folded SFS from the EXISTING chr2 ANGSD SAFs (built 2026-07-04,
# steps/04_angsd_chr2). GL-based, so it is immune to the called-genotype
# ascertainment that defeated the VCF-based spectrum (NOTES sec 8.4).
# -fold 0: the SAF was built with -anc REF, so the "unfolded" spectrum is the
# reference-polarised ALT-count spectrum. Projecting that exactly and folding
# only at the target size is the identical transform applied to the model side,
# so mis-polarisation cancels.
set -uo pipefail
D=/sietch_colab/data_share/illex/popgen_data/analysis/steps/04_angsd_chr2
R=/home/ssmall/programs/angsd/misc/realSFS
O=/sietch_colab/ssmall/projects/msinv_dir/inversion_sims/files/.tmp/angsd_sfs
one () {  # class region_label region
  local g=$1 lab=$2 reg=$3
  echo "[$(date +%T)] START $g/$lab $reg"
  "$R" "$D/saf/${g}.2.saf.idx" -r "$reg" -fold 0 -P 12 -tole 1e-8 -maxIter 200 \
     > "$O/${g}.${lab}.sfs" 2> "$O/${g}.${lab}.log"
  echo "[$(date +%T)] DONE  $g/$lab  entries=$(wc -w < "$O/${g}.${lab}.sfs")"
}
export -f one; export D R O
printf '%s\n' \
  "BB body 2:60500000-79500000" \
  "BB control 2:10000000-30000000" \
  "AA body 2:60500000-79500000" \
  "AA control 2:10000000-30000000" \
 | xargs -P 4 -I{} bash -c 'one $@' _ {}
echo "[$(date +%T)] ALL DONE"
