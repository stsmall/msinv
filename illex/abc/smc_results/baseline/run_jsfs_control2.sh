#!/usr/bin/env bash
# Unfolded AA40 x BB40 2D SFS on the representative collinear control, chr2 40-55 + 85-115 Mb.
# Same individuals (04_angsd_chr2/bamlists/{AA,BB}40_bamlist.txt), sites (chr2.accessible.sites), -anc/-ref and filters
# as the inversion spectrum (inversion_sims .tmp/sfs2d/run.sh); the two intervals go in one SAF via -rf.
# Projection to 20 chromosomes per class and binning: build_observed_ctl2.py (observed.jsfs_from_2d).
#   bash illex/abc/smc_results/baseline/run_jsfs_control2.sh      # from the worktree root
set -uo pipefail
ANA=/sietch_colab/data_share/illex/popgen_data/analysis
REF=/sietch_colab/data_share/illex/andy_links/Illex_F24.primary.clean.fa
ANGSD=/home/ssmall/programs/angsd/angsd
REALSFS=/home/ssmall/programs/angsd/misc/realSFS
D=$ANA/steps/04_angsd_chr2; BL=$D/bamlists; SITES=$D/chr2.accessible.sites
O=${O:-$PWD/.tmp/abc_control2}
TH=${TH:-4}; TH2=${TH2:-8}
printf '2:40000001-55000000\n2:85000001-115000000\n' > "$O/ctl2.regions"
for g in AA BB; do
  out="$O/${g}40.ctl2"
  [ -s "$out.saf.idx" ] && continue
  "$ANGSD" -b "$BL/${g}40_bamlist.txt" -sites "$SITES" -rf "$O/ctl2.regions" \
    -anc "$REF" -ref "$REF" -out "$out" \
    -doSaf 1 -doCounts 1 -GL 1 -P "$TH" \
    -minInd 1 -setMinDepthInd 1 -minQ 20 -minMapQ 20 \
    -remove_bads 1 -only_proper_pairs 1 > "$out.angsd.log" 2>&1 &
done
wait
echo "[$(date +%T)] SAFs done"
"$REALSFS" "$O/AA40.ctl2.saf.idx" "$O/BB40.ctl2.saf.idx" -fold 0 -P "$TH2" -tole 1e-8 -maxIter 200 \
  > "$O/AA_BB.ctl2.2dSFS.part" 2> "$O/AA_BB.ctl2.log" && mv "$O/AA_BB.ctl2.2dSFS.part" "$O/AA_BB.ctl2.2dSFS"
echo "[$(date +%T)] 2dSFS entries=$(wc -w < "$O/AA_BB.ctl2.2dSFS")"
