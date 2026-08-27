#!/usr/bin/env bash
# Align three chr2 regions (inversion body + two collinear controls) directly to
# the coindetii assembly, instead of reading divergence off the pre-existing
# whole-genome AnchorWave alignment.
#
# WHY: div(ill,coin) read off that alignment is not stable along chr2 -- 0.0100
# inside the inversion against 0.0340 and 0.0201 in two collinear regions, i.e.
# the two CONTROLS disagree with each other by 70%. That is the signature of
# alignment ascertainment: substitutions are only counted where alignment
# succeeded, and alignment succeeds preferentially where divergence is low, so
# the estimate is deflated by an amount that tracks local repeat content.
#
# A direct, purpose-built alignment fixes this because it reports the ALIGNED
# FRACTION per region alongside the divergence, so the ascertainment is
# measured rather than hidden.
set -uo pipefail
W=/sietch_colab/ssmall/projects/msinv_dir/inversion_sims/files/.tmp/mmcoin
C=/sietch_colab/data_share/illex/alignments/genomes/Illex_coindetii/GCA_977009265.1_xcIllCoin1.1_genomic.fna
echo "[$(date +%T)] minimap2 start"
# asm10: expected divergence is ~1-3%, so asm20 would be needlessly permissive
# and asm5 might drop the more divergent blocks -- which is exactly the
# ascertainment being diagnosed.
/home/ssmall/bin/minimap2 -x asm10 -t 6 --secondary=no -c "$C" "$W/q.fa" \
  > "$W/aln.paf" 2> "$W/mm.log"
echo "[$(date +%T)] rc=$?  paf lines: $(wc -l < $W/aln.paf)"
