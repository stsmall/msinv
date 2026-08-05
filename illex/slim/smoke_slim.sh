#!/bin/bash
# Smoke-test inversion_abc.slim BEFORE launching the array.
#
# Four single simulations at a large Q (seconds each) that between them exercise
# every branch of the recipe:
#   1. neutral, intermediate p_start   -- baseline
#   2. neutral, single-founder p_start -- the restart/conditioning machinery
#   3. overdominant (s>0, DOM>1)       -- the fitness path
#   4. neutral + flux                  -- the flux branch of recombination()
# Then summarizes each, exercising recapitation and the karyotype read.
#
#   bash illex/slim/smoke_slim.sh
#
# REQUIRES SLiM >= 5.0 (haplosome/tick API), same as the sweep recipes.
#
# What to look for: in the barrier cases pi_i/pi_s must be < 1 and dxy/pi_i > 1.
# If they come out ~1, recombination suppression is NOT working and nothing
# downstream is meaningful.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=${REPO:-$(cd "$HERE/../.." && pwd)}
PYTHON=${PYTHON:-python}
SLIM_BIN=${SLIM_BIN:-$(command -v slim || echo "$HOME/bin/slim")}
# Large Q = fast. NOT production: with the ABC prior's s_max=3e-4 the recipe
# aborts at SEL*Q>=0.1, so case 3 below uses a small s to stay valid at this Q.
QSCALE=${QSCALE:-1000}
SCRATCH=${TMPDIR:-$REPO/.tmp}/illex_inv_smoke

mkdir -p "$SCRATCH"

if [[ ! -x "$SLIM_BIN" ]]; then
    echo "FATAL: no SLiM binary at '$SLIM_BIN'. Set SLIM_BIN=/path/to/slim" >&2
    exit 1
fi
ver=$("$SLIM_BIN" -v 2>&1 | head -1)
echo "SLiM: $ver"
case "$ver" in
    *" 5."*) : ;;
    *) echo "WARNING: this recipe needs SLiM >= 5.0 (haplosome API). Got: $ver" >&2 ;;
esac
echo "Q=$QSCALE (smoke only; production Q=200)"
echo

run_case () {
    local name=$1 t_inv=$2 p_start=$3 sel=$4 dom=$5 p_flux=$6
    local trees="$SCRATCH/${name}.trees"
    echo "--- $name (t_inv=$t_inv p_start=$p_start s=$sel dom=$dom flux=$p_flux)"
    "$SLIM_BIN" -s 12345 \
        -d "Q=$QSCALE" -d "T_INV=$t_inv" -d "P_START=$p_start" \
        -d "SEL=$sel" -d "DOM=$dom" -d "P_FLUX=$p_flux" \
        -d "TRACT_FRAC=1e-4" -d "INV_LEN=100000" -d "FLANK_LEN=25000" \
        -d "R=2.52e-9" -d "MU=3e-9" \
        -d "NREF=547928.0" -d "N0=6808096.0" -d "TGROW=769519.0" \
        -d "OUTPATH=\"$trees\"" -d "SAVEPATH=\"$trees.restart\"" \
        "$HERE/inversion_abc.slim" 2>&1 | grep -E "INVERSION|ERROR|introduced" | tail -4
    if [[ -f "$trees" ]]; then
        ( cd "$REPO" && "$PYTHON" - "$trees" "$QSCALE" <<'PY'
import sys
from illex.slim.summarize import summarize
st = summarize(sys.argv[1], float(sys.argv[2]), 999)
print(f"    OK  pi_i/pi_s={st['pi_i_over_pi_s']:.4f}  "
      f"dxy/pi_i={st['dxy_over_pi_i']:.4f}  p_final={st['p_final']:.4f}  "
      f"restarts={st['n_restarts']}  trees={st['n_trees']:,}")
PY
        )
    else
        echo "    FAILED (no .trees produced)"
    fi
    rm -f "$trees" "$trees.restart"
    echo
}

run_case neutral_soft     800000 0.15      0.0    0.5 0.0
run_case neutral_founder  800000 9.126e-7  0.0    0.5 0.0
run_case overdominant     800000 0.05      1e-5   2.0 0.0
run_case with_flux        800000 0.15      0.0    0.5 1e-3

rm -rf "$SCRATCH"
echo "If pi_i/pi_s is ~1 in the barrier cases, recombination suppression is broken."
