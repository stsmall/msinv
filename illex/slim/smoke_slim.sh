#!/bin/bash
# Smoke-test the SLiM model BEFORE launching a 2000-simulation array.
#
# Runs three single simulations at a deliberately large Q (cheap, seconds each)
# that between them exercise every branch of inversion_abc.slim:
#   1. neutral, intermediate p_start   -- the baseline case
#   2. neutral, single-founder p_start -- the restart/conditioning machinery
#   3. overdominant (s>0, h>1)         -- the fitness callback and the s*Q guard
# Then summarizes each, which exercises recapitation + the karyotype read.
#
# Run this on Talapas (or anywhere SLiM is installed):
#   bash illex/slim/smoke_slim.sh
#
# Expected: three "status=ok" lines with plausible ratios. Anything else means
# fix the model before spending cluster time.

set -uo pipefail

REPO=${REPO:-$(pwd)}
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
SLIM_BIN=${SLIM_BIN:-$(command -v slim || echo "$HOME/bin/slim")}
QSCALE=${QSCALE:-2000}     # large Q = fast; NOT for production
SCRATCH=${TMPDIR:-./.tmp}/illex_smoke
OUT=${OUT:-./.tmp/illex_smoke_out}

mkdir -p "$SCRATCH" "$OUT"

if [[ ! -x "$SLIM_BIN" ]]; then
    echo "FATAL: no SLiM binary at '$SLIM_BIN'. Set SLIM_BIN=/path/to/slim" >&2
    exit 1
fi
echo "SLiM: $($SLIM_BIN -v 2>&1 | head -1)"
echo "Q=$QSCALE (smoke only -- production Q is 200; see README cost table)"
echo

# NOTE Q=2000 with the default s prior would trip the s*Q >= 0.1 guard for large
# s, which is correct behaviour. Case 3 uses a small s so it runs at this Q.
run_case () {
    local name=$1 t_inv=$2 p_start=$3 s=$4 h=$5 p_flux=$6
    local trees="$SCRATCH/${name}.trees"
    echo "--- case: $name (t_inv=$t_inv p_start=$p_start s=$s h=$h flux=$p_flux)"
    "$SLIM_BIN" \
        -d "Q=$QSCALE" -d "T_INV=$t_inv" -d "P_START=$p_start" \
        -d "S_COEF=$s" -d "H_COEF=$h" -d "P_FLUX=$p_flux" \
        -d "TRACT_FRAC=1e-4" -d "INV_LEN=100000" -d "FLANK_LEN=25000" \
        -d "REC_RATE=2.52e-9" -d "N_ANC=547928" -d "N_NOW=6808096" \
        -d "T_GROW=769519" -d "SEED=12345" \
        -d "TREES_PATH=\"$trees\"" \
        "$REPO/illex/slim/inversion_abc.slim" 2>&1 | tail -5
    if [[ -f "$trees" ]]; then
        "$PYTHON" - "$trees" "$QSCALE" <<'PY'
import sys
from illex.slim.summarize import summarize
st = summarize(sys.argv[1], float(sys.argv[2]), 999)
print(f"    status=ok  pi_i/pi_s={st['pi_i_over_pi_s']:.4f}  "
      f"dxy/pi_i={st['dxy_over_pi_i']:.4f}  p_final={st['p_final']:.4f}  "
      f"restarts={st['n_restarts']}  trees={st['n_trees']:,}")
PY
    else
        echo "    status=FAILED (no .trees produced)"
    fi
    rm -f "$trees" "$trees.restart"
    echo
}

cd "$REPO"
run_case neutral_soft      800000 0.15      0       0.5 0
run_case neutral_founder   800000 9.126e-7  0       0.5 0
run_case overdominant      800000 0.05      1e-5    2.0 0
run_case with_flux         800000 0.15      0       0.5 1e-3

echo "Smoke test complete. Check that pi_i/pi_s < 1 and dxy/pi_i > 1 in the"
echo "barrier cases -- if they are ~1, recombination suppression is not working."
rm -rf "$SCRATCH"
