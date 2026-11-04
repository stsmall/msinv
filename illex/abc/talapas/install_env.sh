#!/bin/bash
# Build the Talapas environment for the Illex trajectory ABC-SMC (pyABC + redis + msinv).
#
# Creates (or updates) a conda env and a checkout of the illex/trajectory-abc branch, builds msinv from that
# checkout (Rust extension, via maturin), and verifies the install. Safe to re-run: it pulls the branch and rebuilds.
#
#   bash illex/abc/talapas/install_env.sh              # defaults below
#   ABC_ENV=illex_abc REPO=/gpfs/projects/kernlab/$USER/msinv_illex_abc bash install_env.sh
#
# Versions match the box env the pilots ran in (Python 3.12, numpy 2.4.4, scipy 1.17.1, pandas 3.0.2,
# msprime 1.4.1, tskit 1.0.2, pyarrow 26.0.1, pyabc 0.13.1, redis-py 8.1.0).
set -euo pipefail

ABC_ENV=${ABC_ENV:-illex_abc}
REPO=${REPO:-/gpfs/projects/kernlab/$USER/msinv_illex_abc}
BRANCH=${BRANCH:-illex/trajectory-abc}
GIT_URL=${GIT_URL:-https://github.com/stsmall/msinv.git}
BUILD_JOBS=${BUILD_JOBS:-8}

module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. conda env: interpreter, compiled scientific stack, redis-server, Rust toolchain
if ! conda env list | awk '{print $1}' | grep -qx "$ABC_ENV"; then
    conda create -y -n "$ABC_ENV" -c conda-forge --override-channels \
        python=3.12 "numpy=2.4" "scipy=1.17" "pandas=3.0" "pyarrow=26" "msprime=1.4.1" "tskit=1.0.2" \
        redis-server rust "maturin>=1.5" pip git
fi
conda activate "$ABC_ENV"
command -v redis-server >/dev/null || conda install -y -c conda-forge --override-channels redis-server

# 2. pip-only packages, constrained so pyabc's dependency tree cannot move the core stack
CONSTRAINTS=$(mktemp)
python - "$CONSTRAINTS" <<'EOF'
import sys, numpy, scipy, pandas, msprime, tskit, pyarrow
with open(sys.argv[1], "w") as fh:
    for m in (numpy, scipy, pandas, msprime, tskit, pyarrow):
        fh.write(f"{m.__name__}=={m.__version__}\n")
EOF
pip install -c "$CONSTRAINTS" "pyabc==0.13.1" "redis==8.1.0" pytest
rm -f "$CONSTRAINTS"

# 3. the code: a checkout of the branch (illex/ code + msinv source)
if [[ -d "$REPO/.git" ]]; then
    git -C "$REPO" diff --quiet || { echo "FATAL: $REPO has uncommitted changes; not touching it." >&2; exit 1; }
    git -C "$REPO" fetch origin
    git -C "$REPO" checkout "$BRANCH"
    git -C "$REPO" pull --ff-only origin "$BRANCH"
else
    mkdir -p "$(dirname "$REPO")"
    git clone --branch "$BRANCH" "$GIT_URL" "$REPO"
fi

# 4. build msinv (Rust extension) from the checkout, editable so PYTHONPATH=$REPO and the install agree
cd "$REPO"
CARGO_BUILD_JOBS=$BUILD_JOBS pip install -e . --no-deps

# 5. verify
python - <<'EOF'
import sys, numpy, scipy, pandas, msprime, tskit, pyarrow, pyabc, redis, msinv
print("python", sys.version.split()[0])
for m in (numpy, scipy, pandas, msprime, tskit, pyarrow, pyabc, redis, msinv):
    print(f"{m.__name__:8s} {m.__version__}")
from msinv._msinv_core import __file__ as core
print("msinv core", core)
v = tuple(int(x) for x in msinv.__version__.split(".")[:3])
assert v >= (0, 4, 0), "msinv must be >= 0.4.0 (crossover suppression)"
if v < (0, 4, 1):
    print("NOTE: msinv < 0.4.1 (flank-merge fix not in yet): fine for smoke tests, NOT for production runs.")
import illex.abc.smc  # noqa: F401  (imports the whole ABC stack)
print("illex.abc.smc imports OK")
EOF
redis-server --version
command -v abc-redis-worker abc-redis-manager
echo "install OK: env=$ABC_ENV repo=$REPO branch=$BRANCH ($(git -C "$REPO" rev-parse --short HEAD))"
