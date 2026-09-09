#!/usr/bin/env bash
# af3lis bootstrap -- one-time setup after `git clone`.
#
#   bash bootstrap.sh            # create ./env/.conda and check the cluster side
#   bash bootstrap.sh --check    # verify only, create nothing
#
# Creates a self-contained conda env INSIDE the clone, so nothing depends on
# the user's shell config and a second clone cannot collide with the first.
# Writes env/activate.sh, which every SLURM template sources.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="$REPO/env/.conda"
YML="$REPO/env/af3lis.yml"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

say() { printf '\033[1m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[bootstrap]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[bootstrap]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- python env
find_mgr() {
  for m in micromamba mamba conda; do command -v "$m" >/dev/null 2>&1 && { echo "$m"; return; }; done
  for c in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" \
           "$HOME/mambaforge/bin/mamba" /opt/conda/bin/conda; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
}
if [ -x "$PREFIX/bin/python" ]; then
  say "env already present: $PREFIX"
elif [ "$CHECK_ONLY" -eq 1 ]; then
  warn "no env at $PREFIX (run without --check to create it)"
else
  MGR="$(find_mgr || true)"
  [ -n "$MGR" ] || die "no conda/mamba/micromamba found. Install miniforge, or create
  an env yourself with the packages in env/af3lis.yml and point AF3LIS_PYTHON at it."
  say "creating env with $MGR (a few minutes)..."
  if [[ "$MGR" == *micromamba* ]]; then
    "$MGR" create -y -p "$PREFIX" -f "$YML"
  else
    "$MGR" env create -y -p "$PREFIX" -f "$YML" 2>/dev/null \
      || "$MGR" env create -p "$PREFIX" -f "$YML"
  fi
fi

cat > "$REPO/env/activate.sh" <<EOF
# sourced by af3lis SLURM templates and by you, interactively.
# AF3LIS_PYTHON wins, so a hand-built env can override the bundled one.
export AF3LIS_DIR="$REPO"
if [ -n "\${AF3LIS_PYTHON:-}" ]; then
    :
elif [ -x "$PREFIX/bin/python" ]; then
    export AF3LIS_PYTHON="$PREFIX/bin/python"
    # keep conda's own libs first: a cluster 'module load' prepends system libs
    # to LD_LIBRARY_PATH and conda-built binaries then segfault.
    export LD_LIBRARY_PATH="$PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
else
    export AF3LIS_PYTHON="\$(command -v python3)"
fi
export PYTHONPATH="$REPO\${PYTHONPATH:+:\$PYTHONPATH}"
EOF
say "wrote env/activate.sh"

# shellcheck disable=SC1091
source "$REPO/env/activate.sh"
say "python: $AF3LIS_PYTHON"
"$AF3LIS_PYTHON" - <<'PY' || die "the env is missing required packages"
import importlib, sys
missing = [m for m in ("numpy", "scipy", "pandas", "matplotlib")
           if not importlib.util.find_spec(m)]
print("[bootstrap] packages:", "all present" if not missing else f"MISSING {missing}")
sys.exit(1 if missing else 0)
PY
"$AF3LIS_PYTHON" -c "import af3lis; print('[bootstrap] af3lis imports OK')"

# ------------------------------------------------------------- cluster side
say "checking the cluster side (warnings here are fine on a laptop)"
command -v sbatch >/dev/null 2>&1 && say "sbatch: $(command -v sbatch)" \
  || warn "no sbatch -- SLURM submission will not work from this machine"
if command -v module >/dev/null 2>&1 || [ -n "${MODULESHOME:-}" ]; then
  AF3MOD="${AF3LIS_AF3_MODULE:-bio/alphafold/3.0.1}"
  if module avail "$AF3MOD" 2>&1 | grep -q "${AF3MOD%%/*}"; then
    say "AF3 module visible: $AF3MOD"
  else
    warn "module '$AF3MOD' not found -- set af3_module in config.yaml"
  fi
fi
[ -d "${AF3LIS_MODEL_DIR:-$HOME/af3-models}" ] \
  && say "AF3 weights: ${AF3LIS_MODEL_DIR:-$HOME/af3-models}" \
  || warn "no AF3 weights at ${AF3LIS_MODEL_DIR:-$HOME/af3-models} -- these are
  gated by DeepMind and must be requested and placed there yourself"

say "done. Next:"
echo "    cp config.example.yaml config.yaml   # then fill in the cluster paths"
echo "    bash af3lis.sh run --chains-tsv chains.tsv --name MyScreen"
