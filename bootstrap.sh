#!/usr/bin/env bash
# af3lis bootstrap -- one-time setup after `git clone`.
#
#   bash bootstrap.sh            create the analysis env, check the cluster side
#   bash bootstrap.sh --check    verify only, create nothing
#
# Builds the python env used for building inputs, packing, scoring and
# plotting. AlphaFold 3 itself is NOT installed here -- it comes from the
# cluster module, because its JAX/CUDA build has to match the site's drivers
# and the weights are gated by DeepMind.
#
# WHERE THE ENV GOES, and why it is not inside the clone.
#   conda populates a prefix by HARDLINKING out of its package cache. On
#   parallel filesystems (GPFS on bwHelix) $HOME and a /work workspace are
#   separate filesets, and hardlinks across filesets fail -- conda then dies
#   mid-transaction and cannot even unlink its own half-written files. So the
#   default prefix sits next to the package cache under $HOME, NOT in the
#   clone. Override with AF3LIS_ENV_PREFIX if you know better for your site.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YML="$REPO/env/af3lis.yml"
PREFIX="${AF3LIS_ENV_PREFIX:-$HOME/.af3lis/env}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

say()  { printf '\033[1m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[bootstrap]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[bootstrap]\033[0m %s\n' "$*" >&2; exit 1; }

find_mgr() {
  for m in micromamba mamba conda; do
    command -v "$m" >/dev/null 2>&1 && { echo "$m"; return; }
  done
  for c in "$HOME/bin/micromamba" "$HOME/miniforge3/bin/mamba" \
           "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" \
           "$HOME/mambaforge/bin/mamba" /opt/conda/bin/conda; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
}

# Can we hardlink from $1 into $2? The only reliable same-fileset test.
can_hardlink() {
  local a="$1/.af3lis_ln_$$" b="$2/.af3lis_ln_$$"
  mkdir -p "$1" "$2" 2>/dev/null || return 1
  : > "$a" 2>/dev/null || return 1
  if ln "$a" "$b" 2>/dev/null; then rm -f "$a" "$b"; return 0; fi
  rm -f "$a" "$b" 2>/dev/null || true
  return 1
}

if [ -x "$PREFIX/bin/python" ]; then
  say "env already present: $PREFIX"
elif [ "$CHECK_ONLY" -eq 1 ]; then
  warn "no env at $PREFIX (run without --check to create it)"
else
  MGR="$(find_mgr || true)"
  [ -n "$MGR" ] || die "no conda/mamba/micromamba on PATH. Install miniforge, or
  build an env yourself from env/af3lis.yml and point AF3LIS_PYTHON at it."
  say "manager: $MGR"
  say "prefix:  $PREFIX"

  # Keep the package cache on the SAME fileset as the prefix, so conda can
  # hardlink. Without this the install dies on GPFS (see header).
  PKGS="$(dirname "$PREFIX")/pkgs"
  COPY_FLAG=""
  if can_hardlink "$(dirname "$PREFIX")" "$(dirname "$PREFIX")"; then
    export CONDA_PKGS_DIRS="$PKGS"
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$(dirname "$PREFIX")}"
    say "package cache: $PKGS (same fileset as the prefix)"
  else
    # Exotic filesystem with no hardlink support at all -- copy instead. Uses
    # more space but always works.
    COPY_FLAG="--copy"
    warn "this filesystem does not support hardlinks; installing with --copy"
  fi
  mkdir -p "$(dirname "$PREFIX")"

  say "creating env (a few minutes; conda writes a lot of small files)"
  set +e
  case "$MGR" in
    *micromamba*) "$MGR" create -y -p "$PREFIX" -f "$YML" ;;
    *)            "$MGR" env create -y -p "$PREFIX" -f "$YML" $COPY_FLAG ;;
  esac
  rc=$?
  set -e
  if [ "$rc" -ne 0 ] || [ ! -x "$PREFIX/bin/python" ]; then
    warn "env-file install failed (rc=$rc); retrying as an explicit package list"
    rm -rf "$PREFIX"
    set +e
    "$MGR" create -y -p "$PREFIX" -c conda-forge $COPY_FLAG \
        python=3.11 numpy scipy pandas matplotlib-base pyyaml pytest
    rc=$?
    set -e
  fi
  [ -x "$PREFIX/bin/python" ] || die "could not create the env at $PREFIX (rc=$rc).
  Build one by hand from env/af3lis.yml and re-run with:
      AF3LIS_PYTHON=/path/to/python bash bootstrap.sh --check"
fi

cat > "$REPO/env/activate.sh" <<EOF
# sourced by the af3lis SLURM templates and by you, interactively.
# AF3LIS_PYTHON wins, so a hand-built env can override the bundled one.
export AF3LIS_DIR="$REPO"
if [ -n "\${AF3LIS_PYTHON:-}" ]; then
    :
elif [ -x "$PREFIX/bin/python" ]; then
    export AF3LIS_PYTHON="$PREFIX/bin/python"
    # Keep conda's own libs first: a cluster 'module load' prepends system
    # libs to LD_LIBRARY_PATH and conda-built binaries then segfault.
    export LD_LIBRARY_PATH="$PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
else
    export AF3LIS_PYTHON="\$(command -v python3)"
fi
export PYTHONPATH="$REPO\${PYTHONPATH:+:\$PYTHONPATH}"
EOF
say "wrote env/activate.sh -> $PREFIX"

# shellcheck disable=SC1091
source "$REPO/env/activate.sh"
say "python: $AF3LIS_PYTHON ($("$AF3LIS_PYTHON" -V 2>&1))"
"$AF3LIS_PYTHON" - <<'PY' || die "the env is missing required packages"
import importlib.util as u, sys
missing = [m for m in ("numpy", "scipy", "pandas", "matplotlib")
           if not u.find_spec(m)]
print("[bootstrap] packages:", "all present" if not missing else f"MISSING {missing}")
sys.exit(1 if missing else 0)
PY
"$AF3LIS_PYTHON" -c "import af3lis; print('[bootstrap] af3lis imports OK')"

# ------------------------------------------------------------- cluster side
say "checking the cluster side (warnings are fine on a laptop)"
command -v sbatch >/dev/null 2>&1 && say "sbatch: $(command -v sbatch)" \
  || warn "no sbatch -- SLURM submission will not work from this machine"
if [ -n "${MODULESHOME:-}" ] || command -v module >/dev/null 2>&1; then
  AF3MOD="${AF3LIS_AF3_MODULE:-bio/alphafold/3.0.1}"
  if module avail "$AF3MOD" 2>&1 | grep -q "${AF3MOD%%/*}"; then
    say "AF3 module visible: $AF3MOD"
  else
    warn "module '$AF3MOD' not found -- set af3_module in config.yaml"
  fi
fi
MD="${AF3LIS_MODEL_DIR:-$HOME/af3-models}"
[ -d "$MD" ] && say "AF3 weights: $MD" \
  || warn "no AF3 weights at $MD -- gated by DeepMind; request them and put
  the params there, then point model_dir in config.yaml at that DIRECTORY"

say "done. Next:"
echo "    cp config.example.yaml config.yaml     # fill in the cluster paths"
echo "    bash af3lis.sh smoke                   # 12-fold end-to-end check"
