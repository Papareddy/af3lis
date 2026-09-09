#!/usr/bin/env bash
# af3lis -- one-command AlphaFold-3 interface screening on SLURM.
#
#   bash af3lis.sh setup                          one-time, after git clone
#   bash af3lis.sh template chains.tsv            starter input file
#   bash af3lis.sh run    --chains-tsv chains.tsv --name MyScreen [opts]
#   bash af3lis.sh status MyScreen
#   bash af3lis.sh report MyScreen                re-score/re-plot a finished run
#
# `run` builds the inputs and submits the whole DAG in one go:
#     align array -> pack bridge -> packed GPU groups -> analyse
# and returns immediately. Nothing else is needed; metrics.tsv, figures/ and
# the PDB exports appear under the run directory when the last job finishes.
#
# Every option not listed below is passed straight through to pipeline.py, so
# `--seeds 1,2,3`, `--mode complex`, `--target-hours 2` etc all work here.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="${1:-help}"; shift || true

# shellcheck disable=SC1091
[ -r "$REPO/env/activate.sh" ] && source "$REPO/env/activate.sh"
PY="${AF3LIS_PYTHON:-python3}"
CFG="${AF3LIS_CONFIG:-$REPO/config.yaml}"

die() { printf '\033[31m[af3lis]\033[0m %s\n' "$*" >&2; exit 1; }
say() { printf '\033[1m[af3lis]\033[0m %s\n' "$*"; }

runs_root() {
  # RUNS_ROOT, else the workspace from config.yaml, else ./runs
  if [ -n "${AF3LIS_RUNS_ROOT:-}" ]; then echo "$AF3LIS_RUNS_ROOT"; return; fi
  local ws
  ws="$("$PY" - "$CFG" <<'PY' 2>/dev/null || true
import sys, os
try:
    import yaml
    c = yaml.safe_load(open(sys.argv[1])) or {}
except Exception:
    sys.exit(0)
w = c.get("workspace") or ""
print(os.path.join(w, "runs") if w else "")
PY
)"
  [ -n "$ws" ] && echo "$ws" || echo "$REPO/runs"
}

case "$CMD" in
  setup)
    bash "$REPO/bootstrap.sh" "$@" ;;

  template)
    OUT="${1:-chains.tsv}"
    "$PY" "$REPO/pipeline.py" --chains-tsv-template "$OUT" ;;

  run)
    [ -r "$CFG" ] || die "no config at $CFG -- cp config.example.yaml config.yaml and fill it in"
    NAME=""; ARGS=(); OUTDIR=""; DRY=0
    while [ $# -gt 0 ]; do
      case "$1" in
        --name)   NAME="$2";   ARGS+=("$1" "$2"); shift 2 ;;
        --outdir) OUTDIR="$2"; ARGS+=("$1" "$2"); shift 2 ;;
        --dry-run) DRY=1; ARGS+=("$1"); shift ;;
        *) ARGS+=("$1"); shift ;;
      esac
    done
    [ -n "$NAME" ] || die "--name is required"
    if [ -z "$OUTDIR" ]; then
      OUTDIR="$(runs_root)/$NAME"
      ARGS+=(--outdir "$OUTDIR")
    fi
    say "run dir: $OUTDIR"
    say "1/2 building inputs"
    "$PY" "$REPO/pipeline.py" --config "$CFG" "${ARGS[@]}"
    say "2/2 submitting the DAG (align -> pack bridge -> GPU groups -> analyse)"
    SUB=(--submit --outdir "$OUTDIR" --config "$CFG")
    [ "$DRY" -eq 1 ] && SUB+=(--dry-run)
    "$PY" "$REPO/pipeline.py" "${SUB[@]}"
    say "submitted. watch it with:  bash af3lis.sh status $NAME" ;;

  status)
    NAME="${1:-}"; [ -n "$NAME" ] || die "usage: af3lis.sh status <run-name>"
    D="$(runs_root)/$NAME"
    [ -d "$D" ] || die "no run dir at $D"
    echo "== queue =="
    command -v squeue >/dev/null && squeue -u "$USER" -o "%.10i %.24j %.10P %.9T %.11M %R" || echo "(no squeue)"
    echo; echo "== on disk =="
    printf '  MSAs (align out) : %s\n' "$(find "$D/out" -maxdepth 2 -name '*_data.json' 2>/dev/null | wc -l)"
    printf '  models (.cif)    : %s\n' "$(find "$D/out_pack" "$D/out" -maxdepth 3 -name '*model*.cif' 2>/dev/null | wc -l)"
    printf '  metrics.tsv rows : %s\n' "$( [ -s "$D/metrics.tsv" ] && echo $(($(wc -l < "$D/metrics.tsv")-1)) || echo 0 )"
    printf '  figures          : %s\n' "$(find "$D/figures" -name '*.png' 2>/dev/null | wc -l)"
    printf '  PDB exports      : %s\n' "$(find "$D" -name '*.pdb' 2>/dev/null | wc -l)"
    echo; echo "== recent log lines =="
    grep -hE "MISSING:|ERROR|EXIT=|PACK_RESOURCES|ANALYSE_SUMMARY" "$D"/logs/*.out 2>/dev/null | tail -12 || echo "(no logs yet)" ;;

  report)
    NAME="${1:-}"; [ -n "$NAME" ] || die "usage: af3lis.sh report <run-name>"
    D="$(runs_root)/$NAME"
    OUT="$D/out_pack"; [ -d "$OUT" ] || OUT="$D/out"
    [ -d "$OUT" ] || die "no model output under $D"
    say "scoring $OUT"
    "$PY" "$REPO/pipeline.py" --collect "$OUT" -o "$D/metrics.tsv" --agg per_seed
    "$PY" -m af3lis.plots "$D/metrics.tsv" --out-dir "$OUT" --figdir "$D/figures"
    "$PY" -m af3lis.cif2pdb "$OUT"
    say "metrics: $D/metrics.tsv"; say "figures: $D/figures" ;;

  help|-h|--help)
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
  *)
    die "unknown command '$CMD' (try: setup, template, run, status, report)" ;;
esac
