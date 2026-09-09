#!/usr/bin/env bash
# af3lis -- one-command AlphaFold-3 interface screening on SLURM.
#
#   bash af3lis.sh setup                          one-time, after git clone
#   bash af3lis.sh template chains.tsv            starter input file
#   bash af3lis.sh run    --chains-tsv chains.tsv --name MyScreen [opts]
#   bash af3lis.sh status MyScreen
#   bash af3lis.sh report MyScreen                re-score/re-plot a finished run
#   bash af3lis.sh smoke                          12-fold end-to-end verification
#   bash af3lis.sh check  smoke                   assert a finished run is complete
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

  smoke)
    # End-to-end verification on a new cluster. 12 small folds spanning two
    # token buckets; ~hours for the MSAs, minutes of GPU. Use this BEFORE
    # committing real GPU hours to a screen.
    [ -r "$CFG" ] || die "no config at $CFG -- cp config.example.yaml config.yaml and fill it in"
    NAME="${1:-smoke}"; shift || true
    D="$(runs_root)/$NAME"
    say "smoke run '$NAME' -> $D"
    say "12 folds (2 baits x 6 preys) from examples/smoke.tsv"
    "$PY" "$REPO/pipeline.py" --config "$CFG" --name "$NAME" --outdir "$D" \
        --chains-tsv "$REPO/examples/smoke.tsv" \
        --fasta "$REPO/examples/smoke.fasta" "$@"
    "$PY" "$REPO/pipeline.py" --submit --outdir "$D" --config "$CFG"
    cat <<EOF

[af3lis] submitted. When it finishes, verify with:

    bash af3lis.sh check $NAME

To also prove the afterany/MISSING path works (recommended), delete one MSA
after the align array finishes and before the bridge job starts:

    rm -rf $D/out/<one-pair>/

The bridge should exclude it, print a MISSING: line, and the analyse job
should still score the other 11.
EOF
    ;;

  check)
    # Self-check a finished run. Exits non-zero on the first hard failure, so
    # it is usable in CI or a wrapper script.
    NAME="${1:-smoke}"
    D="$(runs_root)/$NAME"
    [ -d "$D" ] || die "no run dir at $D"
    OUT="$D/out_pack"; [ -d "$OUT" ] || OUT="$D/out"
    fail=0
    chk() { # chk <label> <actual> <expected-min>
      if [ "$2" -ge "$3" ] 2>/dev/null; then
        printf "  \033[32mPASS\033[0m %-34s %s (>= %s)\n" "$1" "$2" "$3"
      else
        printf "  \033[31mFAIL\033[0m %-34s %s (want >= %s)\n" "$1" "$2" "$3"; fail=1
      fi
    }
    echo "== af3lis check: $NAME =="
    NPAIR=$(find "$D/jsons" -name '*.json' 2>/dev/null | wc -l)
    chk "input JSONs built"        "$NPAIR" 1
    chk "MSAs on disk"             "$(find "$D/out" -maxdepth 2 -name '*_data.json' 2>/dev/null | wc -l)" 1
    chk "models (.cif)"            "$(find "$OUT" -maxdepth 3 -name '*model*.cif' 2>/dev/null | wc -l)" 1
    chk "confidences.json"         "$(find "$OUT" -maxdepth 3 -name '*confidences.json' 2>/dev/null | wc -l)" 1
    chk "PDB exports"              "$(find "$OUT" -maxdepth 3 -name '*.pdb' 2>/dev/null | wc -l)" 1
    chk "metrics.tsv rows"         "$( [ -s "$D/metrics.tsv" ] && echo $(($(wc -l < "$D/metrics.tsv")-1)) || echo 0 )" 1
    chk "figures"                  "$(find "$D/figures" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)" 3
    chk "PAE panels"               "$(find "$D/figures/pae" -name '*.png' 2>/dev/null | wc -l)" 1
    # Packed grouping. NOTE: the smoke fixture is deliberately small, so both
    # of its buckets sit on the same 48gb tier -- ONE distinct --mem is the
    # correct result here. The >=4096-token escalation is covered by
    # tests/test_pack.py::test_resources_for_ladder, not by this run.
    if [ -s "$D/af3pack/groups.tsv" ]; then
      NMEM=$(awk -F'\t' 'NR>1{print $7}' "$D/af3pack/groups.tsv" | sort -u | wc -l)
      NBUCK=$(awk -F'\t' 'NR>1{print $2}' "$D/af3pack/groups.tsv" | sort -u | wc -l)
      NGRP=$(awk 'NR>1' "$D/af3pack/groups.tsv" | wc -l)
      printf "  \033[36mINFO\033[0m %-34s %s group(s), %s bucket(s), %s distinct --mem\n" \
             "resource ladder" "$NGRP" "$NBUCK" "$NMEM"
      chk "packed groups built" "$NGRP" 1
    else
      printf "  \033[33mWARN\033[0m %-34s (legacy array route?)\n" "no af3pack/groups.tsv"
    fi
    # a metrics row must carry real numbers, not an empty iLIS column
    if [ -s "$D/metrics.tsv" ]; then
      "$PY" - "$D/metrics.tsv" <<'PYCHK'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
cols = rows[0].keys() if rows else []
def col(m):
    for c in (f"{m}_mean_mean", f"{m}_mean", m):
        if c in cols: return c
for m in ("iLIS", "PEAK", "ipTM"):
    c = col(m)
    if not c:
        print(f"  \033[31mFAIL\033[0m {m+' column':<34} absent"); sys.exit(1)
    vals = [r[c] for r in rows if r[c] not in ("", "nan", "NA")]
    if not vals:
        print(f"  \033[31mFAIL\033[0m {c:<34} all empty "
              "(scipy missing? see 'Reading the metrics' in the README)")
        sys.exit(1)
    print(f"  \033[32mPASS\033[0m {c:<34} {len(vals)}/{len(rows)} rows populated")
PYCHK
      [ $? -ne 0 ] && fail=1 || true
    fi
    # ---- CONTROLS: did it get the biology right, not just produce numbers? ----
    CTL="$REPO/examples/smoke_controls.tsv"
    if [ -s "$D/metrics.tsv" ] && [ -r "$CTL" ]; then
      "$PY" - "$D/metrics.tsv" "$CTL" <<'PYCTL'
import csv, sys, re, statistics as st
metrics, ctlfile = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(metrics), delimiter="\t"))
if not rows:
    sys.exit(0)
cols = rows[0].keys()
def col(m):
    for c in (f"{m}_mean_mean", f"{m}_mean", m):
        if c in cols:
            return c
ic = col("iLIS")
if not ic:
    sys.exit(0)

def norm(n):
    return re.sub(r"_\d{8}_\d{6}$", "", str(n)).lower()

score = {}
for r in rows:
    try:
        v = float(r[ic])
    except (TypeError, ValueError):
        continue
    score[norm(r.get("name", ""))] = max(v, score.get(norm(r.get("name", "")), -1))

ctl = [l.rstrip("\n").split("\t") for l in open(ctlfile)
       if l.strip() and not l.startswith("#")]
pos, neg, seen = [], [], 0
print("  -- controls (declared in examples/smoke_controls.tsv) --")
for a, b, expect, basis in ctl:
    key = f"{a}___{b}".lower()
    if key not in score:
        continue
    seen += 1
    v = score[key]
    (pos if expect == "positive" else neg).append((key, v))
    mark = "+" if expect == "positive" else "-"
    print(f"     {mark} {a+'x'+b:<18} iLIS {v:.3f}   {expect}")
if seen == 0:
    sys.exit(0)                      # not a smoke run; nothing to assert
if not pos or not neg:
    print("  \033[33mWARN\033[0m  controls incomplete -- cannot compare")
    sys.exit(0)

mp, mn = st.mean(v for _, v in pos), st.mean(v for _, v in neg)
ok = True
if mp > mn:
    print(f"  \033[32mPASS\033[0m {'positives outscore negatives':<34} "
          f"{mp:.3f} vs {mn:.3f}")
else:
    print(f"  \033[31mFAIL\033[0m {'positives outscore negatives':<34} "
          f"{mp:.3f} vs {mn:.3f}")
    ok = False
top = max(score.items(), key=lambda kv: kv[1])[0]
if top in dict(pos):
    print(f"  \033[32mPASS\033[0m {'top-ranked pair is a positive':<34} {top}")
else:
    print(f"  \033[31mFAIL\033[0m {'top-ranked pair is a positive':<34} "
          f"{top} ranked first")
    ok = False
for k, v in pos:
    if v <= 0.23:
        print(f"  \033[33mWARN\033[0m {k:<34} iLIS {v:.3f} below the 0.23 hit bar")
if not ok:
    print("  a controls failure is EITHER a pipeline bug OR AF3 missing a known")
    print("  complex -- open figures/pae/<pair>.png to tell them apart.")
sys.exit(0 if ok else 1)
PYCTL
      [ $? -ne 0 ] && fail=1 || true
    fi

    echo
    # `|| true` matters: under `set -e -o pipefail` a grep with no matches
    # aborts the script -- which is precisely the case on a CLEAN run, so
    # without this `check` fails every successful run before reporting.
    grep -hE "MISSING:" "$D"/logs/*.out 2>/dev/null | sed 's/^/  excluded: /' | head -5 || true
    grep -hE "EXIT=[^0]|ERROR" "$D"/logs/*.out 2>/dev/null | sed 's/^/  log: /' | head -5 || true
    echo
    if [ "$fail" -eq 0 ]; then
      printf "\033[32m[af3lis] all checks passed\033[0m -- the pipeline works end to end on this cluster\n"
    else
      printf "\033[31m[af3lis] checks FAILED\033[0m -- see above; 'af3lis.sh status %s' has more detail\n" "$NAME"
    fi
    exit "$fail" ;;

  help|-h|--help)
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
  *)
    die "unknown command '$CMD' (try: setup, template, run, smoke, check, status, report)" ;;
esac
