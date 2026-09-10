#!/bin/bash
# ============================================================
# AF3 Pipeline Orchestrator — Fully Automated SLURM Chaining
# ============================================================
#
# Submits the entire AF3 pipeline in one go:
#   Stage 1: MSA generation (CPU array job)
#   Stage 2: Inference JSON generation (short CPU job, auto-chained)
#   Stage 3: GPU inference with dynamic seeds (GPU array job, auto-chained)
#   Stage 4: Post-inference analysis (short CPU job, auto-chained)
#   Stage 5: Enriched dashboard + collaborator tarball  (OPTIONAL, off
#            unless ENRICH_SCRIPT is set in project.conf)
#
# Each stage waits for the previous one using SLURM --dependency=afterok.
# You run ONE command, then come back when everything is done.
#
# Usage:
#   bash af3_orchestrate.sh <project.conf>
#
# The project.conf file defines all project-specific paths, scripts,
# SLURM parameters, and commands. See project.conf.template for docs.
#
# Requirements:
#   - Run from a login node on bwHelix
#   - alphafold workspace must be active (or will be auto-allocated)
#   - All project scripts referenced in .conf must exist
#
# ============================================================

set -euo pipefail

# ============================================================
# 0. Parse arguments & load config
# ============================================================

if [ $# -lt 1 ]; then
    echo "Usage: $0 <project.conf> [--dry-run]"
    echo ""
    echo "  project.conf   Project configuration file (see project.conf.template)"
    echo "  --dry-run      Show what would be submitted without actually submitting"
    exit 1
fi

CONF_FILE="$1"
DRY_RUN=false
if [ "${2:-}" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE — nothing will be submitted ==="
    echo ""
fi

if [ ! -f "$CONF_FILE" ]; then
    echo "ERROR: Config file not found: $CONF_FILE" >&2
    exit 1
fi

# Source the config file
# shellcheck source=/dev/null
source "$CONF_FILE"

# Resolve paths relative to config file directory
CONF_DIR="$(cd "$(dirname "$CONF_FILE")" && pwd)"

# Find shared_scripts directory (where this orchestrator lives)
SHARED_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============================================================
# 1. Validate required config variables
# ============================================================

required_vars=(
    PROJECT_NAME
    MSA_GEN_CMD
    MSA_INPUT_DIR
    MSA_OUTPUT_DIR
    MSA_JOB_NAME
    MSA_PARTITION
    MSA_TIME
    MSA_MEM
    MSA_NTASKS
    INF_GEN_CMD
    INF_INPUT_DIR
    INF_OUTPUT_DIR
    INF_JOB_NAME
    INF_PARTITION
    INF_GPU
    INF_TIME
    INF_MEM
    INF_CPUS
    SEEDS
    ANALYSIS_DOWNLOAD_DIR
    ANALYSIS_REPORT_NAME
)

missing=()
for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        missing+=("$var")
    fi
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: Missing required config variables:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo ""
    echo "See project.conf.template for documentation." >&2
    exit 1
fi

# Optional variables with defaults
ANALYSIS_ROUND_NAME="${ANALYSIS_ROUND_NAME:-$PROJECT_NAME}"
ANALYSIS_EXTRA_ARGS="${ANALYSIS_EXTRA_ARGS:---full-pae}"
EXTRACT_SCRIPT="${EXTRACT_SCRIPT:-$SHARED_DIR/03b_extract_and_report.py}"
MSA_SLURM_SCRIPT="${MSA_SLURM_SCRIPT:-$SHARED_DIR/AF3_data_CPU_helix.sh}"
INF_SLURM_SCRIPT="${INF_SLURM_SCRIPT:-$SHARED_DIR/AF3_inference_H200_dynseeds.sh}"
SKIP_STAGE_1="${SKIP_STAGE_1:-false}"

# Stage 5 optional variables (OFF unless ENRICH_SCRIPT is set).
ENRICH_SCRIPT="${ENRICH_SCRIPT:-}"
ENRICH_EXTRA_ARGS="${ENRICH_EXTRA_ARGS:-}"
ENRICH_OUTPUT_NAME="${ENRICH_OUTPUT_NAME:-confidence_summary_enriched.tsv}"
DASHBOARD_SCRIPT="${DASHBOARD_SCRIPT:-}"
DASHBOARD_SUBTITLE="${DASHBOARD_SUBTITLE:-}"
DASHBOARD_OUTPUT_NAME="${DASHBOARD_OUTPUT_NAME:-dashboard.html}"
DASHBOARD_EXTRA_ARGS="${DASHBOARD_EXTRA_ARGS:-}"
PACKAGE_TARBALL="${PACKAGE_TARBALL:-false}"
TARBALL_NAME="${TARBALL_NAME:-${PROJECT_NAME}_for_download.tar.gz}"
STAGE5_JOB_NAME="${STAGE5_JOB_NAME:-${PROJECT_NAME}_stage5}"
STAGE5_PARTITION="${STAGE5_PARTITION:-cpu-single}"
STAGE5_TIME="${STAGE5_TIME:-01:00:00}"
STAGE5_MEM="${STAGE5_MEM:-30G}"
STAGE5_NTASKS="${STAGE5_NTASKS:-4}"
# Name of the TSV produced by Stage 4 (03b_extract_and_report.py) inside
# ANALYSIS_DOWNLOAD_DIR. This is the INPUT to the Stage 5 enrichment step.
STAGE4_TSV_NAME="${STAGE4_TSV_NAME:-confidence_summary.tsv}"

# Resolve Stage 5 script paths (relative → relative to config dir).
STAGE5_ENABLED="false"
if [ -n "$ENRICH_SCRIPT" ]; then
    STAGE5_ENABLED="true"
    case "$ENRICH_SCRIPT" in
        /*) ;;
        *) ENRICH_SCRIPT="$CONF_DIR/$ENRICH_SCRIPT" ;;
    esac
    if [ ! -f "$ENRICH_SCRIPT" ]; then
        echo "ERROR: ENRICH_SCRIPT not found: $ENRICH_SCRIPT" >&2
        exit 1
    fi
fi
if [ -n "$DASHBOARD_SCRIPT" ]; then
    case "$DASHBOARD_SCRIPT" in
        /*) ;;
        *) DASHBOARD_SCRIPT="$CONF_DIR/$DASHBOARD_SCRIPT" ;;
    esac
    if [ ! -f "$DASHBOARD_SCRIPT" ]; then
        echo "ERROR: DASHBOARD_SCRIPT not found: $DASHBOARD_SCRIPT" >&2
        exit 1
    fi
fi

# ============================================================
# 2. Workspace setup
# ============================================================

echo "============================================================"
echo " AF3 Pipeline Orchestrator: $PROJECT_NAME"
echo "============================================================"
echo ""
echo "Config file: $CONF_FILE"
echo "Project dir: $CONF_DIR"
echo "Shared dir:  $SHARED_DIR"
echo ""

# Allocate or find workspace
if ! $DRY_RUN; then
    ws_allocate alphafold 30 2>/dev/null || true
fi
WS_DIR="$(ws_find alphafold 2>/dev/null || echo '/tmp/WORKSPACE_PLACEHOLDER')"
echo "Workspace:   $WS_DIR"
echo ""

# Resolve output dirs (relative → inside workspace)
resolve_dir() {
    local dir="$1"
    case "$dir" in
        /*) echo "$dir" ;;
        *)  echo "$WS_DIR/$dir" ;;
    esac
}

FULL_MSA_OUTPUT="$(resolve_dir "$MSA_OUTPUT_DIR")"
FULL_INF_OUTPUT="$(resolve_dir "$INF_OUTPUT_DIR")"
FULL_DOWNLOAD="$(resolve_dir "$ANALYSIS_DOWNLOAD_DIR")"

# Resolve input dirs (relative → relative to config dir)
resolve_local_dir() {
    local dir="$1"
    case "$dir" in
        /*) echo "$dir" ;;
        *)  echo "$CONF_DIR/$dir" ;;
    esac
}

FULL_MSA_INPUT="$(resolve_local_dir "$MSA_INPUT_DIR")"
FULL_INF_INPUT="$(resolve_local_dir "$INF_INPUT_DIR")"

echo "MSA input:       $FULL_MSA_INPUT"
echo "MSA output:      $FULL_MSA_OUTPUT"
echo "Inf JSON input:  $FULL_INF_INPUT"
echo "Inf output:      $FULL_INF_OUTPUT"
echo "Download dir:    $FULL_DOWNLOAD"
if [ "$STAGE5_ENABLED" = "true" ]; then
    echo "Stage 5:         ENABLED"
    echo "  Enrich:        $ENRICH_SCRIPT"
    if [ -n "$DASHBOARD_SCRIPT" ]; then
        echo "  Dashboard:     $DASHBOARD_SCRIPT"
    fi
    if [ "$PACKAGE_TARBALL" = "true" ]; then
        echo "  Tarball:       $WS_DIR/$TARBALL_NAME"
    fi
else
    echo "Stage 5:         disabled (set ENRICH_SCRIPT to enable)"
fi
echo ""

# ============================================================
# 3. Stage 1: Generate MSA input JSONs & submit CPU array job
# ============================================================

if [ "$SKIP_STAGE_1" = "true" ]; then
    echo ">>> SKIPPING Stage 1 (MSA) — SKIP_STAGE_1=true"
    echo "    Assuming MSA outputs already exist at: $FULL_MSA_OUTPUT"
    echo ""
    MSA_JOBID="none"
else
    echo ">>> Stage 1: MSA generation"
    echo "    Running: $MSA_GEN_CMD"

    if ! $DRY_RUN; then
        # Run the MSA JSON generation script (local, fast)
        (cd "$CONF_DIR" && eval "$MSA_GEN_CMD")
    fi

    # Count input files
    NUM_MSA_FILES=$(find "$FULL_MSA_INPUT" -maxdepth 2 -type f -name "*.json" | wc -l)
    if [ "$NUM_MSA_FILES" -eq 0 ]; then
        echo "ERROR: No JSON files found in $FULL_MSA_INPUT after running MSA gen." >&2
        exit 1
    fi
    ARRAY_MAX=$((NUM_MSA_FILES - 1))

    echo "    Found $NUM_MSA_FILES MSA input files (array 0-${ARRAY_MAX})"
    echo "    Submitting MSA array job..."

    if $DRY_RUN; then
        echo "    [DRY RUN] sbatch --array=0-${ARRAY_MAX} --job-name=${MSA_JOB_NAME} ..."
        MSA_JOBID="DRY_MSA_123"
    else
        MSA_JOBID=$(sbatch \
            --array=0-${ARRAY_MAX} \
            --job-name="${MSA_JOB_NAME}" \
            --partition="${MSA_PARTITION}" \
            --time="${MSA_TIME}" \
            --mem="${MSA_MEM}" \
            --ntasks-per-node="${MSA_NTASKS}" \
            --output="slurm-${MSA_JOB_NAME}-%A_%a.out" \
            --parsable \
            "$MSA_SLURM_SCRIPT" "$FULL_MSA_INPUT" "$MSA_OUTPUT_DIR")
    fi

    echo "    MSA job submitted: $MSA_JOBID"
    echo ""
fi

# ============================================================
# 4. Stage 2+3+4: Submit the "bridge" job
# ============================================================
#
# The bridge job runs AFTER all MSA tasks complete.
# It does three things:
#   a) Runs the inference JSON generation script
#   b) Submits the inference array job
#   c) Submits the analysis job chained after inference
#
# We use a heredoc-generated SLURM script for maximum flexibility.
# ============================================================

BRIDGE_SCRIPT="$(mktemp "${CONF_DIR}/.af3_bridge_XXXXXX.sh")"

cat > "$BRIDGE_SCRIPT" <<'BRIDGE_EOF'
#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G

# ============================================================
# AF3 Pipeline Bridge Job
# Auto-generated by af3_orchestrate.sh
# Chains: inference JSON gen → inference submission → analysis
# ============================================================

set -euo pipefail

echo "Bridge job starting on $(hostname) at $(date)"

# --- Variables injected by orchestrator ---
CONF_DIR="__CONF_DIR__"
SHARED_DIR="__SHARED_DIR__"
WS_DIR="__WS_DIR__"
MSA_OUTPUT_DIR="__MSA_OUTPUT_DIR__"
FULL_MSA_OUTPUT="__FULL_MSA_OUTPUT__"
INF_GEN_CMD="__INF_GEN_CMD__"
FULL_INF_INPUT="__FULL_INF_INPUT__"
INF_OUTPUT_DIR="__INF_OUTPUT_DIR__"
FULL_INF_OUTPUT="__FULL_INF_OUTPUT__"
INF_SLURM_SCRIPT="__INF_SLURM_SCRIPT__"
INF_JOB_NAME="__INF_JOB_NAME__"
INF_PARTITION="__INF_PARTITION__"
INF_GPU="__INF_GPU__"
INF_TIME="__INF_TIME__"
INF_MEM="__INF_MEM__"
INF_CPUS="__INF_CPUS__"
SEEDS="__SEEDS__"
EXTRACT_SCRIPT="__EXTRACT_SCRIPT__"
FULL_INF_OUTPUT_ANALYSIS="__FULL_INF_OUTPUT__"
FULL_DOWNLOAD="__FULL_DOWNLOAD__"
ANALYSIS_REPORT_NAME="__ANALYSIS_REPORT_NAME__"
ANALYSIS_ROUND_NAME="__ANALYSIS_ROUND_NAME__"
ANALYSIS_EXTRA_ARGS="__ANALYSIS_EXTRA_ARGS__"
PROJECT_NAME="__PROJECT_NAME__"
STAGE5_ENABLED="__STAGE5_ENABLED__"
ENRICH_SCRIPT="__ENRICH_SCRIPT__"
# ENRICH_EXTRA_ARGS / DASHBOARD_EXTRA_ARGS are NOT stored as bash variables
# (their values may contain embedded quotes which would break the bash
# assignment syntax after sed substitution). They are baked directly into
# the Stage 5 sub-script command lines below as inline text.
ENRICH_OUTPUT_NAME="__ENRICH_OUTPUT_NAME__"
DASHBOARD_SCRIPT="__DASHBOARD_SCRIPT__"
DASHBOARD_SUBTITLE="__DASHBOARD_SUBTITLE__"
DASHBOARD_OUTPUT_NAME="__DASHBOARD_OUTPUT_NAME__"
PACKAGE_TARBALL="__PACKAGE_TARBALL__"
TARBALL_NAME="__TARBALL_NAME__"
STAGE5_JOB_NAME="__STAGE5_JOB_NAME__"
STAGE5_PARTITION="__STAGE5_PARTITION__"
STAGE5_TIME="__STAGE5_TIME__"
STAGE5_MEM="__STAGE5_MEM__"
STAGE5_NTASKS="__STAGE5_NTASKS__"
STAGE4_TSV_NAME="__STAGE4_TSV_NAME__"

# Load AF3 module (needed for environment)
module load bio/alphafold/3.0.1

# ============================================================
# Stage 2: Generate inference JSONs from MSA output
# ============================================================

echo ""
echo ">>> Stage 2: Generating inference JSONs"
echo "    MSA output:     $FULL_MSA_OUTPUT"
echo "    Inf JSON input: $FULL_INF_INPUT"

# Replace placeholders in the inference gen command
INF_CMD_RESOLVED="${INF_GEN_CMD}"
INF_CMD_RESOLVED="${INF_CMD_RESOLVED//\{MSA_OUTPUT_DIR\}/$FULL_MSA_OUTPUT}"
INF_CMD_RESOLVED="${INF_CMD_RESOLVED//\{INF_INPUT_DIR\}/$FULL_INF_INPUT}"
INF_CMD_RESOLVED="${INF_CMD_RESOLVED//\{WS_DIR\}/$WS_DIR}"

echo "    Running: $INF_CMD_RESOLVED"
(cd "$CONF_DIR" && eval "$INF_CMD_RESOLVED")

# Count _data.json files produced
NUM_INF_FILES=$(find "$FULL_INF_INPUT" -type f -name '*_data.json' | wc -l)
if [ "$NUM_INF_FILES" -eq 0 ]; then
    echo "ERROR: No _data.json files found in $FULL_INF_INPUT" >&2
    exit 1
fi

# Calculate total tasks: files × seeds
read -r -a SEED_ARRAY <<< "$SEEDS"
NSEEDS="${#SEED_ARRAY[@]}"
TOTAL_TASKS=$((NUM_INF_FILES * NSEEDS))
ARRAY_MAX=$((TOTAL_TASKS - 1))

echo "    Generated $NUM_INF_FILES inference JSONs × $NSEEDS seeds = $TOTAL_TASKS tasks"

# ============================================================
# Stage 3: Submit inference array job
# ============================================================

echo ""
echo ">>> Stage 3: Submitting inference array job (0-${ARRAY_MAX})"

INF_JOBID=$(sbatch \
    --array=0-${ARRAY_MAX} \
    --job-name="${INF_JOB_NAME}" \
    --partition="${INF_PARTITION}" \
    --gres="gpu:${INF_GPU}:1" \
    --time="${INF_TIME}" \
    --mem="${INF_MEM}" \
    --cpus-per-task="${INF_CPUS}" \
    --output="slurm-${INF_JOB_NAME}-%A_%a.out" \
    --parsable \
    "$INF_SLURM_SCRIPT" "$FULL_INF_INPUT" "$INF_OUTPUT_DIR" "$SEEDS")

echo "    Inference job submitted: $INF_JOBID"

# ============================================================
# Stage 4: Submit analysis job (chained after inference)
# ============================================================

echo ""
echo ">>> Stage 4: Submitting analysis job (after inference $INF_JOBID)"

# Write analysis script to a file (avoids --wrap quoting issues with special chars)
ANALYSIS_SCRIPT="${CONF_DIR}/.af3_analysis_${PROJECT_NAME}.sh"
cat > "$ANALYSIS_SCRIPT" <<ANALYSIS_EOF
#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
set -euo pipefail
module load bio/alphafold/3.0.1
echo "Analysis job starting at \$(date)"
python3 -c "import matplotlib" 2>/dev/null || pip install --user matplotlib
python3 "$EXTRACT_SCRIPT" \\
    "$FULL_INF_OUTPUT_ANALYSIS" \\
    --download-dir "$FULL_DOWNLOAD" \\
    --report-name "$ANALYSIS_REPORT_NAME" \\
    --round-name "$ANALYSIS_ROUND_NAME" \\
    $ANALYSIS_EXTRA_ARGS
echo "Analysis complete at \$(date)"
echo ""
echo "Results in: $FULL_DOWNLOAD"
# Self-cleanup (tolerate SLURM spool read-only edge cases)
rm -f "$ANALYSIS_SCRIPT" 2>/dev/null || true
ANALYSIS_EOF
chmod +x "$ANALYSIS_SCRIPT"

ANALYSIS_JOBID=$(sbatch \
    --dependency=afterok:${INF_JOBID} \
    --job-name="${PROJECT_NAME}_analysis" \
    --output="slurm-${PROJECT_NAME}_analysis-%j.out" \
    --parsable \
    "$ANALYSIS_SCRIPT")

echo "    Analysis job submitted: $ANALYSIS_JOBID"

# ============================================================
# Stage 5: Enriched dashboard + collaborator tarball
# ============================================================

STAGE5_JOBID=""
if [ "$STAGE5_ENABLED" = "true" ]; then
    echo ""
    echo ">>> Stage 5: Submitting enriched-dashboard job (after analysis $ANALYSIS_JOBID)"

    STAGE5_SCRIPT="${CONF_DIR}/.af3_stage5_${PROJECT_NAME}.sh"
    cat > "$STAGE5_SCRIPT" <<STAGE5_EOF
#!/bin/bash
#SBATCH --partition=${STAGE5_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${STAGE5_NTASKS}
#SBATCH --time=${STAGE5_TIME}
#SBATCH --mem=${STAGE5_MEM}
set -euo pipefail

echo "Stage 5 starting on \$(hostname) at \$(date)"

# Environment: generic Python + Biopython + numpy/matplotlib.
module load lang/Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
module load bio/Biopython/1.83-foss-2023a 2>/dev/null || true
python3 -c "import numpy"      2>/dev/null || pip install --user --quiet numpy
python3 -c "import matplotlib" 2>/dev/null || pip install --user --quiet matplotlib

STAGE4_TSV="$FULL_DOWNLOAD/$STAGE4_TSV_NAME"
ENRICHED_TSV="$FULL_DOWNLOAD/$ENRICH_OUTPUT_NAME"
DASHBOARD_HTML="$FULL_DOWNLOAD/$DASHBOARD_OUTPUT_NAME"

if [ ! -f "\$STAGE4_TSV" ]; then
    echo "ERROR: Stage 4 TSV not found: \$STAGE4_TSV" >&2
    echo "       (set STAGE4_TSV_NAME in project.conf if it has a non-default name)" >&2
    exit 1
fi

# --- Enrichment ---
echo ""
echo ">>> Stage 5a: Enrichment"
echo "    Script : $ENRICH_SCRIPT"
echo "    Input  : \$STAGE4_TSV"
echo "    Output : \$ENRICHED_TSV"
# EXTRA_ARGS are baked in as raw text (may contain quoted substrings like
# --title "foo bar"); the shell tokenizer handles them correctly here because
# they appear inline, not via an intermediate bash variable.
python3 "$ENRICH_SCRIPT" \\
    --tsv "\$STAGE4_TSV" \\
    --inference-dir "$FULL_INF_OUTPUT" \\
    --output "\$ENRICHED_TSV" \\
    __INLINE_ENRICH_EXTRA_ARGS__

# --- Dashboard (optional) ---
if [ -n "$DASHBOARD_SCRIPT" ]; then
    echo ""
    echo ">>> Stage 5b: Dashboard"
    echo "    Script : $DASHBOARD_SCRIPT"
    echo "    Output : \$DASHBOARD_HTML"
    python3 "$DASHBOARD_SCRIPT" \\
        --merged-tsv "\$ENRICHED_TSV" \\
        --inference-dir "$FULL_INF_OUTPUT" \\
        --output "\$DASHBOARD_HTML" \\
        --subtitle "$DASHBOARD_SUBTITLE" \\
        __INLINE_DASHBOARD_EXTRA_ARGS__
fi

# --- Tarball (optional) ---
if [ "$PACKAGE_TARBALL" = "true" ]; then
    TARBALL_PATH="$WS_DIR/$TARBALL_NAME"
    DL_PARENT="\$(dirname "$FULL_DOWNLOAD")"
    DL_BASE="\$(basename "$FULL_DOWNLOAD")"
    echo ""
    echo ">>> Stage 5c: Packaging collaborator tarball"
    echo "    Tarball: \$TARBALL_PATH"
    tar -czf "\$TARBALL_PATH" -C "\$DL_PARENT" "\$DL_BASE"
    du -h "\$TARBALL_PATH" | awk '{print "    Size   : " \$1}'
fi

echo ""
echo "Stage 5 complete at \$(date)"
echo ""
echo "Artifacts (in $FULL_DOWNLOAD):"
[ -f "\$ENRICHED_TSV" ]   && echo "  \$(basename "\$ENRICHED_TSV") ( \$(du -h "\$ENRICHED_TSV"   | cut -f1) )"
[ -f "\$DASHBOARD_HTML" ] && echo "  \$(basename "\$DASHBOARD_HTML") ( \$(du -h "\$DASHBOARD_HTML" | cut -f1) )"
if [ "$PACKAGE_TARBALL" = "true" ]; then
    echo "Collaborator tarball:"
    echo "  $WS_DIR/$TARBALL_NAME ( \$(du -h "$WS_DIR/$TARBALL_NAME" | cut -f1) )"
fi

# Self-cleanup (tolerate SLURM spool read-only edge cases)
rm -f "$STAGE5_SCRIPT" 2>/dev/null || true
STAGE5_EOF
    chmod +x "$STAGE5_SCRIPT"

    STAGE5_JOBID=$(sbatch \
        --dependency=afterok:${ANALYSIS_JOBID} \
        --job-name="${STAGE5_JOB_NAME}" \
        --output="slurm-${STAGE5_JOB_NAME}-%j.out" \
        --parsable \
        "$STAGE5_SCRIPT")
    echo "    Stage 5 job submitted: $STAGE5_JOBID"
else
    echo ""
    echo ">>> Stage 5 disabled (ENRICH_SCRIPT not set in project.conf — skipping)"
fi

# ============================================================
# Summary
# ============================================================

echo ""
echo "============================================================"
echo " Pipeline fully chained!"
echo "============================================================"
echo " Inference: $INF_JOBID (array 0-${ARRAY_MAX})"
echo " Analysis:  $ANALYSIS_JOBID (depends on $INF_JOBID)"
if [ -n "$STAGE5_JOBID" ]; then
    echo " Stage 5:   $STAGE5_JOBID (depends on $ANALYSIS_JOBID)"
fi
echo ""
echo " Monitor:   squeue -u \$USER"
echo "============================================================"

# Self-cleanup: remove the orchestrator-written copy of this bridge script.
# NOTE: Under SLURM, "$0" points to /var/spool/slurm/job*/slurm_script (read-only
# cached copy), not the script at BRIDGE_SCRIPT_SELF_PATH. Using "$0" causes a
# "Permission denied" rm which, combined with `set -euo pipefail`, marks the
# bridge FAILED 1:0 despite the pipeline having chained successfully.
rm -f "__BRIDGE_SCRIPT_SELF_PATH__" 2>/dev/null || true

BRIDGE_EOF

# Now replace placeholders in the bridge script with actual values
sed -i "s|__CONF_DIR__|${CONF_DIR}|g" "$BRIDGE_SCRIPT"
sed -i "s|__SHARED_DIR__|${SHARED_DIR}|g" "$BRIDGE_SCRIPT"
sed -i "s|__WS_DIR__|${WS_DIR}|g" "$BRIDGE_SCRIPT"
sed -i "s|__MSA_OUTPUT_DIR__|${MSA_OUTPUT_DIR}|g" "$BRIDGE_SCRIPT"
sed -i "s|__FULL_MSA_OUTPUT__|${FULL_MSA_OUTPUT}|g" "$BRIDGE_SCRIPT"
sed -i "s|__INF_GEN_CMD__|${INF_GEN_CMD}|g" "$BRIDGE_SCRIPT"
sed -i "s|__FULL_INF_INPUT__|${FULL_INF_INPUT}|g" "$BRIDGE_SCRIPT"
sed -i "s|__INF_OUTPUT_DIR__|${INF_OUTPUT_DIR}|g" "$BRIDGE_SCRIPT"
sed -i "s|__FULL_INF_OUTPUT__|${FULL_INF_OUTPUT}|g" "$BRIDGE_SCRIPT"
sed -i "s|__INF_SLURM_SCRIPT__|${INF_SLURM_SCRIPT}|g" "$BRIDGE_SCRIPT"
sed -i "s|__INF_JOB_NAME__|${INF_JOB_NAME}|g" "$BRIDGE_SCRIPT"
sed -i "s|__INF_PARTITION__|${INF_PARTITION}|g" "$BRIDGE_SCRIPT"
sed -i "s|__INF_GPU__|${INF_GPU}|g" "$BRIDGE_SCRIPT"
sed -i "s|__INF_TIME__|${INF_TIME}|g" "$BRIDGE_SCRIPT"
sed -i "s|__INF_MEM__|${INF_MEM}|g" "$BRIDGE_SCRIPT"
sed -i "s|__INF_CPUS__|${INF_CPUS}|g" "$BRIDGE_SCRIPT"
sed -i "s|__SEEDS__|${SEEDS}|g" "$BRIDGE_SCRIPT"
sed -i "s|__EXTRACT_SCRIPT__|${EXTRACT_SCRIPT}|g" "$BRIDGE_SCRIPT"
sed -i "s|__FULL_DOWNLOAD__|${FULL_DOWNLOAD}|g" "$BRIDGE_SCRIPT"
sed -i "s|__ANALYSIS_REPORT_NAME__|${ANALYSIS_REPORT_NAME}|g" "$BRIDGE_SCRIPT"
sed -i "s|__ANALYSIS_ROUND_NAME__|${ANALYSIS_ROUND_NAME}|g" "$BRIDGE_SCRIPT"
sed -i "s|__ANALYSIS_EXTRA_ARGS__|${ANALYSIS_EXTRA_ARGS}|g" "$BRIDGE_SCRIPT"
sed -i "s|__PROJECT_NAME__|${PROJECT_NAME}|g" "$BRIDGE_SCRIPT"
sed -i "s|__BRIDGE_SCRIPT_SELF_PATH__|${BRIDGE_SCRIPT}|g" "$BRIDGE_SCRIPT"

# Stage 5 placeholders (all empty if STAGE5_ENABLED=false)
sed -i "s|__STAGE5_ENABLED__|${STAGE5_ENABLED}|g"               "$BRIDGE_SCRIPT"
sed -i "s|__ENRICH_SCRIPT__|${ENRICH_SCRIPT}|g"                 "$BRIDGE_SCRIPT"
sed -i "s|__ENRICH_OUTPUT_NAME__|${ENRICH_OUTPUT_NAME}|g"       "$BRIDGE_SCRIPT"
sed -i "s|__DASHBOARD_SCRIPT__|${DASHBOARD_SCRIPT}|g"           "$BRIDGE_SCRIPT"
sed -i "s|__DASHBOARD_SUBTITLE__|${DASHBOARD_SUBTITLE}|g"       "$BRIDGE_SCRIPT"
sed -i "s|__DASHBOARD_OUTPUT_NAME__|${DASHBOARD_OUTPUT_NAME}|g" "$BRIDGE_SCRIPT"
sed -i "s|__PACKAGE_TARBALL__|${PACKAGE_TARBALL}|g"             "$BRIDGE_SCRIPT"
sed -i "s|__TARBALL_NAME__|${TARBALL_NAME}|g"                   "$BRIDGE_SCRIPT"
sed -i "s|__STAGE5_JOB_NAME__|${STAGE5_JOB_NAME}|g"             "$BRIDGE_SCRIPT"
sed -i "s|__STAGE5_PARTITION__|${STAGE5_PARTITION}|g"           "$BRIDGE_SCRIPT"
sed -i "s|__STAGE5_TIME__|${STAGE5_TIME}|g"                     "$BRIDGE_SCRIPT"
sed -i "s|__STAGE5_MEM__|${STAGE5_MEM}|g"                       "$BRIDGE_SCRIPT"
sed -i "s|__STAGE5_NTASKS__|${STAGE5_NTASKS}|g"                 "$BRIDGE_SCRIPT"
sed -i "s|__STAGE4_TSV_NAME__|${STAGE4_TSV_NAME}|g"             "$BRIDGE_SCRIPT"

# EXTRA_ARGS inlined directly into the Stage-5 command lines. Values may
# contain embedded double quotes (e.g. --title "foo bar"), so we must not
# route them through an intermediate bash variable or through sed (sed
# would require escaping of |, &, \ in the value). Using python with env
# vars is the cleanest safe substitution.
ENRICH_EXTRA_ARGS="$ENRICH_EXTRA_ARGS" \
DASHBOARD_EXTRA_ARGS="$DASHBOARD_EXTRA_ARGS" \
python3 - "$BRIDGE_SCRIPT" <<'PYREPLACE'
import sys, os
path = sys.argv[1]
enrich = os.environ.get("ENRICH_EXTRA_ARGS", "")
dash   = os.environ.get("DASHBOARD_EXTRA_ARGS", "")
with open(path) as f:
    text = f.read()
text = text.replace("__INLINE_ENRICH_EXTRA_ARGS__",    enrich)
text = text.replace("__INLINE_DASHBOARD_EXTRA_ARGS__", dash)
with open(path, "w") as f:
    f.write(text)
PYREPLACE

echo ">>> Stages 2-4: Bridge job (chains inference JSON gen → inference → analysis)"

if [ "$MSA_JOBID" = "none" ]; then
    # MSA was skipped, submit bridge immediately
    DEPENDENCY_FLAG=""
    echo "    No MSA dependency (stage 1 skipped)"
else
    DEPENDENCY_FLAG="--dependency=afterok:${MSA_JOBID}"
    echo "    Will start after MSA job $MSA_JOBID completes"
fi

if $DRY_RUN; then
    echo "    [DRY RUN] sbatch $DEPENDENCY_FLAG --job-name=${PROJECT_NAME}_bridge $BRIDGE_SCRIPT"
    BRIDGE_JOBID="DRY_BRIDGE_456"
    echo ""
    echo "    Bridge script written to: $BRIDGE_SCRIPT"
    echo "    (In dry-run mode, this file is kept for inspection)"
else
    BRIDGE_JOBID=$(sbatch \
        $DEPENDENCY_FLAG \
        --job-name="${PROJECT_NAME}_bridge" \
        --output="slurm-${PROJECT_NAME}_bridge-%j.out" \
        --parsable \
        "$BRIDGE_SCRIPT")
    # Bridge script self-cleans when it runs; keep it until then
fi

echo "    Bridge job submitted: $BRIDGE_JOBID"

# ============================================================
# Final Summary
# ============================================================

echo ""
echo "============================================================"
echo " AF3 Pipeline Submitted: $PROJECT_NAME"
echo "============================================================"
echo ""
if [ "$MSA_JOBID" != "none" ]; then
    echo " Stage 1 (MSA):             Job $MSA_JOBID"
fi
echo " Stage 2-4 (Bridge):        Job $BRIDGE_JOBID"
echo "   └─ Stage 2: Inf JSON gen  (runs inside bridge)"
echo "   └─ Stage 3: GPU inference (submitted by bridge)"
echo "   └─ Stage 4: Analysis      (submitted by bridge)"
if [ "$STAGE5_ENABLED" = "true" ]; then
    echo "   └─ Stage 5: Enriched dashboard + tarball (submitted by bridge)"
fi
echo ""
echo " Monitor all jobs:  squeue -u $USER"
echo " Cancel pipeline:   scancel $MSA_JOBID $BRIDGE_JOBID"
echo ""
echo " When complete, results will be in:"
echo "   $FULL_DOWNLOAD"
if [ "$STAGE5_ENABLED" = "true" ] && [ "$PACKAGE_TARBALL" = "true" ]; then
    echo ""
    echo " Collaborator tarball will be at:"
    echo "   $WS_DIR/$TARBALL_NAME"
fi
echo ""
echo "============================================================"
