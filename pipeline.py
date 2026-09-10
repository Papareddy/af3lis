#!/usr/bin/env python3
"""af3lis main pipeline -- AlphaFold-3 mirror of boltzlis.

Give chain A and chain B (each one or many protein IDs), get AF3 input JSONs +
A ready-to-submit SLURM pipeline: CPU align array -> PACKED GPU inference
via an afterany bridge job (the DEFAULT since packing was adopted from Mau's
bwHelix AF3 toolkit) + the post-run metric-collect command. `--array-infer`
restores the legacy per-prediction infer array chained on afterok.

The output TSV schema is byte-identical to boltzlis in `--agg flat` mode
(iLIS/LIS/cLIS/iLIA/LIA/cLIA/actifpTM/ipSAE/PEAK/ipTM/pTM/pLDDT_i/pLDDT_j +
n_models) so the downstream beeswarm plotter and PEAK-ranking convention
(PEAK >= 0.7 AND iLIS >= 0.22 = confident hit) are interchangeable.

Examples
--------
# screen: every A x every B as 2-chain folds
python pipeline.py --name UFM_screen \
    --chainA UFL1,UFC1 --chainB DDRGK1,CDK5RAP3 \
    --fasta examples/ufm_machinery.fasta --outdir runs/UFM_screen

# single multi-chain complex (all A + all B chains in one structure)
python pipeline.py --name UFM_complex --mode complex \
    --chainA UFM1,UBA5 --chainB UFC1 --fasta examples/ufm_machinery.fasta \
    --outdir runs/UFM_complex

# multi-seed (seeds baked into JSON in stage 1 -- changing requires rebuild)
python pipeline.py --name UFM_screen --chainA UFL1 --chainB DDRGK1 \
    --seeds 1,2,3 --num-samples 5 --outdir runs/UFM_screen

# submit the two-stage array (CPU MSA -> GPU infer chained on afterok)
python pipeline.py --submit --outdir runs/UFM_screen

# PACKED inference (recommended for big screens on low fairshare): after the
# align array completes, group predictions into few fat single-bucket GPU jobs
# (strategy after Mau's bwHelix AF3 toolkit; see af3lis/pack.py)
python pipeline.py --pack --outdir runs/UFM_screen        # -> af3pack/submit_all.sh
python pipeline.py --pack --probe 10 --outdir runs/UFM_screen   # calibration probe

# just (re)collect metrics for a finished AF3 out dir (array route: out/,
# packed route: out_pack/). Emits our metric set + the Dunbrack ipsae.py
# family (pDockQ/pDockQ2/ipSAE_d0*) unless --no-ipsae.
python pipeline.py --collect runs/UFM_screen/out -o runs/UFM_screen/metrics.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from af3lis import fetch, json_build
from af3lis.af3_io import assert_data_json_exists
from af3lis.collect import collect_all

HERE = os.path.dirname(os.path.abspath(__file__))
TMPL_ALIGN = os.path.join(HERE, "slurm", "af3_align.sbatch.tmpl")
TMPL_INFER = os.path.join(HERE, "slurm", "af3_infer.sbatch.tmpl")
TMPL_PACK = os.path.join(HERE, "slurm", "af3_pack_infer.sbatch.tmpl")
TMPL_BRIDGE = os.path.join(HERE, "slurm", "af3_pack_bridge.sbatch.tmpl")
TMPL_ANALYSE = os.path.join(HERE, "slurm", "af3_analyse.sbatch.tmpl")

# Sentinel for argparse defaults — distinguishes "user didn't pass --seeds"
# from "user passed --seeds 1". Lets us source seeds from config.yaml when
# unset rather than silently overriding config with the CLI default.
_UNSET = object()

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

_DEFAULT_CFG: dict[str, Any] = {
    # AF3 module + weights/db
    "af3_module": "bio/alphafold/3.0.1",
    "model_dir": "/path/to/af3-models",
    "db_dir": "${ALPHAFOLD_DATABASES}",
    # analysis python (numpy/scipy/pandas) for lis.py + collect
    "python": "python3",
    "workspace": "/path/to/scratch/af3_screen",
    "seq_cache": "seq_cache",
    # align (CPU)
    "align_partition": "cpu-single",
    "align_time": "12:00:00",
    "align_mem": "64gb",
    "align_cpus": "16",
    "af3_data_extra": "",
    # infer (GPU)
    "infer_partition": "gpu-single",
    "gpu_gres": "gpu:A100:1",
    "infer_time": "02:00:00",
    "infer_mem": "60gb",
    "infer_cpus": "8",
    "xla_mem_frac": "3.2",
    "flash_attn": "triton",
    "num_samples": "5",
    "num_recycles": "10",
    "af3_infer_extra": "",
}

# nested keys that the YAML-ish config supports as either flat or nested
_NESTED_MAP = {
    "align": {"partition": "align_partition", "time": "align_time",
              "mem": "align_mem", "cpus": "align_cpus",
              "af3_data_extra": "af3_data_extra"},
    "infer": {"partition": "infer_partition", "gpu_gres": "gpu_gres",
              "time": "infer_time", "mem": "infer_mem", "cpus": "infer_cpus",
              "xla_mem_frac": "xla_mem_frac", "flash_attn": "flash_attn",
              "num_samples": "num_samples", "num_recycles": "num_recycles",
              "af3_infer_extra": "af3_infer_extra"},
}


def _strip_quotes(v: str) -> str:
    return v.strip().strip('"').strip("'")


def load_config(path: str | None) -> dict[str, Any]:
    """Tolerant YAML-ish parser (same idea as boltzlis.load_config) supporting
    flat `key: value` AND one level of `section:` / two-space-indented children
    so the on-cluster config.example.yaml shape Just Works.
    Returns a flat dict keyed by _DEFAULT_CFG names.
    """
    cfg: dict[str, Any] = dict(_DEFAULT_CFG)
    if not (path and os.path.exists(path)):
        return cfg
    current_section: str | None = None
    with open(path) as fh:
        for raw in fh:
            ln = raw.split("#", 1)[0].rstrip("\n")
            if not ln.strip():
                continue
            indented = ln.startswith(" ") or ln.startswith("\t")
            stripped = ln.strip()
            if ":" not in stripped:
                continue
            k, _, v = stripped.partition(":")
            k = k.strip()
            v = _strip_quotes(v)
            if not indented:
                # top-level key
                current_section = None
                if v == "" and k in _NESTED_MAP:
                    current_section = k
                    continue
                # seeds: [1,2,3]   or   seeds: 1
                if k == "seeds":
                    cfg["seeds"] = _parse_seeds_value(v)
                else:
                    cfg[k] = v
            else:
                if current_section and current_section in _NESTED_MAP:
                    flat = _NESTED_MAP[current_section].get(k)
                    if flat:
                        cfg[flat] = v
                else:
                    cfg[k] = v
    return cfg


def _parse_seeds_value(v: str) -> list[int]:
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        v = v[1:-1]
    return [int(x) for x in v.replace(",", " ").split() if x.strip()]


# ----------------------------------------------------------------------------
# sbatch rendering
# ----------------------------------------------------------------------------

def _render_template(tmpl_path: str, mapping: dict[str, Any]) -> str:
    with open(tmpl_path) as fh:
        tmpl = fh.read()
    # str.format would barf on bash ${VAR} -- but the templates already escape
    # bash interpolation as ${{...}}, so plain .format works.
    return tmpl.format(**mapping)


def render_sbatches(jobs: list[tuple[str, str]],
                    outdir: str,
                    name: str,
                    cfg: dict[str, Any]) -> tuple[str, str, str]:
    """Render BOTH align and infer sbatch files.

    Returns (align_sbatch_path, infer_sbatch_path, listfile_path).
    The list file holds one absolute JSON path per line and drives BOTH stages.
    """
    listfile = os.path.join(outdir, "af3_json_list.txt")
    with open(listfile, "w") as fh:
        fh.write("\n".join(os.path.abspath(p) for _, p in jobs) + "\n")

    abs_out = os.path.abspath(outdir)
    common = dict(
        JOBNAME=name[:14],
        NJOBS=len(jobs),
        OUTDIR=abs_out,
        LISTFILE=os.path.abspath(listfile),
        AF3_MODULE=cfg["af3_module"],
        DB_DIR=cfg["db_dir"],
        MODEL_DIR=cfg["model_dir"],
    )
    align_map = dict(common,
                     ALIGN_PARTITION=cfg["align_partition"],
                     ALIGN_TIME=cfg["align_time"],
                     ALIGN_MEM=cfg["align_mem"],
                     ALIGN_CPUS=cfg["align_cpus"],
                     AF3_DATA_EXTRA=cfg.get("af3_data_extra", ""))
    infer_map = dict(common,
                     INFER_PARTITION=cfg["infer_partition"],
                     GPU_GRES=cfg["gpu_gres"],
                     INFER_TIME=cfg["infer_time"],
                     INFER_MEM=cfg["infer_mem"],
                     INFER_CPUS=cfg["infer_cpus"],
                     XLA_MEM_FRAC=cfg["xla_mem_frac"],
                     FLASH_ATTN=cfg["flash_attn"],
                     NUM_SAMPLES=cfg["num_samples"],
                     NUM_RECYCLES=cfg["num_recycles"],
                     AF3_INFER_EXTRA=cfg.get("af3_infer_extra", ""))

    align_sb = os.path.join(outdir, "align.sbatch")
    infer_sb = os.path.join(outdir, "infer.sbatch")
    pack_sb = os.path.join(outdir, "pack_infer.sbatch")
    bridge_sb = os.path.join(outdir, "pack_bridge.sbatch")
    analyse_sb = os.path.join(outdir, "analyse.sbatch")
    for tmpl in (TMPL_ALIGN, TMPL_INFER, TMPL_PACK, TMPL_BRIDGE, TMPL_ANALYSE):
        if not os.path.exists(tmpl):
            raise FileNotFoundError(f"sbatch template missing: {tmpl}")
    rendered_align = _render_template(TMPL_ALIGN, align_map)
    rendered_infer = _render_template(TMPL_INFER, infer_map)
    # Packed template takes GROUP_DIR/OUT_ROOT as $1/$2 at submit time (from
    # af3pack/submit_all.sh) — it shares the infer map (env, module, flags).
    rendered_pack = _render_template(TMPL_PACK, infer_map)
    # Bridge runs on the CPU partition (it only groups + sbatches), so it
    # takes the align map plus the pack sizing knob.
    bridge_map = dict(align_map,
                      PIPELINE_DIR=os.path.abspath(HERE),
                      TARGET_HOURS=cfg.get("target_hours", 1.5))
    rendered_bridge = _render_template(TMPL_BRIDGE, bridge_map)
    # Final scoring/plot/PDB stage. CPU partition; needs the af3lis conda env,
    # so it resolves AF3LIS_PYTHON via env/activate.sh at run time.
    analyse_map = dict(align_map,
                       PIPELINE_DIR=os.path.abspath(HERE),
                       ANALYSE_TIME=cfg.get("analyse_time", "02:00:00"),
                       ANALYSE_MEM=cfg.get("analyse_mem", "32gb"),
                       ANALYSE_CPUS=cfg.get("analyse_cpus", 8),
                       AGG_MODE=cfg.get("agg", "per_seed"),
                       OUT_SUBDIR=cfg.get("out_subdir", "out_pack"),
                       PEAK_BAR=cfg.get("peak_bar", 0.70),
                       ILIS_BAR=cfg.get("ilis_bar", 0.23),
                       COLLECT_EXTRA=cfg.get("collect_extra", ""),
                       PDB_EXTRA=cfg.get("pdb_extra", ""))
    rendered_analyse = _render_template(TMPL_ANALYSE, analyse_map)
    # Renderer sanity check: no unsubstituted `{PLACEHOLDER}` should remain.
    # Catches typos in template keys before they hit SLURM. Bash variables
    # like ${VAR} are NOT placeholders — we only flag standalone `{NAME}`
    # NOT preceded by `$`.
    import re as _re
    for tag, text in (("align", rendered_align), ("infer", rendered_infer),
                      ("pack", rendered_pack), ("bridge", rendered_bridge),
                      ("analyse", rendered_analyse)):
        leftover = _re.findall(r"(?<![\$])\{[A-Z_]+\}", text)
        if leftover:
            raise ValueError(
                f"{tag} sbatch has unsubstituted placeholders: {leftover[:5]}"
            )
    with open(align_sb, "w") as fh:
        fh.write(rendered_align)
    with open(infer_sb, "w") as fh:
        fh.write(rendered_infer)
    with open(pack_sb, "w") as fh:
        fh.write(rendered_pack)
    with open(bridge_sb, "w") as fh:
        fh.write(rendered_bridge)
    with open(analyse_sb, "w") as fh:
        fh.write(rendered_analyse)
    return align_sb, infer_sb, listfile


def write_runbook(outdir: str,
                  name: str,
                  jobs: list[tuple[str, str]],
                  mode: str,
                  seeds: list[int],
                  num_samples: int,
                  align_sb: str,
                  infer_sb: str,
                  cfg_path: str | None) -> str:
    runbook = os.path.join(outdir, "RUNBOOK.md")
    abs_out = os.path.abspath(outdir)
    cfg_arg = ("--config %s" % cfg_path) if cfg_path else "--config <config.yaml>"
    body = (
        "# %s -- AF3 run\n\n"
        "**%d fold(s)**, mode=`%s`, seeds=`%s`, diffusion_samples=`%d`, "
        "total samples/pair = `%d`.\n\n"
        "## Two-stage SLURM (CPU MSA -> PACKED GPU inference)\n"
        "```bash\n"
        "# 1. sync this run dir to the cluster workspace.\n"
        "\n"
        "# 2. stage 1: CPU MSA / templates (the long step).\n"
        "ALIGN_ID=$(sbatch --parsable --array=1-%d %s)\n"
        "echo \"align job=$ALIGN_ID\"\n"
        "\n"
        "# 3. stage 2: packed GPU inference via the bridge job.\n"
        "#    afterany, not afterok: one failed align task under afterok\n"
        "#    strands the dependent job in DependencyNeverSatisfied.\n"
        "#    The bridge packs whatever MSAs landed and reports the rest.\n"
        "#    Legacy per-task array instead: add --array-infer to --submit,\n"
        "#    or sbatch --dependency=afterok:$ALIGN_ID --array=1-%d %s\n"
        "INFER_ID=$(sbatch --parsable --dependency=afterany:$ALIGN_ID "
        "%s/pack_bridge.sbatch)\n"
        "echo \"pack-bridge job=$INFER_ID\"\n"
        "\n"
        "# 4. when the packed groups finish, collect ALL metrics:\n"
        "python %s/pipeline.py --collect %s/out_pack -o %s/metrics.tsv %s\n"
        "```\n\n"
        "## Or driven via this script (does the chaining for you)\n"
        "```bash\n"
        "python %s/pipeline.py --submit --outdir %s %s\n"
        "# add --dry-run to see the exact sbatch commands without launching\n"
        "# add --align-only / --infer-only to run a single stage\n"
        "```\n\n"
        "## PACKED inference -- THE DEFAULT (run manually here if you want control)\n"
        "`--submit` already chains this via pack_bridge.sbatch; the commands\n"
        "below are the same thing done by hand. Replaces stage 2 only (align stays an array). Packs many predictions\n"
        "into few fat jobs via AF3 `--input_dir`: weights load once, XLA\n"
        "compiles once per token bucket, and the jobs sit inside bwHelix's\n"
        "~100-job age-accrue window instead of forfeiting age as array tasks.\n"
        "(Strategy after Mau's bwHelix AF3 toolkit.)\n"
        "```bash\n"
        "# after the ALIGN array completes, ON THE CLUSTER:\n"
        "# (a) new size class or GPU? probe + calibrate first:\n"
        "python %s/pipeline.py --pack --probe 10 --outdir %s %s\n"
        "bash %s/af3pack_probe/submit_all.sh\n"
        "#     ...when it finishes:\n"
        "python -m af3lis.calibrate %s/logs/pack_*.out --out %s/calib.json\n"
        "# (b) size + submit the campaign (uses calib.json if present):\n"
        "python %s/pipeline.py --pack --outdir %s %s\n"
        "bash %s/af3pack/submit_all.sh\n"
        "# models land in out_pack/ -> collect with:\n"
        "python %s/pipeline.py --collect %s/out_pack -o %s/metrics.tsv %s\n"
        "```\n\n"
        "## Rerunning inference only (e.g. flash-attn miscompile)\n"
        "```bash\n"
        "python %s/pipeline.py --submit --infer-only --outdir %s %s\n"
        "```\n\n"
        "## Notes\n"
        "- `modelSeeds` is baked into the input JSON in stage 1; "
        "changing `--seeds` requires `--rebuild` (re-emit JSONs and re-align).\n"
        "- `<lname>_data.json` (per-pair, in out/<lname>/) is the "
        "reproducibility artifact. Do NOT `rm -rf` per-pair dirs after collect.\n"
        "- Cold-start JAX compile is ~5-10 min per node -- bake into wallclock.\n"
        % (name, len(jobs), mode, ",".join(str(s) for s in seeds),
           num_samples, len(seeds) * num_samples,
           len(jobs), os.path.basename(align_sb),
           len(jobs), os.path.basename(infer_sb), abs_out,
           HERE, abs_out, abs_out, cfg_arg,
           HERE, abs_out, cfg_arg,
           # packed-inference section
           HERE, abs_out, cfg_arg,          # --pack --probe
           abs_out,                          # bash af3pack_probe/submit_all.sh
           abs_out, abs_out,                 # calibrate log -> calib.json
           HERE, abs_out, cfg_arg,           # --pack
           abs_out,                          # bash af3pack/submit_all.sh
           HERE, abs_out, abs_out, cfg_arg,  # collect out_pack
           HERE, abs_out, cfg_arg)
    )
    with open(runbook, "w") as fh:
        fh.write(body)
    return runbook


# ----------------------------------------------------------------------------
# submit driver
# ----------------------------------------------------------------------------

def _read_list(listfile: str) -> list[str]:
    with open(listfile) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _lname_of(json_path: str) -> str:
    """AF3 lowercases the JSON `name` field into the output subdir."""
    with open(json_path) as fh:
        data = json.load(fh)
    return str(data.get("name", os.path.splitext(os.path.basename(json_path))[0])).lower()


def _read_seeds_from_first_json(listfile: str) -> list[int]:
    """Return modelSeeds from the first JSON in the list file (authoritative)."""
    paths = _read_list(listfile)
    if not paths:
        raise SystemExit(f"{listfile} is empty — nothing to read seeds from")
    with open(paths[0]) as fh:
        d = json.load(fh)
    return [int(x) for x in d.get("modelSeeds", [])]


def _check_seeds_baked(listfile: str, cfg_seeds: list[int]) -> None:
    """Each JSON's modelSeeds must equal cfg_seeds (list compare; rejects dupes).
    If not, the user changed --seeds in config without rebuilding -- abort.
    """
    want = sorted(cfg_seeds)
    if len(set(cfg_seeds)) != len(cfg_seeds):
        raise SystemExit(
            "config seeds=%s contains duplicates — AF3 treats them as distinct, "
            "rejecting." % cfg_seeds
        )
    for jp in _read_list(listfile):
        with open(jp) as fh:
            d = json.load(fh)
        got = sorted(int(x) for x in d.get("modelSeeds", []))
        if got != want:
            raise SystemExit(
                "seeds mismatch in %s: JSON has modelSeeds=%s but config "
                "seeds=%s. Remove the outdir and rebuild to re-emit JSONs."
                % (jp, got, want))


def cmd_submit(outdir: str,
               cfg: dict[str, Any],
               align_only: bool = False,
               infer_only: bool = False,
               no_dependency: bool = False,
               dry_run: bool = False,
               time_align: str | None = None,
               time_infer: str | None = None,
               array_infer: bool = False,
               analyse: bool = True) -> tuple[int | None, int | None]:
    """Submit the rendered two-stage pipeline. Returns (align_id, infer_id).

    Inference defaults to the PACKED route (align array -> bridge job ->
    few fat single-bucket GPU jobs; strategy after Mau's bwHelix AF3
    toolkit), because a per-prediction array is the wrong shape on a
    low-fairshare account: array tasks forfeit accrued age and every task
    re-pays weights-load + XLA compile. Pass array_infer=True for the
    legacy one-task-per-prediction array.
    """
    listfile = os.path.join(outdir, "af3_json_list.txt")
    align_sb = os.path.join(outdir, "align.sbatch")
    infer_sb = os.path.join(outdir, "infer.sbatch")
    if not os.path.exists(listfile):
        raise SystemExit("missing %s -- run build first" % listfile)
    paths = _read_list(listfile)
    if not paths:
        raise SystemExit("%s is empty -- nothing to submit" % listfile)
    n = len(paths)

    # --- array-size guard -------------------------------------------------
    # SLURM refuses --array=1-N above MaxArraySize (1001 on bwHelix) and the
    # rejection is easy to miss: the align array never runs, the afterany
    # bridge fires anyway, finds no MSAs, and the failure surfaces two stages
    # later as "no models". Fail here instead, with the arithmetic shown.
    #
    # NB in af3lis the array is sized by CONDITIONS, not conditions x seeds --
    # seeds are baked into each input JSON, so more seeds make each task longer
    # rather than adding tasks.
    max_array = int(cfg.get("max_array_size", 1001))
    if n > max_array:
        raise SystemExit(
            f"{n} conditions exceeds this site's MaxArraySize ({max_array}), so "
            f"`sbatch --array=1-{n}` would be REJECTED.\n"
            f"  Split the screen across run dirs, e.g. two builds of "
            f"~{n // 2 + 1} pairs each with separate --outdir, or raise\n"
            f"  max_array_size in config.yaml if your site allows more "
            f"(check: scontrol show config | grep MaxArraySize)."
        )

    # seeds baked into JSON in stage 1; abort early if user edited cfg without rebuilding
    if "seeds" in cfg:
        _check_seeds_baked(listfile, cfg["seeds"])

    # --infer-only precondition: every <lname>_data.json must already exist.
    # Skip the check entirely when out/ doesn't exist locally (e.g. user is
    # invoking on their laptop pre-rsync; AF3 will check on the cluster).
    # Override via env var AF3LIS_SKIP_DATA_CHECK=1 for advanced workflows.
    if infer_only:
        out_root = os.path.join(outdir, "out")
        if os.path.isdir(out_root) and not os.environ.get(
                "AF3LIS_SKIP_DATA_CHECK"):
            jobs_with_data = list(_read_list(listfile))
            if jobs_with_data:
                for jp in paths:
                    jd = os.path.join(out_root, _lname_of(jp))
                    if os.path.isdir(jd):
                        # Only assert when the job dir exists; otherwise defer
                        # to the cluster-side template check.
                        assert_data_json_exists(jd)
        else:
            sys.stderr.write(
                "[submit] WARNING: --infer-only precondition check skipped "
                f"(out/ missing under {outdir} — assuming you've rsynced "
                "this build to a cluster where MSAs already exist).\n"
            )

    # commands
    def _sbatch(args: list[str]) -> int | None:
        if dry_run:
            print(" ".join(shlex.quote(x) for x in args))
            return None
        out = subprocess.check_output(args, text=True).strip()
        return int(out)

    align_id: int | None = None
    infer_id: int | None = None
    # In dry-run, fake an align id so the rendered infer line still carries
    # the --dependency=afterok:<id> flag — without this, a user inspecting
    # SUBMIT_CMDS.sh sees two unchained sbatches and (wrongly) assumes the
    # chain isn't set up.
    dry_align_token = "$ALIGN_ID"
    cmds_log: list[str] = []

    if not infer_only and os.path.exists(align_sb):
        a_args = ["sbatch", "--parsable", "--array=1-%d" % n]
        if time_align:
            a_args += ["--time=" + time_align]
        a_args += [align_sb]
        if dry_run:
            # Render as a shell capture so the next step's $ALIGN_ID expands.
            cmds_log.append(
                "ALIGN_ID=$(%s)" % " ".join(shlex.quote(x) for x in a_args)
            )
        else:
            cmds_log.append(" ".join(shlex.quote(x) for x in a_args))
        align_id = _sbatch(a_args)
        if align_id is not None:
            print("align job=%d" % align_id)

    bridge_sb = os.path.join(outdir, "pack_bridge.sbatch")
    if not align_only and not array_infer and os.path.exists(bridge_sb):
        # PACKED route (default). afterany, not afterok: a single failed align
        # task under afterok would strand this job in DependencyNeverSatisfied
        # forever. af3lis.pack excludes and reports conditions whose
        # *_data.json never landed, so a partial align yields partial results
        # plus an explicit missing list.
        b_args = ["sbatch", "--parsable"]
        if (not no_dependency) and align_id is not None:
            b_args += ["--dependency=afterany:%d" % align_id]
        b_args += [bridge_sb]
        quoted = " ".join(shlex.quote(x) for x in b_args)
        if (dry_run and not no_dependency and align_id is None
                and not infer_only):
            parts = quoted.split(" ", 2)  # sbatch --parsable rest
            quoted = " ".join(
                parts[:2] + ["--dependency=afterany:%s" % dry_align_token]
                + parts[2:])
        # capture the id so a sourced SUBMIT_CMDS.sh can chain the
        # analyse job onto it ($INFER_ID would otherwise be unbound)
        cmds_log.append("INFER_ID=$(%s)" % quoted if dry_run else quoted)
        if dry_run:
            print(quoted)
        else:
            infer_id = _sbatch(b_args)
            if infer_id is not None:
                print("pack-bridge job=%d (submits the packed GPU groups "
                      "when align finishes)" % infer_id)
    elif not align_only and os.path.exists(infer_sb):
        i_args = ["sbatch", "--parsable", "--array=1-%d" % n]
        if (not no_dependency) and align_id is not None:
            i_args += ["--dependency=afterok:%d" % align_id]
        if time_infer:
            i_args += ["--time=" + time_infer]
        i_args += [infer_sb]
        # For dry-run with a freshly-built chain, splice in the
        # `--dependency=afterok:$ALIGN_ID` placeholder UNQUOTED so the shell
        # expands $ALIGN_ID when the user sources SUBMIT_CMDS.sh.
        quoted = " ".join(shlex.quote(x) for x in i_args)
        if (dry_run and not no_dependency and align_id is None
                and not infer_only):
            parts = quoted.split(" ", 3)  # sbatch --parsable --array=N rest
            dep = "--dependency=afterok:%s" % dry_align_token
            quoted = " ".join(parts[:3] + [dep] + parts[3:])
        # capture the id so the analyse line below can chain onto it
        cmds_log.append("INFER_ID=$(%s)" % quoted if dry_run else quoted)
        if dry_run:
            print(quoted)
        else:
            infer_id = _sbatch(i_args)
            if infer_id is not None:
                print("infer job=%d" % infer_id)

    # ---- final stage: score + plot + PDB, afterany on inference ----
    analyse_sb = os.path.join(outdir, "analyse.sbatch")
    if analyse and not align_only and os.path.exists(analyse_sb):
        # afterany again: score whatever finished rather than hanging when one
        # GPU group fails. The job itself refuses to run if no models exist.
        n_args = ["sbatch", "--parsable"]
        if (not no_dependency) and infer_id is not None:
            n_args += ["--dependency=afterany:%d" % infer_id]
        n_args += [analyse_sb]
        quoted = " ".join(shlex.quote(x) for x in n_args)
        if dry_run and not no_dependency and infer_id is None:
            parts = quoted.split(" ", 2)
            quoted = " ".join(parts[:2] + ["--dependency=afterany:$INFER_ID"] + parts[2:])
        cmds_log.append(quoted)
        if dry_run:
            print(quoted)
        else:
            aid = _sbatch(n_args)
            if aid is not None:
                print("analyse job=%d (metrics + figures + PDB when inference "
                      "finishes)" % aid)

    if dry_run:
        sh = os.path.join(outdir, "SUBMIT_CMDS.sh")
        with open(sh, "w") as fh:
            fh.write("#!/usr/bin/env bash\nset -euo pipefail\n")
            fh.write("\n".join(cmds_log) + "\n")
        print("dry-run -> %s" % sh)
    return align_id, infer_id


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="run")
    # chain A: --chainA / --a-set / -A
    ap.add_argument("--chainA", "--a-set", "-A", dest="chainA",
                    help="comma-sep IDs (UniProt acc / TAIR locus / LABEL=SEQ)")
    # chain B: --chainB / --b-set / -B
    ap.add_argument("--chainB", "--b-set", "-B", dest="chainB",
                    help="comma-sep IDs")
    ap.add_argument("--outdir", default="runs/run")
    ap.add_argument("--mode", choices=["grid", "complex"], default="grid",
                    help="grid: A x B pairwise 2-chain folds. "
                         "complex: one multi-chain fold.")
    ap.add_argument("--complex-name", default=None,
                    help="name for the single multi-chain fold (mode=complex)")
    ap.add_argument("--seeds", default=_UNSET,
                    help="comma-sep ints. Default sourced from config.yaml "
                         "(seeds:) or '1' if config silent. "
                         "BAKED INTO JSON in stage 1 -- change requires rebuild.")
    ap.add_argument("--num-samples", "--diffusion-samples", dest="num_samples",
                    type=int, default=5,
                    help="AF3 --num_diffusion_samples per seed")
    ap.add_argument("--num-recycles", type=int, default=10)
    ap.add_argument("--cache", "--seq-cache", dest="cache", default="seq_cache",
                    help="sequence cache dir")
    ap.add_argument("--chains-tsv", default=None, metavar="TSV",
                    help="chain-set definition file instead of --chainA/--chainB. "
                         "Columns: chain(A|B), id, [sequence]. See --chains-tsv-template.")
    ap.add_argument("--chains-tsv-template", default=None, metavar="PATH",
                    help="write a starter chains TSV to PATH and exit")
    ap.add_argument("--fasta", action="append", default=[],
                    help="FASTA file(s) overriding ID lookup (repeatable)")
    ap.add_argument("--organism-id", type=int, default=3702,
                    help="UniProt taxonomy ID for TAIR-locus resolution "
                         "(default 3702 = A. thaliana)")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--rebuild", action="store_true",
                    help="force re-emit JSONs/sbatches even if outdir is non-empty")

    # submit
    ap.add_argument("--submit", action="store_true",
                    help="submit the pipeline: align array -> PACKED GPU "
                         "inference via a bridge job (default). Use "
                         "--array-infer for the legacy per-task infer array.")
    ap.add_argument("--no-analyse", action="store_true",
                    help="with --submit: do NOT chain the final scoring/plot/PDB job")
    ap.add_argument("--array-infer", action="store_true",
                    help="with --submit: use the legacy one-task-per-prediction "
                         "infer array instead of the default packed route")
    ap.add_argument("--align-only", action="store_true")
    ap.add_argument("--infer-only", action="store_true")
    ap.add_argument("--no-dependency", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --submit: write SUBMIT_CMDS.sh; don't sbatch")
    ap.add_argument("--time-align", default=None,
                    help="override align wallclock for this submission")
    ap.add_argument("--time-infer", default=None,
                    help="override infer wallclock (e.g. 06:00:00 for large >2500-res complexes)")

    # pack (packed GPU inference -- strategy after Mau's bwHelix AF3 toolkit)
    ap.add_argument("--pack", action="store_true",
                    help="group align-stage *_data.json into few fat "
                         "single-bucket GPU jobs (af3pack/ + submit_all.sh). "
                         "Run AFTER the align array, on the cluster.")
    ap.add_argument("--target-hours", type=float, default=1.5,
                    help="with --pack: aim each packed job near this walltime")
    ap.add_argument("--safety", type=float, default=1.3,
                    help="with --pack: walltime margin on the estimate")
    ap.add_argument("--calib", default=None,
                    help="with --pack: calib.json from `python -m af3lis.calibrate` "
                         "(defaults to <outdir>/calib.json when present)")
    ap.add_argument("--pack-max-jobs", type=int, default=90,
                    help="with --pack: warn above this many jobs (~100 accrue age)")
    ap.add_argument("--pack-copy", action="store_true",
                    help="with --pack: copy data JSONs into groups (default symlink)")
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="with --pack: build ONE probe group of N modal-bucket "
                         "conditions (separate out root) for calibration")

    # collect
    ap.add_argument("--collect", metavar="OUT_DIR",
                    help="skip build; just collect metrics for this AF3 out dir")
    ap.add_argument("-o", "--output", default=None,
                    help="metrics TSV (with --collect)")
    ap.add_argument("--workers", "-w", type=int, default=8)
    ap.add_argument("--rank-by", "--rank", dest="rank_by", default="iLIS_max",
                    help="sort column for the output TSV "
                         "(boltzlis 'iLIS_max' auto-translates in per_seed mode)")
    ap.add_argument("--agg", choices=["per_seed", "flat"], default="per_seed",
                    help="aggregation mode (default per_seed; 'flat' matches "
                         "boltzlis TSV schema byte-for-byte for cross-engine plots "
                         "-- only with --no-ipsae)")
    ap.add_argument("--no-ipsae", action="store_true",
                    help="with --collect: skip the Dunbrack ipsae.py metric "
                         "family (ipSAE_d0res/d0chn/d0dom, ipTM_d0chn, pDockQ, "
                         "pDockQ2, LIS_ipsae)")
    ap.add_argument("--ipsae-pae-cutoff", type=float, default=10.0)
    ap.add_argument("--ipsae-dist-cutoff", type=float, default=10.0)
    ap.add_argument("--lis-py", default=None, help="override vendored lis.py path")
    ap.add_argument("--python", default=None, help="override analysis python")
    # boltzlis-CLI parity: accept and ignore (AF3 only emits CIF).
    ap.add_argument("--output-format", choices=["cif"], default="cif",
                    help="AF3 emits CIF only; flag kept for boltzlis CLI parity")
    return ap


def main() -> None:
    ap = _build_argparser()
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg_path = a.config if (a.config and os.path.exists(a.config)) else None

    # ---- chains-TSV template short-circuit ----
    if a.chains_tsv_template:
        from af3lis.chain_input import write_template
        write_template(a.chains_tsv_template)
        print("wrote starter chains TSV -> %s" % a.chains_tsv_template)
        return

    # ---- collect short-circuit ----
    if a.collect:
        if a.output:
            out = a.output
        else:
            # Default to <collect>/../metrics.tsv via abspath so `--collect .`
            # writes to the parent (not into the AF3 out dir alongside jobs).
            parent = os.path.dirname(os.path.abspath(a.collect.rstrip("/")))
            out = os.path.join(parent, "metrics.tsv")
        python_exe = a.python or cfg.get("python")
        agg = collect_all(a.collect, out, workers=a.workers,
                          rank_by=a.rank_by, python_exe=python_exe,
                          lis_py=a.lis_py, agg_mode=a.agg,
                          ipsae=not a.no_ipsae,
                          ipsae_pae_cutoff=a.ipsae_pae_cutoff,
                          ipsae_dist_cutoff=a.ipsae_dist_cutoff)
        print("wrote %s (%d pairs)" % (out, len(agg)))
        return

    # ---- pack short-circuit ----
    if a.pack:
        if not a.outdir:
            ap.error("--pack requires --outdir")
        from af3lis.pack import cmd_pack
        calib = a.calib
        if calib is None:
            default_calib = os.path.join(a.outdir, "calib.json")
            if os.path.exists(default_calib):
                calib = default_calib
                print("[pack] using %s" % default_calib)
        cmd_pack(a.outdir, target_hours=a.target_hours, safety=a.safety,
                 calib_path=calib, max_jobs=a.pack_max_jobs,
                 copy=a.pack_copy, probe=a.probe)
        return

    # ---- submit short-circuit ----
    if a.submit:
        if not a.outdir:
            ap.error("--submit requires --outdir")
        # Seeds for the bake-check: use --seeds if the user explicitly passed
        # it, else read the authoritative modelSeeds from the first JSON in
        # the list file (this is what's actually baked in). CLI default '1'
        # was making every multi-seed run unsubmittable.
        listfile = os.path.join(a.outdir, "af3_json_list.txt")
        if a.seeds is _UNSET:
            if os.path.exists(listfile):
                cfg = dict(cfg, seeds=_read_seeds_from_first_json(listfile))
            else:
                # Fall back to config.yaml seeds (or default [1]).
                cfg = dict(cfg, seeds=cfg.get("seeds", [1]))
        else:
            cfg = dict(cfg, seeds=_parse_seeds_value(a.seeds))
        cmd_submit(a.outdir, cfg,
                   align_only=a.align_only, infer_only=a.infer_only,
                   no_dependency=a.no_dependency, dry_run=a.dry_run,
                   time_align=a.time_align, time_infer=a.time_infer,
                   array_infer=a.array_infer, analyse=not a.no_analyse)
        return

    # ---- build ----
    tsv_overrides: dict[str, str] = {}
    if a.chains_tsv:
        from af3lis.chain_input import read_chain_tsv
        if a.chainA or a.chainB:
            ap.error("--chains-tsv is mutually exclusive with --chainA/--chainB")
        a.chainA, a.chainB, tsv_overrides = read_chain_tsv(a.chains_tsv)
        print("[build] %s -> chainA=%s  chainB=%s%s"
              % (a.chains_tsv, a.chainA, a.chainB,
                 ("  (+%d inline sequence(s))" % len(tsv_overrides))
                 if tsv_overrides else ""))
    if not (a.chainA and a.chainB):
        ap.error("--chainA and --chainB (or --chains-tsv) are required "
                 "(unless --collect or --submit)")
    # complex mode: --complex-name is recommended but optional. cname falls
    # back to a.name below; the dead "a.name == 'run'" guard was removed.
    if a.mode == "complex" and not a.complex_name:
        sys.stderr.write(
            "[build] WARNING: --mode complex without --complex-name; falling "
            "back to --name (%r) as the assembly name.\n" % a.name
        )

    # Seed sourcing precedence: explicit --seeds > config.yaml > default [1].
    if a.seeds is _UNSET:
        seeds = cfg.get("seeds", [1])
        if isinstance(seeds, str):
            seeds = _parse_seeds_value(seeds)
    else:
        seeds = _parse_seeds_value(a.seeds)
    if not seeds:
        ap.error("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        ap.error("--seeds contains duplicates %s — AF3 treats them as "
                 "distinct; refusing to silently dedupe" % seeds)

    # Guard against accidental clobber of an existing build.
    listfile_existing = os.path.join(a.outdir, "af3_json_list.txt")
    if os.path.exists(listfile_existing) and not a.rebuild:
        ap.error(
            "%s already has a built screen (af3_json_list.txt exists). "
            "Pass --rebuild to overwrite, or use a fresh --outdir."
            % a.outdir
        )

    overrides: dict[str, str] = dict(tsv_overrides)
    for f in a.fasta:
        overrides.update(fetch.load_fasta(f))   # --fasta wins over inline TSV

    os.makedirs(a.outdir, exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "out"), exist_ok=True)

    A = fetch.resolve_chain(a.chainA, a.cache, overrides,
                            organism_id=a.organism_id)
    B = fetch.resolve_chain(a.chainB, a.cache, overrides,
                            organism_id=a.organism_id)
    print("chain A: %s" % ", ".join("%s(%daa)" % (l, len(s)) for l, s in A))
    print("chain B: %s" % ", ".join("%s(%daa)" % (l, len(s)) for l, s in B))

    jdir = os.path.join(a.outdir, "jsons")
    os.makedirs(jdir, exist_ok=True)
    if a.mode == "grid":
        jobs = json_build.build_grid(A, B, jdir, seeds=seeds)
    else:
        cname = a.complex_name or a.name
        jobs = json_build.build_complex(A, B, cname, jdir, seeds=seeds)
    print("built %d AF3 JSON(s) -> %s" % (len(jobs), jdir))

    align_sb, infer_sb, listfile = render_sbatches(jobs, a.outdir, a.name, cfg)
    runbook = write_runbook(a.outdir, a.name, jobs, a.mode, seeds,
                            a.num_samples, align_sb, infer_sb, cfg_path)

    print("align sbatch -> %s   (sbatch --array=1-%d %s)"
          % (align_sb, len(jobs), align_sb))
    print("infer sbatch -> %s   (sbatch --dependency=afterok:<align_id> "
          "--array=1-%d %s)" % (infer_sb, len(jobs), infer_sb))
    print("list file    -> %s" % listfile)
    print("runbook      -> %s" % runbook)
    print("\nNext: sync %s to the cluster, then either follow RUNBOOK.md or "
          "run `python pipeline.py --submit --outdir %s`."
          % (a.outdir, a.outdir))


if __name__ == "__main__":
    main()
