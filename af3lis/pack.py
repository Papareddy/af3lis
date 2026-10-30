"""Packed GPU inference -- group many predictions into few fat SLURM jobs.

Strategy adapted (with thanks) from Mau's bwHelix AF3 pipeline toolkit
(af3_group.py, Aug 2026). The bwHelix queue reality that motivates it:

  * MaxJobsAccruePU = 100  -- only ~100 of your jobs accrue age priority at once
  * bf_max_job_user = 128  -- only 128 jobs are backfill-tested per cycle
  * array tasks FORFEIT age (accrue clock resets on activation)

so a per-prediction array is mostly inert queue weight, and every task re-pays
weights-load + XLA compile. Instead: AF3's ``run_alphafold.py --input_dir`` runs
every ``*_data.json`` in a directory in ONE process -- weights load once, XLA
compiles once per token bucket and is reused.

This module splits an af3lis run dir's align-stage outputs into single-bucket
groups sized to hit a target walltime, symlinks the ``*_data.json`` into
``af3pack/group_NNN_b<bucket>/``, and writes ``submit_all.sh`` driving the
rendered ``pack_infer.sbatch`` (which carries our Helix XLA/memory env).

Timing model per group (Mau's, unchanged):
    walltime = (startup + compile + n_conditions * sec[bucket] * n_seeds) * safety

The default per-bucket table is deliberately conservative. Buckets 512/768
were measured on A100 (~30/~75 s per condition, 1 seed x 5 samples) and 3584
on the large-model tier; the rest are ~bucket^2.2 interpolations. Run a probe
group + ``python -m af3lis.calibrate`` and pass ``--calib calib.json`` before
sizing a real campaign on a new size class / GPU.

CLI:
    python -m af3lis.pack --outdir runs/SCREEN [--target-hours 1.5]
        [--safety 1.3] [--calib calib.json] [--max-jobs 90] [--copy]
        [--probe 10]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

# AF3's default token-bucket ladder: an input is padded up to the smallest
# bucket >= its token count, and XLA compiles once per bucket. This is the
# --buckets flag default of run_alphafold.py at tag v3.0.1
# (https://github.com/google-deepmind/alphafold3/blob/v3.0.1/run_alphafold.py).
# Above the largest bucket AF3 makes a new bucket of EXACTLY the token count
# (pipeline.calculate_bucket_size), so each such input compiles on its own.
# This list used to skip 2560/3584/4608, so a 3,275-token complex that AF3 ran
# in bucket 3584 was sized here as a 4096 -- and still hit its walltime.
BUCKETS = [256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096,
           4608, 5120]

# Seconds per condition (1 seed x 5 diffusion samples). Conservative on
# purpose: an over-ask costs little queue time on this cluster, an under-ask
# kills the job at its walltime after the GPU time is spent.
#   * 512 / 768   MEASURED on A100 (Mau, Aug 2026).
#   * 1024 / 1280 a later probe measured ~87 / ~140 s per condition, startup
#                 included; the table stays above that.
#   * 3584        MEASURED on the large-model tier (XLA attention, 96gb, any
#                 card): one 3,275-token condition needed 1:28:02 of job time,
#                 startup included. 5400 s covers it.
#   * 4096-5120   re-anchored on the 3584 measurement at ~bucket^2.2. They
#                 were 3000 / 4800, i.e. LESS than a smaller input measured.
#   * the rest    ~bucket^2.2 interpolations between measured points.
# Recalibrate (--probe + af3lis.calibrate) for other GPUs / size classes.
DEFAULT_CALIB = {256: 15, 512: 35, 768: 85, 1024: 160, 1280: 260, 1536: 380,
                 2048: 700, 2560: 1150, 3072: 1600, 3584: 5400, 4096: 7300,
                 4608: 9400, 5120: 11900}
SCALING_EXPONENT = 2.2   # seconds ~ bucket^2.2 between and beyond measurements
# ---------------------------------------------------------------------------
# Size-aware SLURM resources.
#
# Every group holds ONE token bucket, so the group's resource need is known
# before submission and each job can ask for exactly what its size class
# requires instead of one global setting sized for the worst case.
#
# The memory ladder is the load-bearing part on bwHelix and it is NOT a simple
# "bigger is safer": asking >= 64 GB excludes the 26 abundant 40 GB gpu4 nodes
# and restricts the job to 4 scarce gpu8 nodes, so a needlessly large --mem
# costs hours of queue. Small buckets therefore stay at 48 GB deliberately;
# only > 3072 tokens (bucket 3584 and up) escalates, because those genuinely
# need a large card or a host-memory spill.
#
# Do NOT pin a GPU type unless the size class needs one. Measured with
# `sbatch --test-only` on bwHelix, same job, same --mem:
#
#     --gres=gpu:1        start 2026-10-03T21:22   <- earliest
#     --gres=gpu:A100:1   start 2026-10-03T23:37   (+2h15m)
#     --gres=gpu:A40:1    start 2026-10-04T00:45   (+3h23m)
#
# Pinning costs queue time for no benefit below ~3072 tokens: every GPU type on
# this cluster (a100 / a40 / h200) is Ampere+ so Triton flash-attention works on
# all of them, and all carry >= 40 GB, which AF3 clears for ~3100-residue
# complexes with unified memory enabled. Small jobs are minutes of compute
# after days of queue -- 2:25 and 1:44 for the b256 and b512 groups of the
# smoke run -- so backfill eligibility dominates everything else.
#
# NB `gpumem_per_gpu` does NOT influence scheduling here: 40G/48G/80G all
# return the same estimate as an unpinned request, so it cannot be used to
# guarantee a large card. --mem is the real lever (>=64gb restricts to the
# scarce gpu8 nodes, which is exactly what a big model needs and what a small
# one must avoid).
#
# (bucket_max, gres, mem, flash_attn, xla_mem_frac, note)
RESOURCE_LADDER = [
    (1536, "gpu:1", "48gb", "triton", 3.2,
     "any GPU type; 48gb keeps the abundant gpu4 nodes eligible"),
    (3072, "gpu:1", "60gb", "triton", 3.2,
     "any GPU type, still under the 64gb gpu4 cutoff; more host RAM for spill"),
    (10 ** 9, "gpu:1", "96gb", "xla", 4.0,
     ">3072 tokens: any type; >=64gb reaches the big nodes, XLA kernel + a "
     "4.0 unified-memory pool covers a smaller card by spilling to host"),
]

# The ladder never names a card. Measured on bwHelix at the big-model tier
# (--mem=96gb, same job):
#
#     --gres=gpu:1        start 2026-10-03T21:29   <- earliest
#     --gres=gpu:H200:1   start 2026-10-03T21:29      (identical)
#     --gres=gpu:A100:1   start 2026-10-03T23:37      (+2h08m)
#
# A100 is the CONTENDED card here, so pinning it -- the intuitive "give the big
# model the big card" move -- is the slowest option available. A multi-type
# request (gpu:A100:1,gpu:H200:1) does not parse, so unpinned is the only way
# to stay flexible, and it happens to be the fastest.
#
# Quality is not traded for it. --mem >= 64gb is what actually selects the
# large nodes (Mau's documented gpu4/gpu8 split), and TF_FORCE_UNIFIED_MEMORY
# with XLA_CLIENT_MEM_FRACTION 4.0 means a job that does land on a 48 GB card
# spills to host RAM and runs slower rather than failing. The failure mode is
# degraded speed, never a wrong or missing answer.


def resources_for(bucket: int) -> dict:
    """SLURM resources + XLA knobs for a single-bucket group."""
    for hi, gres, mem, flash, frac, note in RESOURCE_LADDER:
        if bucket <= hi:
            return dict(bucket=bucket, gres=gres, mem=mem, flash_attn=flash,
                        xla_mem_frac=frac, note=note)
    raise AssertionError("unreachable: ladder has a catch-all")


DEFAULT_STARTUP = 90.0    # weights load + first-ever compile (s)
DEFAULT_COMPILE = 40.0    # per-bucket compile (s); 1 bucket/group -> paid once


def bucket_of(tokens: int) -> int:
    """AF3's bucket for an input: the smallest ladder bucket >= tokens, and
    above the ladder exactly `tokens` -- AF3 makes a bucket of that size."""
    for b in BUCKETS:
        if tokens <= b:
            return b
    return int(tokens)


def default_sec(bucket: int) -> float:
    """Built-in seconds per condition for any bucket, including AF3's
    exact-size buckets above the ladder (~bucket^2.2 from the largest tabled
    bucket below). An untabled bucket used to get a flat 700 s -- a
    6000-token input sized at a seventh of a 5120 one."""
    if bucket in DEFAULT_CALIB:
        return float(DEFAULT_CALIB[bucket])
    below = [b for b in DEFAULT_CALIB if b <= bucket]
    ref = max(below) if below else min(DEFAULT_CALIB)
    return float(DEFAULT_CALIB[ref]) * (bucket / ref) ** SCALING_EXPONENT


def token_count(input_json: dict) -> int:
    """Approximate AF3 token count of one input JSON (protein/dna/rna residues
    x chain multiplicity + 1 token per ligand copy). Matches Mau's toks()."""
    n = 0
    for s in input_json.get("sequences", []):
        for mol in ("protein", "dna", "rna"):
            if mol in s:
                seq = s[mol].get("sequence", "")
                ids = s[mol].get("id", "A")
                n += len(seq) * (len(ids) if isinstance(ids, list) else 1)
        if "ligand" in s:
            ids = s["ligand"].get("id", "A")
            n += (len(ids) if isinstance(ids, list) else 1)
    return n


# A measured table more than this much faster than the built-in one, in any
# bucket, is refused unless forced. The built-in table over-asks by up to ~2x
# where it has been checked, so a genuine calibration stays inside the limit;
# the one that did not was ~5x too fast from a method error, and would have
# packed ~2 h of work into a ~30 min walltime.
CALIB_MAX_SPEEDUP = 2.0


def check_calib(table: dict[int, float],
                max_speedup: float = CALIB_MAX_SPEEDUP) -> list[str]:
    """Why a measured {bucket: s/condition} table is implausible (empty if
    it is not): any bucket > max_speedup x faster than the built-in time."""
    reasons = []
    for b in sorted(table):
        ref, v = default_sec(b), float(table[b])
        if v * max_speedup < ref:
            reasons.append(
                f"bucket {b}: {v:.1f} s/condition is {ref / max(v, 1e-9):.1f}x "
                f"faster than the built-in {ref:.0f} s (limit {max_speedup:g}x)")
    return reasons


def load_calib(path: str | None, force: bool = False
               ) -> tuple[dict[int, float], float, float, dict]:
    """(calib, startup, compile_s, info) for a --calib path.

    A calibration is used ONLY when a path is given -- nothing is picked up
    from the run dir by itself. A path that is missing, unreadable, or
    implausibly fast (check_calib; overridden by force=True or by a
    calibration that af3lis.calibrate wrote with --force) falls back to the
    built-in conservative table, and ``info`` says which and why, so the
    caller can say so loudly. Falling back rather than crashing matters:
    pack runs unattended in the bridge job, where a crash strands the whole
    campaign after alignment, and the built-in table errs on the safe side.
    """
    calib = {int(k): float(v) for k, v in DEFAULT_CALIB.items()}
    startup, compile_s = DEFAULT_STARTUP, DEFAULT_COMPILE
    info: dict = {"source": "built-in conservative table", "path": None,
                  "status": "default", "reasons": []}
    if not path:
        return calib, startup, compile_s, info
    info["path"] = os.path.abspath(path)
    if not os.path.exists(path):
        info.update(status="missing", reasons=[f"{path} does not exist"])
        return calib, startup, compile_s, info
    try:
        with open(path) as fh:
            c = json.load(fh)
        table = {int(k): float(v) for k, v in c.items()
                 if not str(k).startswith("_") and k not in ("startup", "compile")}
        st = float(c.get("startup", startup))
        co = float(c.get("compile", compile_s))
    except (OSError, ValueError, TypeError, AttributeError) as e:
        info.update(status="unreadable", reasons=[f"{path}: {e}"])
        return calib, startup, compile_s, info
    reasons = check_calib(table)
    if reasons and not (force or c.get("_forced")):
        info.update(status="refused", reasons=reasons)
        return calib, startup, compile_s, info
    calib.update(table)
    info.update(source=os.path.abspath(path),
                status="forced" if reasons else "measured",
                reasons=reasons, buckets=sorted(table))
    return calib, st, co, info


def _report_calib(info: dict, startup: float, compile_s: float) -> None:
    st = info["status"]
    if st == "default":
        print("[pack] calibration: built-in conservative table "
              "(no --calib given)")
    elif st in ("measured", "forced"):
        print(f"[pack] calibration: {info['path']} -- measured bucket(s) "
              f"{', '.join(map(str, info['buckets']))}, built-in table for "
              f"the rest; startup {startup:g} s, compile {compile_s:g} s")
        if st == "forced":
            sys.stderr.write("[pack] WARNING: this calibration is FORCED past "
                             "the plausibility check:\n")
            for r in info["reasons"]:
                sys.stderr.write(f"[pack]   {r}\n")
    else:
        sys.stderr.write(
            f"[pack] WARNING: calibration {info['path']} {st.upper()} -- "
            f"sizing from the built-in conservative table instead:\n")
        for r in info["reasons"]:
            sys.stderr.write(f"[pack]   {r}\n")
        if st == "refused":
            sys.stderr.write("[pack]   (pass --force-calib if a faster GPU "
                             "genuinely explains it)\n")


def _write_calib_used(pack_root: str, calib: dict[int, float],
                      startup: float, compile_s: float, info: dict) -> None:
    """Record what actually sized this pack, beside groups.tsv."""
    os.makedirs(pack_root, exist_ok=True)
    with open(os.path.join(pack_root, "calib_used.json"), "w") as fh:
        json.dump({"source": info["source"], "status": info["status"],
                   "requested": info["path"], "reasons": info["reasons"],
                   "startup": startup, "compile": compile_s,
                   "sec_per_condition": {str(b): calib[b]
                                         for b in sorted(calib)}},
                  fh, indent=2)


def _walltime_str(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def scan_run_dir(outdir: str) -> tuple[list[dict], list[str]]:
    """Inventory the run dir: one record per input JSON with its bucket,
    seed count and align-stage data.json path.

    Token counts come from the small build-stage ``jsons/*.json`` (sequences
    only), NOT the MSA-laden ``*_data.json`` -- scanning thousands of those on
    a login node would be slow for no gain (sequences are identical).

    Returns (records, missing) where ``missing`` lists conditions whose align
    output is absent (failed/unfinished align tasks). Following Mau's
    afterany philosophy, missing conditions are reported and EXCLUDED rather
    than aborting the whole pack.
    """
    jdir = os.path.join(outdir, "jsons")
    out_root = os.path.join(outdir, "out")
    inputs = sorted(glob.glob(os.path.join(jdir, "*.json")))
    if not inputs:
        raise SystemExit(f"no input JSONs under {jdir} -- is this an af3lis run dir?")
    records: list[dict] = []
    missing: list[str] = []
    for jp in inputs:
        with open(jp) as fh:
            d = json.load(fh)
        name = str(d.get("name", os.path.splitext(os.path.basename(jp))[0]))
        lname = name.lower()
        data = os.path.join(out_root, lname, f"{lname}_data.json")
        if not os.path.exists(data):
            # AF3 sanitizes names beyond lowercasing in edge cases; fall back
            # to an exact-filename search (same convention as the infer tmpl).
            hits = glob.glob(os.path.join(out_root, "*", f"{lname}_data.json"))
            data = hits[0] if hits else ""
        if not data:
            missing.append(lname)
            continue
        records.append(dict(
            lname=lname,
            data=os.path.abspath(data),
            bucket=bucket_of(token_count(d)),
            nseeds=max(1, len(d.get("modelSeeds", [1]))),
        ))
    return records, missing


def build_groups(records: list[dict],
                 pack_root: str,
                 target_hours: float,
                 safety: float,
                 calib: dict[int, float],
                 startup: float,
                 compile_s: float,
                 copy: bool = False) -> list[dict]:
    """Chunk records into single-bucket groups sized to ``target_hours``.

    Each group dir gets symlinks (or copies with ``copy=True``) to its
    ``*_data.json``. Symlinks are the default: data JSONs embed full MSAs and
    copying a large screen would duplicate many GB on the workspace.
    Returns one dict per group: dir, bucket, members, est_s, walltime.
    """
    os.makedirs(pack_root, exist_ok=True)
    tgt = target_hours * 3600.0
    by_bucket: dict[int, list[dict]] = {}
    for r in records:
        by_bucket.setdefault(r["bucket"], []).append(r)

    groups: list[dict] = []
    gi = 0
    for b in sorted(by_bucket):
        recs = by_bucket[b]
        sec = calib.get(b, default_sec(b))
        # conditions per group budgeted at the bucket's modal seed count
        nseeds_mode = Counter(r["nseeds"] for r in recs).most_common(1)[0][0]
        per = max(1, int((tgt - startup - compile_s) // max(sec * nseeds_mode, 1.0)))
        for i in range(0, len(recs), per):
            chunk = recs[i:i + per]
            gd = os.path.join(pack_root, f"group_{gi:03d}_b{b}")
            os.makedirs(gd, exist_ok=True)
            for r in chunk:
                dst = os.path.join(gd, os.path.basename(r["data"]))
                if os.path.lexists(dst):
                    os.remove(dst)
                if copy:
                    import shutil
                    shutil.copy2(r["data"], dst)
                else:
                    os.symlink(r["data"], dst)
            est = startup + compile_s + sum(sec * r["nseeds"] for r in chunk)
            mins = max(20, math.ceil(est * safety / 60.0 / 5) * 5)
            groups.append(dict(
                dir=os.path.abspath(gd), bucket=b,
                members=[r["lname"] for r in chunk],
                est_s=est, walltime=_walltime_str(mins),
            ))
            gi += 1
    return groups


def write_submit(groups: list[dict],
                 pack_root: str,
                 sbatch_path: str,
                 out_pack: str,
                 max_jobs: int) -> str:
    lines = ["#!/bin/bash",
             "set -e",
             f"# packed AF3 inference -- {len(groups)} jobs; keep <= ~100 pending",
             f"# (bwHelix MaxJobsAccruePU=100; trickle-submit if you queue more elsewhere)",
             "#",
             "# Job ids are captured to job_ids.txt because the ANALYSE stage must",
             "# depend on these jobs, and their ids do not exist until this script",
             "# runs -- so analyse cannot be chained when the campaign is submitted.",
             'IDS=""']
    for g in groups:
        r = resources_for(g["bucket"])
        # sbatch CLI flags override the #SBATCH directives rendered from
        # config.yaml; the two XLA knobs are not sbatch options so they travel
        # as env vars the template reads with a config-rendered fallback.
        lines.append(
            f"# bucket {r['bucket']}: {r['note']}\n"
            f"IDS=\"$IDS $(sbatch --parsable --time={g['walltime']} "
            f"--gres={r['gres']} --mem={r['mem']} "
            f"--export=ALL,AF3LIS_FLASH_ATTN={r['flash_attn']},"
            f"AF3LIS_XLA_MEM_FRAC={r['xla_mem_frac']} "
            f"--job-name=af3pack_{os.path.basename(g['dir'])} "
            f"{sbatch_path} {g['dir']} {out_pack})\""
        )
    lines += [
        "",
        "# leave the ids where the bridge can pick them up",
        f'printf "%s\\n" $IDS > "{os.path.join(pack_root, "job_ids.txt")}"',
        'echo "PACK_SUBMITTED$IDS"',
    ]
    sub = os.path.join(pack_root, "submit_all.sh")
    with open(sub, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(sub, 0o755)

    tsv = os.path.join(pack_root, "groups.tsv")
    with open(tsv, "w") as fh:
        fh.write("group\tbucket\tn_conditions\test_min\twalltime\tgres\tmem\t"
                 "flash_attn\txla_mem_frac\tmembers\n")
        for g in groups:
            r = resources_for(g["bucket"])
            fh.write(f"{os.path.basename(g['dir'])}\t{g['bucket']}\t"
                     f"{len(g['members'])}\t{g['est_s'] / 60:.1f}\t{g['walltime']}\t"
                     f"{r['gres']}\t{r['mem']}\t{r['flash_attn']}\t"
                     f"{r['xla_mem_frac']}\t{','.join(g['members'])}\n")

    if len(groups) > max_jobs:
        sys.stderr.write(
            f"[pack] WARNING: {len(groups)} jobs > --max-jobs {max_jobs}. Raise "
            f"--target-hours to pack more per job, or submit in waves keeping "
            f"~{max_jobs} pending (bwHelix age-accrue window is ~100).\n")
    return sub


def cmd_pack(outdir: str,
             target_hours: float = 1.5,
             safety: float = 1.3,
             calib_path: str | None = None,
             max_jobs: int = 90,
             copy: bool = False,
             probe: int = 0,
             force_calib: bool = False) -> None:
    outdir = os.path.abspath(outdir)
    sbatch_path = os.path.join(outdir, "pack_infer.sbatch")
    if not os.path.exists(sbatch_path):
        raise SystemExit(
            f"missing {sbatch_path} -- rebuild the screen with the current "
            f"pipeline.py (it renders pack_infer.sbatch at build time)")

    calib, startup, compile_s, cinfo = load_calib(calib_path, force=force_calib)
    _report_calib(cinfo, startup, compile_s)
    records, missing = scan_run_dir(outdir)
    if missing:
        mfile = os.path.join(outdir, "af3pack_missing.txt")
        with open(mfile, "w") as fh:
            fh.write("\n".join(missing) + "\n")
        sys.stderr.write(
            f"[pack] WARNING: {len(missing)}/{len(records) + len(missing)} "
            f"conditions have no align-stage *_data.json -- EXCLUDED from the "
            f"pack (list: {mfile}). Re-run align for them, then re-pack.\n")
    if not records:
        raise SystemExit("no conditions with align output -- run the align stage first")

    if probe:
        # Probe = one group of N conditions from the MODAL bucket, sent to a
        # SEPARATE output root so its models never mix with the real campaign.
        modal = Counter(r["bucket"] for r in records).most_common(1)[0][0]
        recs = [r for r in records if r["bucket"] == modal][:probe]
        pack_root = os.path.join(outdir, "af3pack_probe")
        out_pack = os.path.join(outdir, "probe_out")
        groups = build_groups(recs, pack_root, target_hours, safety,
                              calib, startup, compile_s, copy=copy)
        sub = write_submit(groups, pack_root, sbatch_path, out_pack, max_jobs)
        _write_calib_used(pack_root, calib, startup, compile_s, cinfo)
        print(f"probe: {len(recs)} conditions @ bucket {modal} -> {sub}")
        print(f"after it finishes:  python -m af3lis.calibrate "
              f"{outdir}/logs/pack_af3pack_group_*.out --out {outdir}/calib.json")
        print(f"then re-pack with:  --calib {outdir}/calib.json   "
              f"(it is never picked up without the flag)")
        return

    pack_root = os.path.join(outdir, "af3pack")
    out_pack = os.path.join(outdir, "out_pack")
    groups = build_groups(records, pack_root, target_hours, safety,
                          calib, startup, compile_s, copy=copy)
    sub = write_submit(groups, pack_root, sbatch_path, out_pack, max_jobs)
    _write_calib_used(pack_root, calib, startup, compile_s, cinfo)

    nseeds = Counter(r["nseeds"] for r in records).most_common(1)[0][0]
    print(f"{len(records)} conditions, {nseeds} seed(s) -> {len(groups)} packed "
          f"jobs across buckets {sorted({g['bucket'] for g in groups})}")
    print(f"{'group':>16} {'bucket':>6} {'nconds':>6} {'est_min':>8} {'walltime':>9}")
    for g in groups:
        print(f"{os.path.basename(g['dir']):>16} {g['bucket']:>6} "
              f"{len(g['members']):>6} {g['est_s'] / 60:>8.1f} {g['walltime']:>9}")
    if cinfo["status"] not in ("measured", "forced"):
        print("\nNOTE: sized from the built-in conservative table (see "
              "af3pack/calib_used.json). For a new size class or GPU, run "
              "--probe first and re-pack with --calib.")
    print(f"\nsubmit: bash {sub}   "
          f"(models -> {out_pack}; collect with --collect {out_pack})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", required=True, help="af3lis run dir (has jsons/, out/)")
    ap.add_argument("--target-hours", type=float, default=1.5,
                    help="aim each packed job near this walltime (default 1.5)")
    ap.add_argument("--safety", type=float, default=1.3,
                    help="walltime margin on the estimate (default 1.3)")
    ap.add_argument("--calib", default=None,
                    help="calib.json from af3lis.calibrate (per-bucket seconds). "
                         "Used only when given; nothing is auto-loaded.")
    ap.add_argument("--force-calib", action="store_true",
                    help="use --calib even if it is >2x faster than the "
                         "built-in table")
    ap.add_argument("--max-jobs", type=int, default=90,
                    help="warn above this many jobs (age-accrue window ~100)")
    ap.add_argument("--copy", action="store_true",
                    help="copy *_data.json into group dirs instead of symlinking")
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="build ONE probe group of N modal-bucket conditions "
                         "(separate out root) for calibration, then exit")
    a = ap.parse_args()
    cmd_pack(a.outdir, target_hours=a.target_hours, safety=a.safety,
             calib_path=a.calib, max_jobs=a.max_jobs, copy=a.copy,
             probe=a.probe, force_calib=a.force_calib)


if __name__ == "__main__":
    main()
