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

The default per-bucket table is A100; ONLY buckets 512/768 were measured
(~30/~75 s per condition, 1 seed x 5 samples) -- larger buckets are ~bucket^2.2
extrapolations. Run a probe group + ``python -m af3lis.calibrate`` and pass
``--calib calib.json`` before sizing a real campaign on a new size class / GPU.

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

# AF3 default token-bucket ladder (inputs are padded up to these sizes).
BUCKETS = [256, 512, 768, 1024, 1280, 1536, 2048, 3072, 4096, 5120]

# Seconds per condition (1 seed x 5 diffusion samples) on A100-40GB.
# 512 and 768 MEASURED on bwHelix (Mau, Aug 2026); the rest are conservative
# ~bucket^2.2 extrapolations. Recalibrate for other GPUs / size classes.
DEFAULT_CALIB = {256: 15, 512: 35, 768: 85, 1024: 160, 1280: 260, 1536: 380,
                 2048: 700, 3072: 1600, 4096: 3000, 5120: 4800}
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
# only >= 4096 tokens escalates, because those genuinely need an 80 GB card.
#
# (bucket_max, gres, mem, flash_attn, xla_mem_frac, note)
RESOURCE_LADDER = [
    (1536, "gpu:A100:1", "48gb", "triton", 3.2,
     "fits A100-40GB; 48gb keeps the abundant gpu4 nodes eligible"),
    (3072, "gpu:A100:1", "60gb", "triton", 3.2,
     "still under the 64gb gpu4 cutoff; more host RAM for unified-memory spill"),
    (10 ** 9, "gpu:A100:1", "96gb", "xla", 4.0,
     ">=4096 tokens: needs an 80GB card (gpu8) and the XLA attention kernel"),
]


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
    """Smallest ladder bucket >= tokens; beyond the ladder, round up to 1024."""
    for b in BUCKETS:
        if tokens <= b:
            return b
    return int(math.ceil(tokens / 1024.0) * 1024)


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


def _load_calib(path: str | None) -> tuple[dict[int, float], float, float]:
    calib = {int(k): float(v) for k, v in DEFAULT_CALIB.items()}
    startup, compile_s = DEFAULT_STARTUP, DEFAULT_COMPILE
    if path:
        with open(path) as fh:
            c = json.load(fh)
        startup = float(c.pop("startup", startup))
        compile_s = float(c.pop("compile", compile_s))
        calib.update({int(k): float(v) for k, v in c.items()})
    return calib, startup, compile_s


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
        sec = calib.get(b, DEFAULT_CALIB.get(b, 700.0))
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
             f"# (bwHelix MaxJobsAccruePU=100; trickle-submit if you queue more elsewhere)"]
    for g in groups:
        r = resources_for(g["bucket"])
        # sbatch CLI flags override the #SBATCH directives rendered from
        # config.yaml; the two XLA knobs are not sbatch options so they travel
        # as env vars the template reads with a config-rendered fallback.
        lines.append(
            f"# bucket {r['bucket']}: {r['note']}\n"
            f"sbatch --time={g['walltime']} "
            f"--gres={r['gres']} --mem={r['mem']} "
            f"--export=ALL,AF3LIS_FLASH_ATTN={r['flash_attn']},"
            f"AF3LIS_XLA_MEM_FRAC={r['xla_mem_frac']} "
            f"--job-name=af3pack_{os.path.basename(g['dir'])} "
            f"{sbatch_path} {g['dir']} {out_pack}"
        )
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
             probe: int = 0) -> None:
    outdir = os.path.abspath(outdir)
    sbatch_path = os.path.join(outdir, "pack_infer.sbatch")
    if not os.path.exists(sbatch_path):
        raise SystemExit(
            f"missing {sbatch_path} -- rebuild the screen with the current "
            f"pipeline.py (it renders pack_infer.sbatch at build time)")

    calib, startup, compile_s = _load_calib(calib_path)
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
        print(f"probe: {len(recs)} conditions @ bucket {modal} -> {sub}")
        print(f"after it finishes:  python -m af3lis.calibrate "
              f"{outdir}/logs/pack_*.out --out {outdir}/calib.json")
        print(f"then re-pack with:  --calib {outdir}/calib.json")
        return

    pack_root = os.path.join(outdir, "af3pack")
    out_pack = os.path.join(outdir, "out_pack")
    groups = build_groups(records, pack_root, target_hours, safety,
                          calib, startup, compile_s, copy=copy)
    sub = write_submit(groups, pack_root, sbatch_path, out_pack, max_jobs)

    nseeds = Counter(r["nseeds"] for r in records).most_common(1)[0][0]
    print(f"{len(records)} conditions, {nseeds} seed(s) -> {len(groups)} packed "
          f"jobs across buckets {sorted({g['bucket'] for g in groups})}")
    print(f"{'group':>16} {'bucket':>6} {'nconds':>6} {'est_min':>8} {'walltime':>9}")
    for g in groups:
        print(f"{os.path.basename(g['dir']):>16} {g['bucket']:>6} "
              f"{len(g['members']):>6} {g['est_s'] / 60:>8.1f} {g['walltime']:>9}")
    if not calib_path:
        print("\nNOTE: sized from the DEFAULT calibration (only buckets 512/768 "
              "measured, A100). For a new size class or GPU, run --probe first "
              "and re-pack with --calib.")
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
                    help="calib.json from af3lis.calibrate (per-bucket seconds)")
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
             probe=a.probe)


if __name__ == "__main__":
    main()
