"""Calibrate per-bucket AF3 inference time from packed-job logs.

Started as a port of Mau's af3_calibrate.py (bwHelix AF3 toolkit, Aug 2026),
so ``af3lis.pack`` can size walltimes from measured numbers instead of the
built-in conservative table.

Method. AF3's absl log prints "Calculating bucket size for input with N
tokens ... Got bucket size B" once per SEED per condition: run_alphafold.py
featurises every seed of a condition up front (featurisation.featurise_input
loops over rng_seeds), then runs inference seed by seed. So the bucket lines
of one condition arrive in a burst of n_seeds, seconds apart, followed by the
long inference gap. A condition's time is the gap from the first line of its
burst to the first line of the next condition's burst, divided by n_seeds to
get the per-seed figure ``af3lis.pack`` multiplies back up.

The original method took the gap between ANY two successive bucket lines.
That is right for one seed and badly wrong for several: with 5 seeds, four of
every five gaps are featurisation only, so the median lands on featurisation
time -- the most likely reason a calibration came out ~5x too fast (1024 ->
17.6 s and 1280 -> 78.2 s per condition, against ~87 s and ~140 s measured),
which would have sized a real campaign to die at its walltime.

Guards, each recorded in the output's ``_meta``:
  * every log is parsed on its own -- no gap ever spans two jobs;
  * the first condition of each bucket in each log carries the XLA compile
    and is left out;
  * a condition faster than ``--min-seconds`` per seed is not a real
    5-sample inference (a skipped or failed input) and is left out;
  * a bucket with fewer than ``--min-per-bucket`` measured conditions is
    dropped, so ``af3lis.pack`` falls back to the built-in time for it;
  * a calibration more than 2x faster than the built-in conservative table
    in any bucket is REFUSED unless ``--force``, and the reason is written
    next to where the calibration would have gone.

Usage:
    python -m af3lis.calibrate <outdir>/logs/pack_*.out --out calib.json
    python pipeline.py --pack --outdir <outdir> --calib calib.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from functools import reduce

TS = re.compile(r"(\d\d):(\d\d):(\d\d)\.(\d+)")  # absl log timestamp
BK = re.compile(r"Calculating bucket size for input with (\d+) tokens")
GOT = re.compile(r"Got bucket size (\d+)")
# run_alphafold.py prints the seed tuple once per condition (stdout, no
# timestamp -- used only to COUNT seeds, never for timing or order)
SEEDS = re.compile(r"Featurising data for seeds [\(\[]([^\)\]]*)[\)\]]")

MIN_PER_BUCKET = 3
MIN_SECONDS = 10.0


def _secs(m: re.Match) -> float:
    h, mm, s, frac = m.groups()
    return int(h) * 3600 + int(mm) * 60 + int(s) + float("0." + frac)


def parse_log(log_path: str) -> dict:
    """One packed-job log -> {"lines": [(ts, tokens, bucket)], "first_ts",
    "seed_counts": [n per 'Featurising data for seeds' line]}.

    Handles midnight wraparound in the HH:MM:SS timestamps.
    """
    lines: list[tuple[float, int, int]] = []
    pending: tuple[float, int] | None = None
    seed_counts: list[int] = []
    first_ts: float | None = None
    prev_raw: float | None = None
    day_offset = 0.0
    with open(log_path, errors="replace") as fh:
        for line in fh:
            ms = SEEDS.search(line)
            if ms:
                seed_counts.append(len(re.findall(r"\d+", ms.group(1))))
            mt = TS.search(line)
            if not mt:
                continue
            raw = _secs(mt)
            if prev_raw is not None and raw < prev_raw - 1.0:
                day_offset += 86400.0  # clock wrapped past midnight
            prev_raw = raw
            ts = raw + day_offset
            if first_ts is None:
                first_ts = ts
            mb = BK.search(line)
            if mb:
                pending = (ts, int(mb.group(1)))
                continue
            mg = GOT.search(line)
            if mg and pending is not None:
                lines.append((pending[0], pending[1], int(mg.group(1))))
                pending = None
    return {"lines": lines, "first_ts": first_ts, "seed_counts": seed_counts}


def infer_nseeds(parsed: dict) -> tuple[int, str]:
    """Seeds per condition for one log, and how it was decided.

    Preferred: the 'Featurising data for seeds (...)' print. Otherwise the
    GCD of the run lengths of identical consecutive token counts (one run per
    condition, unless two neighbours happen to share a length) -- the last
    run is left out because a job killed mid-featurisation truncates it.
    """
    counts = set(parsed["seed_counts"])
    if len(counts) == 1:
        return counts.pop(), "seed print"
    runs: list[int] = []
    prev = None
    for _, t, _ in parsed["lines"]:
        if runs and t == prev:
            runs[-1] += 1
        else:
            runs.append(1)
        prev = t
    usable = runs[:-1] if len(runs) > 1 else runs
    if not usable:
        return 1, "default"
    return reduce(math.gcd, usable), "token-count runs"


def conditions(parsed: dict, nseeds: int) -> list[tuple[float, int]]:
    """[(start_ts, bucket)] per condition: chunks of nseeds bucket lines.

    Raises ValueError if a chunk mixes token counts (the seed count is wrong
    for this log). A trailing partial chunk -- a job killed mid-featurisation
    -- is dropped.
    """
    lines = parsed["lines"]
    out = []
    for i in range(0, len(lines) - nseeds + 1, nseeds):
        chunk = lines[i:i + nseeds]
        if len({t for _, t, _ in chunk}) != 1:
            raise ValueError(
                f"bucket lines {i}-{i + nseeds - 1} mix token counts -- "
                f"{nseeds} seed(s) per condition does not fit this log")
        out.append((chunk[0][0], chunk[0][2]))
    return out


def calibrate(log_paths, compile_s: float = 40.0, nseeds: int | None = None,
              min_per_bucket: int = MIN_PER_BUCKET,
              min_seconds: float = MIN_SECONDS) -> dict:
    """Packed-job log(s) -> {"<bucket>": median per-seed s, ..., "startup",
    "compile", "_meta": {...}}.

    ``log_paths`` is one path or a list; each log is parsed on its own.
    """
    if isinstance(log_paths, (str, os.PathLike)):
        log_paths = [log_paths]
    per_bucket: dict[int, list[float]] = {}
    startups: list[float] = []
    meta = {"method": "per-condition (seed-aware), one log at a time",
            "logs": 0, "logs_without_bucket_lines": 0, "seeds": {},
            "skipped_logs": {}, "excluded_compile": 0, "excluded_fast": 0,
            "min_seconds": min_seconds, "min_per_bucket": min_per_bucket}
    for lp in log_paths:
        parsed = parse_log(lp)
        if not parsed["lines"]:
            meta["logs_without_bucket_lines"] += 1
            continue
        meta["logs"] += 1
        if nseeds:
            ns, how = nseeds, "--seeds"
        else:
            ns, how = infer_nseeds(parsed)
        meta["seeds"][os.path.basename(lp)] = f"{ns} ({how})"
        try:
            conds = conditions(parsed, ns)
        except ValueError as e:
            meta["skipped_logs"][os.path.basename(lp)] = str(e)
            continue
        if conds and parsed["first_ts"] is not None:
            startups.append(conds[0][0] - parsed["first_ts"])
        seen: set[int] = set()
        for (ts, b), (ts2, _) in zip(conds, conds[1:]):
            if b not in seen:
                seen.add(b)               # first of its bucket: XLA compile
                meta["excluded_compile"] += 1
                continue
            sec = (ts2 - ts) / ns
            if sec < min_seconds:
                meta["excluded_fast"] += 1
                continue
            per_bucket.setdefault(b, []).append(sec)

    out: dict = {}
    meta["n_per_bucket"] = {}
    meta["dropped_buckets"] = {}
    for b in sorted(per_bucket):
        n = len(per_bucket[b])
        if n >= min_per_bucket:
            out[str(b)] = round(statistics.median(per_bucket[b]), 1)
            meta["n_per_bucket"][str(b)] = n
        else:
            meta["dropped_buckets"][str(b)] = (
                f"{n} measured condition(s) < {min_per_bucket}")
    if startups:
        out["startup"] = round(statistics.median(startups), 1)
    out["compile"] = compile_s
    out["_meta"] = meta
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+",
                    help="packed jobs' slurm-out files (logs/pack_*.out); "
                         "each is parsed on its own")
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--seeds", type=int, default=None,
                    help="seeds per condition (default: read from the log)")
    ap.add_argument("--min-per-bucket", type=int, default=MIN_PER_BUCKET,
                    help="drop a bucket measured on fewer conditions")
    ap.add_argument("--min-seconds", type=float, default=MIN_SECONDS,
                    help="ignore conditions faster than this per seed")
    ap.add_argument("--compile", type=float, default=40.0,
                    help="per-bucket compile seconds (not measurable from one log)")
    ap.add_argument("--force", action="store_true",
                    help="write the calibration even if it is >2x faster than "
                         "the built-in table (the reason is recorded in it)")
    a = ap.parse_args(argv)

    from af3lis.pack import check_calib
    out = calibrate(a.logs, compile_s=a.compile, nseeds=a.seeds,
                    min_per_bucket=a.min_per_bucket, min_seconds=a.min_seconds)
    meta = out["_meta"]
    buckets = [k for k in out if not k.startswith("_")
               and k not in ("startup", "compile")]
    for k, why in meta["skipped_logs"].items():
        print(f"skipped {k}: {why}", file=sys.stderr)
    for b, why in meta["dropped_buckets"].items():
        print(f"bucket {b} dropped: {why} (pack keeps the built-in time)",
              file=sys.stderr)
    if not buckets:
        raise SystemExit(
            f"no bucket measured on >= {a.min_per_bucket} conditions across "
            f"{meta['logs']} log(s) with bucket lines -- is this a packed "
            f"--input_dir log? (a probe needs >= {a.min_per_bucket + 2} "
            f"conditions: the first one of each bucket carries the compile "
            f"and the last one has no end to measure)")
    print(f"{'bucket':>6} {'s/seed':>8} {'n':>4}")
    for b in sorted(buckets, key=int):
        print(f"{b:>6} {out[b]:>8.1f} {meta['n_per_bucket'][b]:>4}")
    if "startup" in out:
        print(f"startup (job start -> first condition): {out['startup']} s")
    print(f"seeds per condition: {', '.join(sorted(set(meta['seeds'].values())))}")

    reasons = check_calib({int(b): out[b] for b in buckets})
    if reasons and not a.force:
        rej = a.out + ".rejected"
        out["_rejected"] = reasons
        with open(rej, "w") as fh:
            json.dump(out, fh, indent=2)
        print("\nREFUSED -- implausibly fast against the built-in table:",
              file=sys.stderr)
        for r in reasons:
            print(f"  {r}", file=sys.stderr)
        print(f"Nothing written to {a.out}; the numbers and the reason are in "
              f"{rej}.\nIf a faster GPU genuinely explains it, rerun with "
              f"--force (the reason is then recorded in the calibration).",
              file=sys.stderr)
        raise SystemExit(2)
    if reasons:
        out["_forced"] = reasons
        print("\nWARNING: written with --force despite:", file=sys.stderr)
        for r in reasons:
            print(f"  {r}", file=sys.stderr)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {a.out}  ->  python pipeline.py --pack --outdir ... "
          f"--calib {a.out}")


if __name__ == "__main__":
    main()
