"""Calibrate per-bucket AF3 inference time from a packed-job log.

Port of Mau's af3_calibrate.py (bwHelix AF3 toolkit, Aug 2026), so
``af3lis.pack`` can size walltimes from measured numbers instead of the
built-in A100 defaults (of which only buckets 512/768 were measured).

Method: AF3's absl log prints "Calculating bucket size for input with N
tokens ... Got bucket size B" per condition; each condition's wall time is the
gap between successive bucket lines. Gaps are grouped by the PREVIOUS
condition's bucket; the per-bucket MEDIAN goes into calib.json. Startup = log
start -> first bucket line. The 'compile' term is not separable from a single
log; override --compile only if you have a measured value.

Usage:
    python -m af3lis.calibrate <packed_slurm_out> [--out calib.json]
    python -m af3lis.pack --outdir ... --calib calib.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics

TS = re.compile(r"(\d\d):(\d\d):(\d\d)\.(\d+)")  # absl log timestamp
BK = re.compile(r"Calculating bucket size for input with (\d+) tokens")
GOT = re.compile(r"Got bucket size (\d+)")


def _secs(m: re.Match) -> float:
    h, mm, s, frac = m.groups()
    return int(h) * 3600 + int(mm) * 60 + int(s) + float("0." + frac)


def calibrate(log_path: str, compile_s: float = 40.0) -> dict:
    """Parse one packed-job log -> {"<bucket>": median_s, ..., "startup": s,
    "compile": s}. Handles midnight wraparound in the HH:MM:SS timestamps."""
    events: list[tuple[str, float, int]] = []
    first_ts: float | None = None
    prev_raw: float | None = None
    day_offset = 0.0
    with open(log_path) as fh:
        for line in fh:
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
            if BK.search(line):
                events.append(("bucketline", ts, 0))
            mg = GOT.search(line)
            if mg and events and events[-1][0] == "bucketline":
                events[-1] = ("cond", events[-1][1], int(mg.group(1)))

    conds = [e for e in events if e[0] == "cond"]
    by_bucket: dict[int, list[float]] = {}
    for i in range(len(conds) - 1):
        _, ts, b = conds[i]
        _, ts2, _ = conds[i + 1]
        by_bucket.setdefault(b, []).append(ts2 - ts)

    out: dict = {}
    for b in sorted(by_bucket):
        out[str(b)] = round(statistics.median(by_bucket[b]), 1)
    if conds and first_ts is not None:
        out["startup"] = round(conds[0][1] - first_ts, 1)
    out["compile"] = compile_s
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", help="packed job's slurm-out (logs/pack_*.out)")
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--compile", type=float, default=40.0,
                    help="per-bucket compile seconds (not measurable from one log)")
    a = ap.parse_args()

    out = calibrate(a.log, compile_s=a.compile)
    buckets = [k for k in out if k not in ("startup", "compile")]
    if not buckets:
        raise SystemExit(
            "no paired bucket lines found -- is this a packed --input_dir log?")
    print(f"{'bucket':>6} {'median_s':>9}")
    for b in sorted(buckets, key=int):
        print(f"{b:>6} {out[b]:>9.1f}")
    if "startup" in out:
        print(f"startup (job start -> first condition): {out['startup']} s")
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {a.out}  ->  python -m af3lis.pack --calib {a.out} ...")


if __name__ == "__main__":
    main()
