"""collect_all -- one call to compute EVERY interface metric for an AF3 out dir.

AF3 mirror of `boltzlis.collect`. Runs the vendored AFM-LIS `lis.py` with
`--platform alphafold3` (iLIS/LIS/cLIS/LIA/ipSAE/actifpTM/ipTM, parallel `-w`),
adds PEAK ( = 1 - min_interchain_PAE/30 ) per chain pair computed on the AF3
token axis, then aggregates the per-model rows to a ranked per-pair table.

Two aggregation modes:
  * `per_seed` (DEFAULT) -- seed-mean/seed-max first, then mean/max across seeds.
        Emits 4 columns per metric (`_mean_mean`, `_mean_max`, `_max_mean`,
        `_max_max`) plus `n_seeds` and `n_samples_per_seed`. Honest for multi-seed
        AF3 runs (correlated diffusion samples within a seed).
  * `flat` -- boltzlis-style flat mean/max over ALL (seed, sample). Emits
        `_mean`/`_max` per metric plus `n_models`. Byte-identical TSV schema to
        boltzlis (single-seed runs are equivalent).

CLI:  python -m af3lis.collect <out_dir> -o metrics.tsv [-w 8] [--rank iLIS_max]
                                                       [--agg per_seed|flat]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional

import numpy as np
import pandas as pd

from . import af3_io
from . import structure as st

HERE = os.path.dirname(os.path.abspath(__file__))
LIS_PY = os.path.join(HERE, "lis.py")

# Metrics aggregated across (seed, sample). Order matches boltzlis.collect.AGG.
AGG = [
    "iLIS", "LIS", "cLIS", "iLIA", "LIA", "cLIA",
    "ipSAE", "actifpTM", "ipTM", "pTM", "PEAK",
    "pLDDT_i", "pLDDT_j",
]

# Join keys for merging PEAK back into the lis.py per-model table.
KEYS = ["name", "chain_i", "chain_j"]


# ---------------------------------------------------------------------------
# PEAK table -- token-axis, per AF3 sample
# ---------------------------------------------------------------------------

def peak_table(out_dir: str) -> pd.DataFrame:
    """PEAK per (name, rank, chain_i, chain_j) from AF3 PAE + token_chain_ids.

    Walks `<out_dir>/<lname>/` job directories via `af3_io.iter_jobs`, then each
    flat `<lname>_model_<N>.cif` sample via `af3_io.iter_samples`. PEAK is
    computed on the AF3 token axis (length == pae.shape[0]) so it is correct
    for any ligand/PTM token-vs-residue mismatch.

    Returns
    -------
    DataFrame with columns ['name', 'rank', 'chain_i', 'chain_j', 'PEAK',
    'seed', 'sample', 'flat_index']. `rank` is the string flat-sample-index
    AF3 emits on disk (``str(N)``) — matches lis.py's standard-AF3 ``rank``
    column for the join in :func:`collect_all`. Seed/sample are also surfaced
    so callers can attach provenance without re-reading ranking_scores.csv.
    Monomer jobs (single chain) contribute zero rows.
    """
    rows: list[dict] = []
    for job_dir in af3_io.iter_jobs(out_dir):
        try:
            ranking = af3_io.read_ranking_scores(job_dir)
        except (FileNotFoundError, ValueError) as e:
            sys.stderr.write(
                f"[collect] skip {job_dir}: ranking CSV unreadable ({e})\n"
            )
            continue
        for flat_index, _cif_path in af3_io.iter_samples(job_dir):
            try:
                # Hoist the per-job ranking read out of the inner loop — saves
                # ~25 pandas.read_csv calls per job on a typical 5-seed run.
                fr = af3_io.load_sample(job_dir, flat_index, ranking=ranking)
            except Exception as e:  # noqa: BLE001 -- log & skip bad samples
                sys.stderr.write(
                    f"[collect] skip {job_dir}#{flat_index}: load failed ({e})\n"
                )
                continue
            try:
                pairs = st.peak_per_chainpair(fr.pae, fr.token_chain_ids)
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(
                    f"[collect] skip {job_dir}#{flat_index}: PEAK failed ({e})\n"
                )
                continue
            for (ci, cj), peak in pairs.items():
                # peak_per_chainpair emits both (ca,cb) and (cb,ca); keep
                # only one direction here so collect_all's lexicographic
                # normalization doesn't accidentally produce duplicate rows.
                if ci >= cj:
                    continue
                rows.append(
                    # `name` MUST match lis.py's `name` column = dir basename
                    # (lis.py uses os.path.basename(structure_file_parent), which
                    # for AF3 is the job dir name — possibly with a timestamp
                    # suffix that AF3 auto-appends to avoid clobbering an
                    # existing run). DO NOT use lname_of() here — it strips the
                    # suffix and breaks the (name, rank) join.
                    # `rank` matches lis.py's AF3 rank column = f"{seed}_{sample}"
                    # (NOT str(flat_index) — earlier draft had this wrong).
                    # Seed/sample carried separately as authoritative provenance.
                    dict(name=os.path.basename(job_dir), rank=fr.rank_key,
                         chain_i=ci, chain_j=cj, PEAK=float(peak),
                         seed=fr.seed, sample=fr.sample,
                         flat_index=fr.flat_index)
                )
    if not rows:
        return pd.DataFrame(columns=["name", "rank", "chain_i", "chain_j",
                                     "PEAK", "seed", "sample", "flat_index"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# lis.py subprocess driver -- AF3 mode
# ---------------------------------------------------------------------------

def run_lis(out_dir: str,
            csv_path: str,
            workers: int = 8,
            python_exe: Optional[str] = None,
            lis_py: Optional[str] = None) -> pd.DataFrame:
    """Invoke the vendored `lis.py` as a subprocess in AF3 mode.

    Forces `--platform alphafold3` to skip autodetect -- cross-mounted scratch
    dirs can carry stale `*.npz` files from Boltz runs and would otherwise
    confuse the layout sniff.

    Returns the per-model CSV as a DataFrame.
    """
    python_exe = python_exe or sys.executable
    lis_py = lis_py or LIS_PY
    out_d = os.path.dirname(os.path.abspath(csv_path)) or "."
    os.makedirs(out_d, exist_ok=True)
    # Remove a stale csv up-front: lis.py's --skip-existing default would
    # otherwise silently keep rows from a previous run, mixing them with
    # fresh AF3 output and producing silently-corrupt aggregations.
    if os.path.exists(csv_path):
        os.remove(csv_path)
    cmd = [
        python_exe, lis_py, out_dir,
        "-w", str(workers),
        "-o", os.path.basename(csv_path),
        "-d", out_d,
        "--platform", "alphafold3",
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0 or not os.path.exists(csv_path):
        # lis.py exits non-zero (and writes no CSV) when out_dir has no AF3
        # outputs. Don't propagate — emit an empty DataFrame with the right
        # column shape and let collect_all's empty-input short-circuit fire.
        sys.stderr.write(
            f"[collect] lis.py produced no output (rc={rc}); treating as "
            f"empty input.\n"
        )
        return pd.DataFrame(columns=["name", "rank", "chain_i", "chain_j"])
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _flat_columns(metrics: list[str]) -> list[str]:
    """boltzlis-identical schema: <metric>_mean, <metric>_max per metric."""
    out: list[str] = []
    for m in metrics:
        out += [f"{m}_mean", f"{m}_max"]
    return out


def _per_seed_columns(metrics: list[str]) -> list[str]:
    """4-column-per-metric schema: <m>_mean_mean, _mean_max, _max_mean, _max_max."""
    out: list[str] = []
    for m in metrics:
        out += [f"{m}_mean_mean", f"{m}_mean_max",
                f"{m}_max_mean", f"{m}_max_max"]
    return out


def _attach_seed_sample(lis: pd.DataFrame, peak: pd.DataFrame) -> pd.DataFrame:
    """Attach (seed, sample, flat_index) to `lis` from the peak_table side.

    The peak_table is the authoritative source for (seed, sample) provenance
    because it reads ranking_scores.csv via af3_io. lis.py's per-row 'rank'
    is the flat sample index (str), not "{seed}_{sample}" — so we can't
    derive provenance by parsing it.

    Raises ValueError if the join would leave any lis row without
    seed/sample (indicates a real schema mismatch upstream).
    """
    if peak.empty:
        # Cannot derive provenance without peak rows; collect_all will short-
        # circuit empty-output paths upstream of aggregation.
        return lis.assign(seed=pd.Series(dtype=int),
                          sample=pd.Series(dtype=int),
                          flat_index=pd.Series(dtype=int))
    # Per-job (name, rank) -> (seed, sample, flat_index) — unique on those keys.
    prov = (peak[["name", "rank", "seed", "sample", "flat_index"]]
            .drop_duplicates(subset=["name", "rank"]))
    # lis['rank'] arrives from CSV as int; coerce both sides to str for the join.
    out = lis.assign(rank=lis["rank"].astype(str)).merge(
        prov.assign(rank=prov["rank"].astype(str)),
        on=["name", "rank"], how="left"
    )
    unmatched = int(out["seed"].isna().sum())
    if unmatched:
        sys.stderr.write(
            f"[collect] WARNING: {unmatched}/{len(out)} lis rows lack "
            f"seed/sample provenance after the (name,rank) join — check that "
            f"lis.py's `rank` column equals str(flat_index) for AF3 outputs.\n"
        )
    return out


def _aggregate_flat(lis: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """boltzlis-style flat mean/max over all (seed, sample)."""
    g = lis.groupby(KEYS, sort=False)
    agg = g[metrics].agg(["mean", "max"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_models"] = g.size()
    return agg.reset_index()


def _aggregate_per_seed(lis: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per-seed-then-cross aggregation.

    Step 1: for each (name, chain_i, chain_j, seed) compute mean and max over
            diffusion samples -> 2 stats per metric per seed.
    Step 2: for each (name, chain_i, chain_j) compute mean and max of each
            of those stats across seeds -> 4 stats per metric per pair.

    Emits `n_seeds` and `n_samples_per_seed` instead of `n_models`. If sample
    count varies across seeds (shouldn't, but defensive), `n_samples_per_seed`
    is the modal count.

    Requires `seed` / `sample` columns to be pre-attached by
    :func:`_attach_seed_sample` (sourced from ranking_scores.csv via the peak
    table). The old `_parse_seed_sample(rank)` fallback was removed — it
    silently coerced standard-AF3 ranks (bare flat index) into (-1, -1),
    collapsing all samples into a phantom single seed.
    """
    if "seed" not in lis.columns or "sample" not in lis.columns:
        raise RuntimeError(
            "_aggregate_per_seed requires 'seed' and 'sample' columns; call "
            "_attach_seed_sample(lis, peak) first."
        )

    # Step 1: collapse samples within each seed.
    seed_lvl = (lis.groupby(KEYS + ["seed"], sort=False)[metrics]
                   .agg(["mean", "max"]))
    seed_lvl.columns = [f"{m}_{s}" for m, s in seed_lvl.columns]
    seed_lvl = seed_lvl.reset_index()

    # Step 2: collapse across seeds. For each stat from step 1, take mean+max.
    inner_cols = [c for c in seed_lvl.columns
                  if c not in KEYS + ["seed"]]
    g2 = seed_lvl.groupby(KEYS, sort=False)
    agg = g2[inner_cols].agg(["mean", "max"])
    # Flatten "<m>_<s1>" + "<s2>" -> "<m>_<s1>_<s2>"
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.reset_index()

    # n_seeds and n_samples_per_seed
    n_seeds = g2.size().rename("n_seeds")

    samples_per_seed = (lis.groupby(KEYS + ["seed"], sort=False)
                           .size()
                           .reset_index(name="_n"))
    # Modal sample count per pair (across seeds).
    nsps = (samples_per_seed.groupby(KEYS, sort=False)["_n"]
                            .agg(lambda s: int(s.mode().iat[0]) if not s.mode().empty else int(s.iloc[0]))
                            .rename("n_samples_per_seed"))

    agg = agg.merge(n_seeds.reset_index(), on=KEYS, how="left")
    agg = agg.merge(nsps.reset_index(), on=KEYS, how="left")
    return agg


def _translate_rank_by(rank_by: str, agg_mode: str, columns: list[str]) -> str:
    """Translate boltzlis vocabulary ('iLIS_max') to per_seed ('iLIS_max_max').

    Only auto-translates the well-defined boltzlis<->per_seed vocabulary;
    anything else raises ValueError. Silent fallback (which used to coerce
    typos to PEAK_max_max) silently changes the sort order without any
    user signal — refuse it.
    """
    if rank_by in columns:
        return rank_by
    if agg_mode == "per_seed":
        # 'iLIS_max' -> 'iLIS_max_max'; 'iLIS_mean' -> 'iLIS_mean_mean'.
        if rank_by.endswith("_max") and f"{rank_by}_max" in columns:
            return f"{rank_by}_max"
        if rank_by.endswith("_mean") and f"{rank_by}_mean" in columns:
            return f"{rank_by}_mean"
    available = [c for c in columns if c not in KEYS]
    raise ValueError(
        f"unknown rank_by {rank_by!r}; available columns: {available}"
    )


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------

def collect_all(out_dir: str,
                out_tsv: str,
                workers: int = 8,
                rank_by: str = "iLIS_max",
                python_exe: Optional[str] = None,
                lis_py: Optional[str] = None,
                agg_mode: str = "per_seed") -> pd.DataFrame:
    """Full pipeline: lis.py -> merge PEAK -> aggregate -> write TSV.

    Parameters
    ----------
    out_dir : AF3 output root (contains `<lname>/<lname>_model_<N>.cif` etc).
    out_tsv : output TSV path. Per-model CSV is written alongside as
              `<out_tsv>.permodel.csv` (carries residue-level LIR/cLIR).
    workers : parallel workers for lis.py.
    rank_by : sort column. boltzlis names ('iLIS_max', 'PEAK_max') are
              auto-translated to per_seed equivalents.
    agg_mode : 'per_seed' (default) or 'flat'.
    """
    if agg_mode not in ("per_seed", "flat"):
        raise ValueError(
            f"agg_mode must be 'per_seed' or 'flat', got {agg_mode!r}"
        )

    lis_csv = os.path.splitext(out_tsv)[0] + ".permodel.csv"
    lis = run_lis(out_dir, lis_csv, workers=workers,
                  python_exe=python_exe, lis_py=lis_py)
    if "rank" not in lis.columns:
        # lis.py CSV_HEADER uses 'rank' as the per-row key; refuse to proceed
        # if a future lis.py version renames it.
        raise RuntimeError(
            "lis.py per-model CSV is missing the 'rank' column; "
            f"got columns={list(lis.columns)}"
        )

    # ---- normalize chain ordering on the lis side ----
    # peak_per_chainpair preserves first-appearance order; lis.py emits in
    # token_chain_ids insertion order with i<j. To make the join robust to
    # any per-row order disagreement (which can happen if AF3 ever emits
    # chains out of label_asym_id order), sort each row's (chain_i, chain_j)
    # lexicographically on BOTH sides.
    if not lis.empty and {"chain_i", "chain_j"}.issubset(lis.columns):
        ij = np.sort(lis[["chain_i", "chain_j"]].astype(str).to_numpy(), axis=1)
        lis["chain_i"] = ij[:, 0]
        lis["chain_j"] = ij[:, 1]

    # ---- merge PEAK per (name, rank, chain_i, chain_j) ----
    peak = peak_table(out_dir)
    if not peak.empty:
        peak_norm = peak.copy()
        ij = np.sort(peak_norm[["chain_i", "chain_j"]].astype(str).to_numpy(), axis=1)
        peak_norm["chain_i"] = ij[:, 0]
        peak_norm["chain_j"] = ij[:, 1]
        peak_norm["rank"] = peak_norm["rank"].astype(str)
        lis = lis.assign(rank=lis["rank"].astype(str)).merge(
            peak_norm[["name", "rank", "chain_i", "chain_j", "PEAK"]],
            on=["name", "rank", "chain_i", "chain_j"],
            how="left",
        )
    else:
        lis["PEAK"] = np.nan

    matched = int(lis["PEAK"].notna().sum())
    total = len(lis)
    match_rate = (matched / total) if total else 0.0
    sys.stderr.write(
        f"[collect] peak rows={len(peak)}, PEAK matched {matched}/{total} "
        f"lis rows ({match_rate:.1%})\n"
    )
    if total and match_rate < 0.90:
        sys.stderr.write(
            "[collect] WARNING: PEAK match rate < 90% -- check chain_id_source "
            "alignment between lis.py and structure.parse_structure "
            "(should both be label_asym_id for AF3).\n"
        )

    # Attach seed/sample provenance from peak_table (authoritative via
    # ranking_scores.csv). Done BEFORE the self-pair drop and BEFORE writing
    # the permodel CSV so the persisted per-row table carries provenance.
    lis = _attach_seed_sample(lis, peak)

    # Persist the full per-model CSV (with PEAK + seed/sample) — this is the
    # consumer-facing artifact. Write it BEFORE the chain_i!=chain_j filter
    # so monomer self-pairs are preserved for diagnostic use.
    out_d = os.path.dirname(os.path.abspath(out_tsv)) or "."
    os.makedirs(out_d, exist_ok=True)
    lis.to_csv(lis_csv, index=False)

    # Drop self-pairs (monomers / diagonal) before aggregation.
    lis = lis[lis["chain_i"] != lis["chain_j"]].copy()

    have = [c for c in AGG if c in lis.columns]

    # ---- aggregate ----
    if agg_mode == "flat":
        agg = _aggregate_flat(lis, have)
        expected = KEYS + _flat_columns(have) + ["n_models"]
    else:
        agg = _aggregate_per_seed(lis, have)
        expected = KEYS + _per_seed_columns(have) + ["n_seeds", "n_samples_per_seed"]

    # ---- empty-input short-circuit ----
    # Empty out_dir (or all jobs failed to load) -> emit a header-only TSV
    # with the canonical schema so downstream tools don't choke on the
    # column-drift guard.
    if agg.empty:
        empty = pd.DataFrame(columns=expected)
        empty.to_csv(out_tsv, sep="\t", index=False)
        return empty

    # ---- column-drift guard ----
    missing = [c for c in expected if c not in agg.columns]
    if missing:
        raise RuntimeError(
            f"collect_all: expected columns missing from aggregated table: "
            f"{missing}. Got: {list(agg.columns)}"
        )

    # Reorder to canonical layout (defensive against groupby column ordering).
    agg = agg[expected]

    # ---- sort + write ----
    rb = _translate_rank_by(rank_by, agg_mode, list(agg.columns))
    if rb != rank_by:
        sys.stderr.write(f"[collect] rank_by {rank_by!r} -> {rb!r}\n")
    agg = agg.sort_values(rb, ascending=False, kind="mergesort")
    agg.to_csv(out_tsv, sep="\t", index=False)
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute all interface metrics for an AF3 out dir "
                    "(iLIS/LIS/cLIS/LIA/ipSAE/actifpTM/ipTM/pTM/PEAK + pLDDT).",
    )
    ap.add_argument("out_dir",
                    help="AF3 output root (contains <lname>/<lname>_model_<N>.cif ...)")
    ap.add_argument("-o", "--output", default="metrics.tsv",
                    help="Output TSV path (default: metrics.tsv).")
    ap.add_argument("-w", "--workers", type=int, default=8,
                    help="Parallel workers for lis.py (default: 8).")
    ap.add_argument("--python", default=None,
                    help="Python exe with numpy+scipy for lis.py (default: sys.executable).")
    ap.add_argument("--lis", default=LIS_PY,
                    help="Path to vendored lis.py (default: af3lis/lis.py).")
    ap.add_argument("--rank", default="iLIS_max",
                    help="Sort column. boltzlis names auto-translated in per_seed mode "
                         "(default: iLIS_max -> iLIS_max_max).")
    ap.add_argument("--agg", choices=("per_seed", "flat"), default="per_seed",
                    help="Aggregation mode. 'per_seed' (default) emits 4 columns "
                         "per metric; 'flat' is byte-identical to boltzlis schema.")
    a = ap.parse_args()

    agg = collect_all(
        a.out_dir, a.output,
        workers=a.workers,
        rank_by=a.rank,
        python_exe=a.python,
        lis_py=a.lis,
        agg_mode=a.agg,
    )

    print(f"wrote {a.output}  ({len(agg)} pairs)")
    # Echo a short ranked preview -- prefer the per_seed flagship columns,
    # fall back to flat names when agg='flat'.
    preview_cols = [c for c in (
        "name",
        "n_seeds", "n_samples_per_seed", "n_models",
        "iLIS_max_max", "iLIS_max",
        "PEAK_max_max", "PEAK_max",
        "actifpTM_max_max", "actifpTM_max",
        "ipSAE_max_max", "ipSAE_max",
    ) if c in agg.columns]
    if preview_cols:
        print(agg[preview_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
