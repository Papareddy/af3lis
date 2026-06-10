"""AF3 output I/O — single source of truth for all AF3 path globs + loaders.

Every other module in af3lis goes through this module rather than globbing raw AF3
paths. The AF3 ≥3.0.1 open-source CLI ("bio/alphafold/3.0.1" on Helix) emits, under
``<out_root>/<lname>/`` (``<lname>`` = AF3-lowercased ``name`` from the input JSON,
where ``<dir>`` may carry a timestamp suffix like ``<lname>_20260608_125545/`` when
AF3 auto-renames to avoid clobbering a prior run — see :func:`lname_of`):

    <lname>_model.cif                           top-level best-ranked copy
    <lname>_summary_confidences.json            top-level best-ranked copy
    <lname>_confidences.json                    top-level best-ranked full data
    <lname>_data.json                           MSA-augmented input (reproducibility artifact)
    ranking_scores.csv                          seed,sample,ranking_score (one row per sample;
                                                row index = AF3 flat sample index <N>)
    seed-<S>_sample-<M>/model.cif               per-sample structure (CIF)
    seed-<S>_sample-<M>/confidences.json        per-sample full data (pae, token_chain_ids,
                                                atom_plddts, atom_chain_ids)
    seed-<S>_sample-<M>/summary_confidences.json per-sample summary

``<N>`` is a flat 0-based sample index across the ``(seed × diffusion_sample)`` cross
product; the (seed, sample) provenance is the row index of ``ranking_scores.csv``,
and seeds/samples on disk live in ``seed-<S>_sample-<M>/`` subdirs (NOT in flat per-N
files at the top level — AF3's actual layout, not the flat-file layout assumed by
the original af3lis design draft).

This module exposes:
    - FoldResult dataclass: normalized per-sample bundle.
    - iter_jobs / iter_samples / lname_of: discovery.
    - read_ranking_scores: strict CSV parser.
    - load_sample / load_best: load one FoldResult.
    - chain_order_from_cif: row/col order of chain_pair_* matrices (label_asym_id order).
    - assert_data_json_exists: --infer-only precondition gate.

Design notes:
    - All paths returned are absolute and case-sensitive on disk; the only case-blindness
      is in :func:`iter_jobs`, which globs ``<out_root>/*/`` and filters dirs that look
      like AF3 outputs (i.e. contain at least one ``*_model_<N>.cif`` or a flat
      ``<lname>_model.cif``).
    - NaN handling: ``pae`` is loaded with ``np.array(..., dtype=float)`` then any NaN is
      replaced with 31.0 (matches the ``lis.py`` ``extract_pae`` convention so PEAK on
      our token axis matches lis.py's PEAK runtime).
    - ``chain_pair_iptm`` / ``chain_pair_pae_min`` are stored as ``np.ndarray`` with row
      ordering matching :func:`chain_order_from_cif` (i.e. label_asym_id order).
    - ``read_ranking_scores`` is **strict**: requires header == ``{'seed','sample',
      'ranking_score'}`` — no positional fallback. This is the deliberate fix for the
      original design draft's positional-fallback bug.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Regex / constants
# ---------------------------------------------------------------------------

# Match per-sample subdir "seed-<S>_sample-<M>".
_RE_SEED_DIR = re.compile(r"^seed-(?P<seed>\d+)_sample-(?P<sample>\d+)$")

# Strict ranking_scores.csv header set.
_RANKING_HEADER = {"seed", "sample", "ranking_score"}

# PAE NaN sentinel — matches lis.py extract_pae convention.
_PAE_NAN_FILL = 31.0


# ---------------------------------------------------------------------------
# FoldResult
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """One AF3 sample's data, normalized for downstream scoring.

    All arrays are numpy; all chain-id lists are plain ``list[str]``. The (seed, sample)
    pair is the authoritative provenance — ``flat_index`` is AF3's on-disk ``<N>``.

    ``chain_order`` is the row/col ordering of ``chain_pair_iptm`` and
    ``chain_pair_pae_min`` (== unique ``label_asym_id`` in CIF order).
    """

    name: str                                # job name == lowercased dir basename
    flat_index: int                          # AF3 <N> — 0..(n_seeds*n_samples-1)
    seed: int
    sample: int
    rank_key: str                            # f"{seed}_{sample}" — used as lis.py 'rank'
    cif_path: str
    full_data_path: str
    summary_path: str
    pae: np.ndarray                          # [num_tokens, num_tokens] float, NaN -> 31
    token_chain_ids: list[str]               # len == pae.shape[0]
    atom_plddts: np.ndarray                  # [num_atoms]
    atom_chain_ids: list[str]                # [num_atoms]
    ptm: float
    iptm: float
    chain_pair_iptm: np.ndarray              # [n_chains, n_chains], label_asym_id order
    chain_pair_pae_min: np.ndarray           # [n_chains, n_chains], label_asym_id order
    chain_ptm: np.ndarray                    # [n_chains]
    chain_iptm: np.ndarray                   # [n_chains]
    ranking_score: float
    has_clash: bool
    fraction_disordered: float
    chain_order: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def lname_of(job_dir: str) -> str:
    """Return the ``<lname>`` prefix used by top-level files inside ``job_dir``.

    AF3 lowercases the input JSON ``name`` and uses it as the file prefix for
    ``<lname>_model.cif`` etc. The DIRECTORY basename usually equals ``<lname>``,
    but AF3 may auto-append a timestamp suffix (e.g. ``ddrgk1__c53_20260608_125545/``)
    when it detects a pre-existing dir with the target name. In that case the
    files inside still use the original ``<lname>`` (no timestamp), so we derive
    it by probing actual filenames.

    Strategy:
      1. Look for the FIRST top-level file matching ``*_model.cif`` and strip the suffix.
      2. Fallback to ``*_summary_confidences.json``.
      3. Fallback to ``os.path.basename(job_dir)`` (works for the normal no-timestamp case).
    """
    job_dir = os.path.normpath(job_dir)
    if os.path.isdir(job_dir):
        for b in sorted(os.listdir(job_dir)):
            if b.endswith("_model.cif"):
                return b[: -len("_model.cif")]
        for b in sorted(os.listdir(job_dir)):
            if b.endswith("_summary_confidences.json"):
                return b[: -len("_summary_confidences.json")]
    return os.path.basename(job_dir)


def iter_jobs(out_root: str) -> Iterator[str]:
    """Yield each ``<out_root>/<lname>/`` directory that looks like an AF3 job output.

    A directory qualifies if it contains at least one ``seed-<S>_sample-<M>/`` subdir
    OR a top-level ``<lname>_model.cif`` (best-of-N copy — present even if the
    per-sample subdirs were cleaned up).

    Output paths are absolute and sorted (deterministic ordering for reproducibility).
    """
    if not os.path.isdir(out_root):
        return
    candidates: list[str] = []
    for entry in sorted(os.listdir(out_root)):
        d = os.path.join(out_root, entry)
        if not os.path.isdir(d):
            continue
        contents = os.listdir(d)
        has_seed_dir = any(
            _RE_SEED_DIR.match(b) and os.path.isdir(os.path.join(d, b))
            for b in contents
        )
        has_best = any(b.endswith("_model.cif") for b in contents)
        if has_seed_dir or has_best:
            candidates.append(os.path.abspath(d))
    yield from candidates


def iter_samples(job_dir: str) -> Iterator[tuple[int, str]]:
    """Yield ``(flat_index, cif_path)`` for every ``seed-<S>_sample-<M>/model.cif`` in ``job_dir``.

    ``flat_index`` is the row index in ``ranking_scores.csv`` (which AF3 writes in
    ranking-score-descending order — NOT in seed/sample lexicographic order).
    ``cif_path`` is absolute.

    Samples are yielded in flat_index order (== ranking_scores.csv row order).
    Subdirs missing ``model.cif`` are silently skipped (an incomplete AF3 run).
    """
    if not os.path.isdir(job_dir):
        return
    try:
        ranking = read_ranking_scores(job_dir)
    except FileNotFoundError:
        return
    for flat_index, row in ranking.iterrows():
        seed = int(row["seed"])
        sample = int(row["sample"])
        seed_dir = os.path.join(job_dir, f"seed-{seed}_sample-{sample}")
        cif = os.path.join(seed_dir, "model.cif")
        if os.path.exists(cif):
            yield int(flat_index), os.path.abspath(cif)


def iter_data_jobs(out_root: str) -> Iterator[str]:
    """Yield ``<out_root>/<lname>/`` directories that have ``<lname>_data.json``.

    Used by the ``--infer-only`` precondition gate: a directory qualifies the
    moment stage 1 (CPU MSA) has produced its augmented data JSON, even before
    any stage-2 model files exist. ``iter_jobs`` only sees dirs with model
    output and is therefore the wrong iterator for that gate.
    """
    if not os.path.isdir(out_root):
        return
    for entry in sorted(os.listdir(out_root)):
        d = os.path.join(out_root, entry)
        if not os.path.isdir(d):
            continue
        lname = entry
        if os.path.exists(os.path.join(d, f"{lname}_data.json")):
            yield os.path.abspath(d)


def find_pairs(out_root: str) -> list[str]:
    """List comprehension wrapper around :func:`iter_jobs` for ergonomics.

    Equivalent to ``list(iter_jobs(out_root))``. Kept as a named helper because
    ``collect.py`` and ``pipeline.py`` both want a materialized list.
    """
    return list(iter_jobs(out_root))


# ---------------------------------------------------------------------------
# Ranking CSV (seed/sample provenance)
# ---------------------------------------------------------------------------

def read_ranking_scores(job_dir: str) -> pd.DataFrame:
    """Parse ``ranking_scores.csv`` (or legacy ``<lname>_ranking_scores.csv``).

    Strict: requires header set ``{'seed','sample','ranking_score'}``. Extra columns are
    permitted (AF3 may add more in future releases) but the required three must all be
    present. No positional fallback — if the header is malformed we raise immediately.

    AF3 ≥3.0.1 writes the file as plain ``ranking_scores.csv`` (no lname prefix).
    We also accept ``<lname>_ranking_scores.csv`` as a fallback for older builds
    or for output dirs we may have renamed.

    Returns a DataFrame whose row index equals AF3's flat sample index ``<N>``; i.e.
    ``df.iloc[N]`` gives the seed/sample/ranking_score for sample ``<N>``.
    """
    csv_path = os.path.join(job_dir, "ranking_scores.csv")
    if not os.path.exists(csv_path):
        lname = lname_of(job_dir)
        alt = os.path.join(job_dir, f"{lname}_ranking_scores.csv")
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(
                f"missing ranking_scores.csv (also tried {lname}_ranking_scores.csv) in {job_dir}"
            )
    df = pd.read_csv(csv_path)
    missing = _RANKING_HEADER - set(df.columns)
    if missing:
        raise ValueError(
            f"ranking_scores.csv at {csv_path} missing required columns "
            f"{sorted(missing)}; got {list(df.columns)}"
        )
    # Cast seed/sample to int (AF3 emits ints but pandas may infer float if NaNs).
    df["seed"] = df["seed"].astype(int)
    df["sample"] = df["sample"].astype(int)
    df["ranking_score"] = df["ranking_score"].astype(float)
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CIF — chain order extraction
# ---------------------------------------------------------------------------

def chain_order_from_cif(cif_path: str) -> list[str]:
    """Return unique ``label_asym_id`` values in the order they appear in ``_atom_site``.

    This is the row/col ordering of ``chain_pair_iptm`` and ``chain_pair_pae_min`` in the
    AF3 summary JSON. Implemented as a tiny streaming parser over the ``_atom_site`` loop
    so we don't pull in an mmCIF library for this single column lookup.

    Uses an explicit OUTSIDE/IN_HEADER/IN_DATA state machine to robustly handle
    mmCIF files where the _atom_site loop is followed by other loops (the prior
    naive parser kept appending rows from subsequent loops as fake atoms).
    """
    header: list[str] = []
    state = "OUTSIDE"  # OUTSIDE | IN_HEADER | IN_DATA
    label_asym_idx: int | None = None
    seen: list[str] = []
    seen_set: set[str] = set()

    def _resolve_idx() -> int:
        """Find the column index for label_asym_id (or auth_asym_id fallback)."""
        for i, h in enumerate(header):
            if h.split(".", 1)[1] == "label_asym_id":
                return i
        for i, h in enumerate(header):
            if h.split(".", 1)[1] == "auth_asym_id":
                return i
        raise ValueError(
            f"no label_asym_id/auth_asym_id column in {cif_path}"
        )

    with open(cif_path) as fh:
        for ln in fh:
            s = ln.strip()
            if state == "OUTSIDE":
                if s.startswith("_atom_site."):
                    header.append(s)
                    state = "IN_HEADER"
                continue
            if state == "IN_HEADER":
                if s.startswith("_atom_site."):
                    header.append(s)
                    continue
                # Header for a different category or end of loop -> abort.
                if s.startswith("_") or s.startswith("loop_") or \
                   s.startswith("data_") or s.startswith("save_") or s == "#":
                    break
                if not s:
                    continue
                # First data row.
                label_asym_idx = _resolve_idx()
                state = "IN_DATA"
                parts = s.split()
                if label_asym_idx < len(parts):
                    ch = parts[label_asym_idx]
                    if ch not in seen_set:
                        seen_set.add(ch)
                        seen.append(ch)
                continue
            if state == "IN_DATA":
                if s.startswith("_") or s.startswith("loop_") or \
                   s.startswith("data_") or s.startswith("save_") or s == "#":
                    break
                if not s:
                    continue
                parts = s.split()
                if label_asym_idx is None or label_asym_idx >= len(parts):
                    continue
                ch = parts[label_asym_idx]
                if ch not in seen_set:
                    seen_set.add(ch)
                    seen.append(ch)
    return seen


# ---------------------------------------------------------------------------
# Per-file JSON loaders
# ---------------------------------------------------------------------------

def _load_pae(full_data: dict) -> np.ndarray:
    """Extract PAE from ``<lname>_full_data_<N>.json`` and replace NaN with 31."""
    pae = np.asarray(full_data["pae"], dtype=float)
    if pae.ndim != 2:
        raise ValueError(f"pae must be 2-D, got shape {pae.shape}")
    nan_mask = np.isnan(pae)
    if nan_mask.any():
        pae = pae.copy()
        pae[nan_mask] = _PAE_NAN_FILL
    return pae


def load_confidences(full_data_path: str) -> dict:
    """Load and lightly normalize ``<lname>_full_data_<N>.json``.

    Returns a dict with keys:
        ``pae``               : ``np.ndarray`` ``[num_tokens, num_tokens]``, NaN->31
        ``token_chain_ids``   : ``list[str]`` length ``num_tokens``
        ``atom_plddts``       : ``np.ndarray`` ``[num_atoms]`` (float)
        ``atom_chain_ids``    : ``list[str]`` length ``num_atoms`` (may be empty if absent)
        ``raw``               : the underlying dict (for forward-compat key access)
    """
    with open(full_data_path) as fh:
        d = json.load(fh)
    pae = _load_pae(d)
    if "token_chain_ids" not in d:
        raise ValueError(
            f"{full_data_path}: required field 'token_chain_ids' is missing "
            f"(AF3 >=3.0.1 always emits it)"
        )
    token_chain_ids = list(d["token_chain_ids"])
    if len(token_chain_ids) != pae.shape[0]:
        raise ValueError(
            f"{full_data_path}: token_chain_ids length ({len(token_chain_ids)}) "
            f"!= pae.shape[0] ({pae.shape[0]})"
        )
    if "atom_plddts" not in d:
        raise ValueError(
            f"{full_data_path}: required field 'atom_plddts' is missing"
        )
    atom_plddts = np.asarray(d["atom_plddts"], dtype=float)
    if "atom_chain_ids" not in d:
        raise ValueError(
            f"{full_data_path}: required field 'atom_chain_ids' is missing"
        )
    atom_chain_ids = list(d["atom_chain_ids"])
    return {
        "pae": pae,
        "token_chain_ids": token_chain_ids,
        "atom_plddts": atom_plddts,
        "atom_chain_ids": atom_chain_ids,
        "raw": d,
    }


def load_summary_confidences(summary_path: str) -> dict:
    """Load and normalize ``<lname>_summary_confidences_<N>.json``.

    Returns a dict with keys:
        ``ptm``                  : float
        ``iptm``                 : float (NaN-safe; monomers have None -> NaN)
        ``chain_pair_iptm``      : ``np.ndarray`` ``[n_chains, n_chains]`` (label_asym_id order)
        ``chain_pair_pae_min``   : ``np.ndarray`` ``[n_chains, n_chains]`` (label_asym_id order)
        ``chain_ptm``            : ``np.ndarray`` ``[n_chains]``
        ``chain_iptm``           : ``np.ndarray`` ``[n_chains]``
        ``ranking_score``        : float
        ``has_clash``            : bool
        ``fraction_disordered``  : float
        ``raw``                  : the underlying dict
    """
    with open(summary_path) as fh:
        d = json.load(fh)

    def _f(key: str, default: float = float("nan")) -> float:
        v = d.get(key, default)
        return float("nan") if v is None else float(v)

    def _arr(key: str, dtype=float) -> np.ndarray:
        v = d.get(key)
        if v is None:
            return np.asarray([], dtype=dtype)
        return np.asarray(v, dtype=dtype)

    def _arr2d(key: str) -> np.ndarray:
        """2-D chain-pair matrix. Empty/monomer -> shape (0,0)."""
        a = _arr(key)
        if a.size == 0:
            return np.zeros((0, 0), dtype=float)
        if a.ndim == 1:
            # Some AF3 builds flatten nested-list-of-list when n_chains==1.
            n = int(round(np.sqrt(a.size)))
            if n * n == a.size:
                return a.reshape(n, n)
        return a

    has_clash_raw = d.get("has_clash", False)
    # AF3 emits has_clash as bool or 0.0/1.0; bool() handles both correctly
    # without the truthiness-eating int() round-trip.
    has_clash = bool(has_clash_raw)

    return {
        "ptm": _f("ptm"),
        "iptm": _f("iptm"),
        "chain_pair_iptm": _arr2d("chain_pair_iptm"),
        "chain_pair_pae_min": _arr2d("chain_pair_pae_min"),
        "chain_ptm": _arr("chain_ptm"),
        "chain_iptm": _arr("chain_iptm"),
        "ranking_score": _f("ranking_score"),
        "has_clash": has_clash,
        "fraction_disordered": _f("fraction_disordered", 0.0),
        "raw": d,
    }


# ---------------------------------------------------------------------------
# Composite loaders
# ---------------------------------------------------------------------------

def iter_seed_samples(job_dir: str) -> Iterator[tuple[int, int, int]]:
    """Yield ``(flat_index, seed, sample)`` triples for every sample in ``job_dir``.

    Walks :func:`iter_samples` and looks up the (seed, sample) for each flat index in
    ``<lname>_ranking_scores.csv``. Sorted by flat index. Provided as a convenience for
    callers (``collect.py``) that want provenance without loading the full FoldResult.
    """
    ranking = read_ranking_scores(job_dir)
    for flat_index, _cif in iter_samples(job_dir):
        if flat_index >= len(ranking):
            raise IndexError(
                f"flat_index {flat_index} out of range for ranking_scores.csv "
                f"of length {len(ranking)} in {job_dir}"
            )
        row = ranking.iloc[flat_index]
        yield flat_index, int(row["seed"]), int(row["sample"])


def load_sample(job_dir: str, flat_index: int,
                ranking: pd.DataFrame | None = None) -> FoldResult:
    """Load one sample by flat index ``N``.

    Reads files from ``seed-<S>_sample-<M>/`` where ``(S, M)`` are looked up in
    ``ranking_scores.csv`` (row ``N``):
        seed-<S>_sample-<M>/model.cif
        seed-<S>_sample-<M>/confidences.json
        seed-<S>_sample-<M>/summary_confidences.json

    ``ranking`` may be passed by the caller to hoist the per-job CSV read out
    of a tight loop (e.g. ``collect.peak_table`` calls ``load_sample`` once
    per sample; reading the CSV 25× per job is wasted I/O).
    """
    # Absolute job_dir up-front: paths emitted on FoldResult must not depend
    # on caller cwd (cron/SLURM contexts inherit unpredictable cwd).
    job_dir = os.path.abspath(job_dir)
    lname = lname_of(job_dir)
    if ranking is None:
        ranking = read_ranking_scores(job_dir)
    if flat_index >= len(ranking):
        raise IndexError(
            f"flat_index {flat_index} out of range for ranking_scores.csv "
            f"of length {len(ranking)} in {job_dir}"
        )
    row = ranking.iloc[flat_index]
    seed = int(row["seed"])
    sample = int(row["sample"])

    seed_dir = os.path.join(job_dir, f"seed-{seed}_sample-{sample}")
    cif_path = os.path.join(seed_dir, "model.cif")
    full_data_path = os.path.join(seed_dir, "confidences.json")
    summary_path = os.path.join(seed_dir, "summary_confidences.json")
    for p in (cif_path, full_data_path, summary_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"AF3 sample file missing: {p}")

    conf = load_confidences(full_data_path)
    summary = load_summary_confidences(summary_path)
    chain_order = chain_order_from_cif(cif_path)

    return FoldResult(
        name=lname,
        flat_index=flat_index,
        seed=seed,
        sample=sample,
        rank_key=f"{seed}_{sample}",
        cif_path=os.path.abspath(cif_path),
        full_data_path=os.path.abspath(full_data_path),
        summary_path=os.path.abspath(summary_path),
        pae=conf["pae"],
        token_chain_ids=conf["token_chain_ids"],
        atom_plddts=conf["atom_plddts"],
        atom_chain_ids=conf["atom_chain_ids"],
        ptm=summary["ptm"],
        iptm=summary["iptm"],
        chain_pair_iptm=summary["chain_pair_iptm"],
        chain_pair_pae_min=summary["chain_pair_pae_min"],
        chain_ptm=summary["chain_ptm"],
        chain_iptm=summary["chain_iptm"],
        ranking_score=summary["ranking_score"],
        has_clash=summary["has_clash"],
        fraction_disordered=summary["fraction_disordered"],
        chain_order=chain_order,
    )


def load_best(job_dir: str) -> FoldResult:
    """Load the top-ranked sample for ``job_dir``.

    AF3 writes ``<lname>_model.cif`` as a copy of the highest-``ranking_score`` per-sample
    CIF. We mirror that selection by argmax'ing ``ranking_scores.csv['ranking_score']``
    and routing through :func:`load_sample` — this guarantees the (seed, sample) reported
    on the returned FoldResult is authoritative even when the top-level file is a copy.
    """
    ranking = read_ranking_scores(job_dir)
    if len(ranking) == 0:
        raise ValueError(f"empty ranking_scores.csv in {job_dir}")
    # Use np.argmax (positional) rather than idxmax (label-based) for
    # robustness against callers passing non-default-indexed DataFrames.
    best_idx = int(np.argmax(ranking["ranking_score"].to_numpy()))
    return load_sample(job_dir, best_idx)


# ---------------------------------------------------------------------------
# Preconditions / gates
# ---------------------------------------------------------------------------

def assert_data_json_exists(job_dir: str) -> None:
    """Raise ``FileNotFoundError`` if ``<lname>_data.json`` is missing in ``job_dir``.

    Used by ``pipeline.cmd_submit --infer-only`` as a client-side precondition gate so we
    fail fast (and cheaply, before any GPU sbatch) rather than burning A100 time on jobs
    whose MSA stage never completed.
    """
    lname = lname_of(job_dir)
    data_path = os.path.join(job_dir, f"{lname}_data.json")
    if not os.path.exists(data_path):
        # Best-effort fallback for AF3 builds that sanitize dir names beyond simple
        # lowercasing (e.g. '.' -> '_').
        hits = glob.glob(os.path.join(job_dir, "*_data.json"))
        if not hits:
            raise FileNotFoundError(
                f"AF3 MSA stage not complete for {job_dir!r}: expected {data_path}"
            )
