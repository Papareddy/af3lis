"""Drive the vendored Dunbrack ``ipsae.py`` over an AF3 out dir.

Adds the ipsae.py metric family to the af3lis table (requested: OUR metrics
AND Mau's). ipsae.py (Dunbrack v3, 2025-04-06, shipped in Mau's bwHelix AF3
toolkit; https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1) computes
per chain pair per model:

  * ipSAE (d0res / d0chn / d0dom variants)   -- PAE-based interface score
  * ipTM_d0chn                               -- PAE-derived ipTM rescaled
  * pDockQ / pDockQ2                         -- Bryant 2022 / Zhu 2023
  * LIS                                      -- Kim 2024 (their implementation)

Column mapping into the af3lis per-model table (avoids clashes with the
lis.py-derived columns of the same name):

  ipSAE       -> ipSAE_d0res     (our lis.py 'ipSAE' column is kept as-is)
  LIS         -> LIS_ipsae       (our lis.py 'LIS' column is kept as-is)
  everything else keeps ipsae.py's name (ipSAE_d0chn, ipSAE_d0dom,
  ipTM_d0chn, pDockQ, pDockQ2)

Per pair we keep ipsae.py's ``max`` row (its recommended pair-level value,
max over the two asym directions) -- consistent with our symmetric
(chain_i < chain_j) schema.

ipsae.py is argv-driven and writes its outputs next to the structure file, so
each sample is scored in a scratch dir with a symlinked CIF -- AF3 output
dirs stay clean.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd

from . import af3_io

HERE = os.path.dirname(os.path.abspath(__file__))
IPSAE_PY = os.path.join(HERE, "ipsae.py")

# ipsae.py txt column -> af3lis per-model column
COLMAP = {
    "ipSAE": "ipSAE_d0res",
    "ipSAE_d0chn": "ipSAE_d0chn",
    "ipSAE_d0dom": "ipSAE_d0dom",
    "ipTM_d0chn": "ipTM_d0chn",
    "pDockQ": "pDockQ",
    "pDockQ2": "pDockQ2",
    "LIS": "LIS_ipsae",
}
METRICS = list(COLMAP.values())
_EMPTY_COLS = ["name", "rank", "chain_i", "chain_j"] + METRICS


def _cutoff_str(v: float) -> str:
    """ipsae.py's zero-padded cutoff token used in its output filenames."""
    s = str(int(v))
    return "0" + s if v < 10 else s


def parse_ipsae_txt(txt_path: str) -> list[dict]:
    """Parse ipsae.py's main ``.txt`` -- return one dict per ``max`` row."""
    rows: list[dict] = []
    header: list[str] | None = None
    with open(txt_path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Chn1":
                header = parts
                continue
            if header is None or len(parts) != len(header):
                continue
            rec = dict(zip(header, parts))
            if rec.get("Type") != "max":
                continue
            row: dict = {"chain_i": rec["Chn1"], "chain_j": rec["Chn2"]}
            try:
                for src, dst in COLMAP.items():
                    row[dst] = float(rec[src])
            except (KeyError, ValueError):
                continue
            rows.append(row)
    return rows


def _score_sample(job_name: str,
                  rank_key: str,
                  cif_path: str,
                  conf_path: str,
                  pae_cutoff: float,
                  dist_cutoff: float,
                  python_exe: str) -> list[dict]:
    """Run ipsae.py for one (cif, confidences) sample in a scratch dir."""
    tmpd = tempfile.mkdtemp(prefix="ipsae_")
    try:
        # Unique stem so parallel workers can't collide even if tempdirs merge.
        stem = f"{job_name}__{rank_key}"
        cif_link = os.path.join(tmpd, stem + ".cif")
        os.symlink(os.path.abspath(cif_path), cif_link)
        cmd = [python_exe, IPSAE_PY, os.path.abspath(conf_path), cif_link,
               str(pae_cutoff), str(dist_cutoff)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        txt = os.path.join(
            tmpd,
            f"{stem}_{_cutoff_str(pae_cutoff)}_{_cutoff_str(dist_cutoff)}.txt")
        if res.returncode != 0 or not os.path.exists(txt):
            sys.stderr.write(
                f"[ipsae] skip {job_name}#{rank_key}: rc={res.returncode} "
                f"{(res.stderr or '').strip().splitlines()[-1:] or ''}\n")
            return []
        rows = parse_ipsae_txt(txt)
        for r in rows:
            r["name"] = job_name
            r["rank"] = rank_key
        return rows
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def run_ipsae(out_dir: str,
              workers: int = 8,
              pae_cutoff: float = 10.0,
              dist_cutoff: float = 10.0,
              python_exe: Optional[str] = None) -> pd.DataFrame:
    """Score every sample of every job under ``out_dir``.

    Returns a per-model DataFrame keyed (name, rank, chain_i, chain_j) with
    the ipsae.py metric columns -- ``rank`` is ``f"{seed}_{sample}"``, the
    same key lis.py emits for AF3, so it joins directly in collect_all.
    """
    python_exe = python_exe or sys.executable
    tasks: list[tuple[str, str, str, str]] = []
    for job_dir in af3_io.iter_jobs(out_dir):
        job_name = os.path.basename(job_dir)
        try:
            ranking = af3_io.read_ranking_scores(job_dir)
        except (FileNotFoundError, ValueError):
            continue
        for flat_index, cif in af3_io.iter_samples(job_dir):
            row = ranking.iloc[flat_index]
            rank_key = f"{int(row['seed'])}_{int(row['sample'])}"
            conf = os.path.join(os.path.dirname(cif), "confidences.json")
            if os.path.exists(conf):
                tasks.append((job_name, rank_key, cif, conf))

    if not tasks:
        return pd.DataFrame(columns=_EMPTY_COLS)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_score_sample, jn, rk, cif, conf,
                          pae_cutoff, dist_cutoff, python_exe)
                for jn, rk, cif, conf in tasks]
        for f in futs:
            rows.extend(f.result())

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLS)
    return pd.DataFrame(rows)[_EMPTY_COLS]
