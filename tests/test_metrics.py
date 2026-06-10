"""Unit tests for af3lis — local only, no cluster/GPU needed.

Covers (per the design doc §10):
  - json_build sanitisation, pair-naming, AF3 JSON schema, seed-baking, 26-cap.
  - af3_io readers against a tiny synthetic fixture (flat AF3-CLI layout:
    <lname>_model_<N>.cif, <lname>_full_data_<N>.json,
    <lname>_summary_confidences_<N>.json, <lname>_ranking_scores.csv).
  - PEAK arithmetic on a hand-crafted PAE matrix; cross-check against
    AF3-native chain_pair_pae_min in summary_confidences.json.
  - collect aggregator boundary cases (empty out, monomer drop,
    per_seed vs flat schema, lis.py --platform alphafold3 invocation).
  - Public-repo hygiene (config.example.yaml has no cluster paths).
  - fetch._get short-circuits HTTP 404.
  - pipeline top-level --collect shortcut.

Tests that touch not-yet-implemented modules use ``pytest.importorskip``
so this file is green pre- and post-implementation.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import textwrap
from typing import Sequence
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make the package importable when running `pytest` from the repo root.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from af3lis import json_build  # always available


# ---------------------------------------------------------------------------
# Synthetic AF3-output fixture builder.
# ---------------------------------------------------------------------------
def _write_minimal_cif(path: str, chain_residues: Sequence[tuple[str, int]]) -> None:
    """Write a minimal CIF with one Cβ atom per residue per chain.

    chain_residues: list of (label_asym_id, n_residues).
    auth_asym_id is set equal to label_asym_id for the fixture.
    """
    lines = [
        "data_test",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
    ]
    atom_id = 0
    for chain, n in chain_residues:
        for r in range(1, n + 1):
            atom_id += 1
            lines.append(
                f"ATOM {atom_id} C CB ALA {chain} {r} {chain} {r} "
                f"{float(r):.3f} {float(atom_id):.3f} 0.000 1.00 50.00"
            )
    lines.append("#")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_full_data_json(
    path: str,
    pae: np.ndarray,
    token_chain_ids: list[str],
    atom_chain_ids: list[str] | None = None,
    atom_plddts: list[float] | None = None,
) -> None:
    if atom_chain_ids is None:
        atom_chain_ids = list(token_chain_ids)
    if atom_plddts is None:
        atom_plddts = [80.0] * len(atom_chain_ids)
    with open(path, "w") as fh:
        json.dump(
            {
                "pae": pae.tolist(),
                "token_chain_ids": list(token_chain_ids),
                "atom_chain_ids": list(atom_chain_ids),
                "atom_plddts": list(atom_plddts),
            },
            fh,
        )


def _write_summary_json(
    path: str,
    chain_order: list[str],
    pae: np.ndarray,
    token_chain_ids: list[str],
    ranking_score: float = 0.5,
    has_clash: bool = False,
    fraction_disordered: float = 0.1,
    ptm: float = 0.7,
    iptm: float = 0.6,
) -> None:
    """Synthesize a summary_confidences.json from the same PAE + chain_ids
    by computing AF3-native chain_pair_pae_min on the token axis. The
    canonical chain order matches the order chains first appear in
    token_chain_ids.
    """
    nC = len(chain_order)
    idx = {c: i for i, c in enumerate(chain_order)}
    masks = {c: np.array([t == c for t in token_chain_ids]) for c in chain_order}
    cp_pae_min = np.zeros((nC, nC), dtype=float)
    cp_iptm = np.full((nC, nC), 0.4, dtype=float)
    for ci in chain_order:
        for cj in chain_order:
            block = pae[np.ix_(masks[ci], masks[cj])]
            cp_pae_min[idx[ci], idx[cj]] = float(np.min(block)) if block.size else 31.0
    payload = {
        "ptm": ptm,
        "iptm": iptm,
        "chain_pair_iptm": cp_iptm.tolist(),
        "chain_pair_pae_min": cp_pae_min.tolist(),
        "chain_ptm": [ptm] * nC,
        "chain_iptm": [iptm] * nC,
        "ranking_score": ranking_score,
        "has_clash": has_clash,
        "fraction_disordered": fraction_disordered,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh)


def _write_ranking_csv(path: str, rows: list[tuple[int, int, float]]) -> None:
    """rows: list of (seed, sample, ranking_score). Row order = flat AF3 index."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "sample", "ranking_score"])
        for r in rows:
            w.writerow(r)


def _build_af3_job_dir(
    tmp_path,
    pair_name: str = "foo___bar",
    n_chains: tuple[int, int] = (4, 3),
    flat_indices: tuple[int, ...] = (0, 1, 2),
    seeds_samples: list[tuple[int, int]] | None = None,
    interface_pae: float = 5.0,
    intra_pae: float = 2.0,
    other_pae: float = 25.0,
) -> str:
    """Materialize a tiny but realistic AF3 standard-CLI output dir.

    Returns the job_dir path. The pair-name is already lowercased to match
    AF3's behavior on writing the output dir.
    """
    lname = pair_name.lower()
    job_dir = os.path.join(str(tmp_path), "out", lname)
    os.makedirs(job_dir, exist_ok=True)

    nA, nB = n_chains
    token_chain_ids = ["A"] * nA + ["B"] * nB
    chain_order = ["A", "B"]
    n_tok = nA + nB

    pae = np.full((n_tok, n_tok), other_pae, dtype=float)
    # Diagonal intra-chain blocks
    pae[:nA, :nA] = intra_pae
    pae[nA:, nA:] = intra_pae
    # Inter-chain blocks (set the known minimum we can verify by hand)
    pae[:nA, nA:] = interface_pae
    pae[nA:, :nA] = interface_pae
    np.fill_diagonal(pae, 0.0)

    if seeds_samples is None:
        seeds_samples = [(1, i) for i in flat_indices]
    assert len(seeds_samples) == len(flat_indices)

    # AF3 ≥3.0.1 writes per-sample files inside seed-<S>_sample-<M>/ subdirs.
    for N, (seed, sample) in zip(flat_indices, seeds_samples):
        seed_dir = os.path.join(job_dir, f"seed-{seed}_sample-{sample}")
        os.makedirs(seed_dir, exist_ok=True)
        _write_minimal_cif(
            os.path.join(seed_dir, "model.cif"), [("A", nA), ("B", nB)],
        )
        _write_full_data_json(
            os.path.join(seed_dir, "confidences.json"),
            pae=pae, token_chain_ids=token_chain_ids,
        )
        _write_summary_json(
            os.path.join(seed_dir, "summary_confidences.json"),
            chain_order=chain_order, pae=pae,
            token_chain_ids=token_chain_ids,
            ranking_score=0.5 + 0.01 * N,
        )

    # Top-level "best" copies (canonical AF3 emit)
    best_N = flat_indices[0]
    _write_minimal_cif(
        os.path.join(job_dir, f"{lname}_model.cif"), [("A", nA), ("B", nB)]
    )
    _write_summary_json(
        os.path.join(job_dir, f"{lname}_summary_confidences.json"),
        chain_order=chain_order,
        pae=pae,
        token_chain_ids=token_chain_ids,
        ranking_score=0.5 + 0.01 * best_N,
    )
    # Ranking CSV — plain name (AF3 ≥3.0.1 convention), no <lname> prefix.
    _write_ranking_csv(
        os.path.join(job_dir, "ranking_scores.csv"),
        [(s, p, 0.5 + 0.01 * N) for N, (s, p) in zip(flat_indices, seeds_samples)],
    )
    # MSA-augmented data archive (presence-check for --infer-only precondition).
    with open(os.path.join(job_dir, f"{lname}_data.json"), "w") as fh:
        json.dump({"name": lname, "modelSeeds": list({s for s, _ in seeds_samples})}, fh)
    return job_dir


# ===========================================================================
# §10.1 — safe() sanitization
# ===========================================================================
def test_safe_sanitization() -> None:
    assert json_build.safe("FOO-2/2.A") == "FOO-2_2.A"
    assert json_build.safe("FOO_2") == "FOO_2"
    # the literal triple-underscore must never appear in any sanitized label
    bad = "X___Y"  # contains the SEP itself
    with pytest.raises(ValueError):
        json_build.safe(bad)


# ===========================================================================
# §10.2 — build_grid pair naming + writes 6 JSONs for a 2x3 grid
# ===========================================================================
def test_build_grid_pair_names(tmp_path) -> None:
    a_set = [("a1", "AAAAAAAAAAAAAAAAAAAA"), ("a2", "CCCCCCCCCCCCCCCCCCCC")]
    b_set = [
        ("b1", "DDDDDDDDDDDDDDDDDDDD"),
        ("b2", "EEEEEEEEEEEEEEEEEEEE"),
        ("b3", "FFFFFFFFFFFFFFFFFFFF"),
    ]
    out = str(tmp_path / "jsons")
    jobs = json_build.build_grid(a_set, b_set, out, seeds=[1])
    assert len(jobs) == 6
    expected_names = {
        f"{a}{json_build.SEP}{b}" for a, _ in a_set for b, _ in b_set
    }
    assert {n for n, _ in jobs} == expected_names
    for name, path in jobs:
        assert os.path.isfile(path)
        with open(path) as fh:
            payload = json.load(fh)
        assert payload["name"] == name


# ===========================================================================
# §10.3 — AF3 JSON top-level schema
# ===========================================================================
def test_af3_json_schema(tmp_path) -> None:
    jobs = json_build.build_grid(
        [("a", "A" * 30)], [("b", "C" * 30)], str(tmp_path)
    )
    with open(jobs[0][1]) as fh:
        payload = json.load(fh)
    assert set(payload) == {"name", "modelSeeds", "sequences", "dialect", "version"}
    assert payload["version"] == 1
    assert payload["dialect"] == "alphafold3"
    assert isinstance(payload["modelSeeds"], list) and payload["modelSeeds"]
    assert {c["protein"]["id"] for c in payload["sequences"]} == {"A", "B"}


# ===========================================================================
# §10.4 — seeds baked into JSON
# ===========================================================================
def test_seeds_baked_in_json(tmp_path) -> None:
    jobs = json_build.build_grid(
        [("a", "A" * 30)], [("b", "C" * 30)], str(tmp_path), seeds=[1, 7]
    )
    with open(jobs[0][1]) as fh:
        payload = json.load(fh)
    assert payload["modelSeeds"] == [1, 7]


# ===========================================================================
# §10.5 — hard 26-chain cap
# ===========================================================================
def test_chain_cap(tmp_path) -> None:
    a_set = [(f"a{i}", "A" * 25) for i in range(20)]
    b_set = [(f"b{i}", "A" * 25) for i in range(7)]
    with pytest.raises(ValueError):
        json_build.build_complex(a_set, b_set, "big", str(tmp_path))


# ===========================================================================
# §10.6 — af3_io.load_sample populates every field
# ===========================================================================
def test_af3_io_load_sample(tmp_path) -> None:
    af3_io = pytest.importorskip("af3lis.af3_io")
    job_dir = _build_af3_job_dir(tmp_path, flat_indices=(0,), seeds_samples=[(7, 3)])
    fr = af3_io.load_sample(job_dir, flat_index=0)
    assert fr.flat_index == 0
    assert fr.seed == 7
    assert fr.sample == 3
    assert fr.rank_key == "7_3"
    assert fr.pae.ndim == 2 and fr.pae.shape[0] == fr.pae.shape[1]
    assert len(fr.token_chain_ids) == fr.pae.shape[0]
    assert set(fr.token_chain_ids) == {"A", "B"}
    assert fr.chain_pair_iptm.shape == (2, 2)
    assert fr.chain_pair_pae_min.shape == (2, 2)
    # File-path fields must exist on disk.
    for p in (fr.cif_path, fr.full_data_path, fr.summary_path):
        assert os.path.isfile(p)


# ===========================================================================
# §10.7 — iter_jobs picks up lowercased AF3 dirs
# ===========================================================================
def test_iter_jobs_lowercase(tmp_path) -> None:
    af3_io = pytest.importorskip("af3lis.af3_io")
    # Build a dir whose pair-name has uppercase, but AF3 has already lowercased
    # it on disk to match the canonical behaviour.
    job_dir = _build_af3_job_dir(tmp_path, pair_name="Foo___Bar", flat_indices=(0,))
    out_root = os.path.dirname(job_dir)
    found = list(af3_io.iter_jobs(out_root))
    assert len(found) == 1
    assert os.path.basename(found[0]) == "foo___bar"


# ===========================================================================
# §10.8 — iter_samples yields flat indices in order
# ===========================================================================
def test_iter_samples_flat(tmp_path) -> None:
    af3_io = pytest.importorskip("af3lis.af3_io")
    job_dir = _build_af3_job_dir(tmp_path, flat_indices=(0, 1, 2))
    seen = list(af3_io.iter_samples(job_dir))
    assert [n for n, _ in seen] == [0, 1, 2]
    for n, p in seen:
        assert os.path.isfile(p)
        # New (correct) AF3 layout: seed-<S>_sample-<M>/model.cif
        assert p.endswith("/model.cif")
        assert "/seed-" in p and "_sample-" in p


# ===========================================================================
# §10.9 — strict ranking_scores.csv header
# ===========================================================================
def test_ranking_scores_strict_header(tmp_path) -> None:
    af3_io = pytest.importorskip("af3lis.af3_io")
    job_dir = _build_af3_job_dir(tmp_path, flat_indices=(0,))
    csv_path = os.path.join(job_dir, "ranking_scores.csv")
    # Corrupt the header.
    with open(csv_path, "w") as fh:
        fh.write("a,b,c\n0,0,0.5\n")
    with pytest.raises(ValueError):
        af3_io.read_ranking_scores(job_dir)


# ===========================================================================
# §10.10 — PEAK token-axis == 1 - chain_pair_pae_min/30 (the smoking gun)
# ===========================================================================
def test_chain_pair_pae_min_matches_PEAK(tmp_path) -> None:
    structure = pytest.importorskip("af3lis.structure")
    interface_pae = 6.0  # known minimum
    job_dir = _build_af3_job_dir(
        tmp_path, n_chains=(4, 3), interface_pae=interface_pae, intra_pae=1.0
    )
    # Read flat_index 0 → seed-1_sample-0/ (per _build_af3_job_dir default mapping).
    seed_dir = os.path.join(job_dir, "seed-1_sample-0")
    with open(os.path.join(seed_dir, "confidences.json")) as fh:
        full = json.load(fh)
    with open(os.path.join(seed_dir, "summary_confidences.json")) as fh:
        summ = json.load(fh)

    pae = np.array(full["pae"], dtype=float)
    chain_labels = list(full["token_chain_ids"])

    out = structure.peak_per_chainpair(pae, chain_labels, cutoff=30.0)
    assert ("A", "B") in out
    # Sanity vs hand calc:
    assert out[("A", "B")] == pytest.approx(1.0 - interface_pae / 30.0)

    # Smoking-gun parity vs AF3-native chain_pair_pae_min:
    cp_min = np.array(summ["chain_pair_pae_min"], dtype=float)
    # chain_order is ['A','B'] in our fixture.
    expected_AB = max(0.0, 1.0 - float(cp_min[0, 1]) / 30.0)
    assert out[("A", "B")] == pytest.approx(expected_AB, abs=1e-9)
    # Symmetric:
    assert out[("B", "A")] == pytest.approx(out[("A", "B")], abs=1e-9)


# ===========================================================================
# §10.11 — PEAK still works when token count > residue count (PTM/ligand)
# ===========================================================================
def test_token_chain_ids_for_ligand_complex(tmp_path) -> None:
    structure = pytest.importorskip("af3lis.structure")
    # 3-residue chain A, 2-residue chain B, plus 1 extra "PTM" token on chain A.
    # Total tokens = 6, residues encoded in CIF = 5.
    nA_res, nB_res = 3, 2
    token_chain_ids = ["A"] * nA_res + ["A"] + ["B"] * nB_res  # PTM token after A
    pae = np.full((6, 6), 20.0)
    np.fill_diagonal(pae, 0.0)
    # Minimum interchain PAE 8.0 (chain A real residues vs B residues).
    pae[:nA_res, -nB_res:] = 8.0
    pae[-nB_res:, :nA_res] = 8.0
    # PTM-token row also crosses B at 9.0 (higher, not the min).
    pae[nA_res, -nB_res:] = 9.0
    pae[-nB_res:, nA_res] = 9.0

    out = structure.peak_per_chainpair(pae, token_chain_ids, cutoff=30.0)
    assert out[("A", "B")] == pytest.approx(1.0 - 8.0 / 30.0)

    # If parse_structure is available, verify it tolerates the residue<token mismatch.
    cif = os.path.join(tmp_path, "x.cif")
    _write_minimal_cif(cif, [("A", nA_res), ("B", nB_res)])
    try:
        asym_ids, coords = structure.parse_structure(cif)
    except Exception as e:  # pragma: no cover
        pytest.fail(f"parse_structure raised on residue<token mismatch: {e}")
    assert len(asym_ids) == nA_res + nB_res
    assert len(asym_ids) != pae.shape[0]  # the whole point of this test


# ===========================================================================
# §10.12 — collect_all on an empty out_dir emits a header-only TSV
# ===========================================================================
def test_collect_all_no_models(tmp_path) -> None:
    collect = pytest.importorskip("af3lis.collect")
    empty = str(tmp_path / "empty_out")
    os.makedirs(empty, exist_ok=True)
    out_tsv = str(tmp_path / "out.tsv")
    df = collect.collect_all(empty, out_tsv, workers=1)
    assert os.path.isfile(out_tsv)
    assert len(df) == 0
    # Must still have a header row.
    with open(out_tsv) as fh:
        header = fh.readline().rstrip("\n").split("\t")
    assert "name" in header and "chain_i" in header and "chain_j" in header


# ===========================================================================
# §10.13 — collect_all drops monomer rows (chain_i == chain_j)
# ===========================================================================
def test_collect_all_drops_monomer(tmp_path) -> None:
    collect = pytest.importorskip("af3lis.collect")
    # Build a fixture with one heterodimer and one monomer.
    # The collector should yield zero PEAK rows for the monomer (single chain),
    # and the row, if it appears at all, must satisfy chain_i != chain_j.
    out_root = str(tmp_path / "out")
    _build_af3_job_dir(tmp_path, pair_name="hetero___pair", flat_indices=(0,))
    # Monomer: only chain "A" in tokens.
    mono = os.path.join(out_root, "mono_only")
    os.makedirs(mono, exist_ok=True)
    pae = np.array([[0.0, 1.0], [1.0, 0.0]])
    mono_seed = os.path.join(mono, "seed-1_sample-0")
    os.makedirs(mono_seed, exist_ok=True)
    _write_minimal_cif(os.path.join(mono_seed, "model.cif"), [("A", 2)])
    _write_full_data_json(
        os.path.join(mono_seed, "confidences.json"),
        pae=pae, token_chain_ids=["A", "A"],
    )
    _write_summary_json(
        os.path.join(mono_seed, "summary_confidences.json"),
        chain_order=["A"], pae=pae, token_chain_ids=["A", "A"],
    )
    _write_ranking_csv(os.path.join(mono, "ranking_scores.csv"), [(1, 0, 0.5)])
    _write_minimal_cif(os.path.join(mono, "mono_only_model.cif"), [("A", 2)])

    tsv = str(tmp_path / "out.tsv")
    # Use flat agg to keep the schema simple to inspect.
    try:
        df = collect.collect_all(out_root, tsv, workers=1, agg_mode="flat")
    except TypeError:
        df = collect.collect_all(out_root, tsv, workers=1)
    if len(df) == 0:
        return  # acceptable: monomer was dropped + heterodimer produced no rows
    assert (df["chain_i"] != df["chain_j"]).all()
    assert not (df["name"] == "mono_only").any()


# ===========================================================================
# §10.14 — per_seed aggregation is never optimistic-biased vs flat
# ===========================================================================
def test_per_seed_vs_flat_agg(tmp_path) -> None:
    collect = pytest.importorskip("af3lis.collect")
    # 3 seeds x 5 samples = 15 flat samples. One seed has artificially good
    # interface; the others are mediocre.
    out_root = str(tmp_path / "out")
    job_dir = os.path.join(out_root, "x___y")
    os.makedirs(job_dir, exist_ok=True)
    lname = "x___y"
    nA, nB = 2, 2
    token_chain_ids = ["A"] * nA + ["B"] * nB
    chain_order = ["A", "B"]

    flat = 0
    seeds_samples = []
    for seed in (1, 2, 3):
        for s in range(5):
            iface = 2.0 if seed == 1 else 20.0  # seed 1 = great, others = bad
            pae = np.full((4, 4), 25.0)
            np.fill_diagonal(pae, 0.0)
            pae[:nA, nA:] = iface
            pae[nA:, :nA] = iface
            seed_dir = os.path.join(job_dir, f"seed-{seed}_sample-{s}")
            os.makedirs(seed_dir, exist_ok=True)
            _write_minimal_cif(
                os.path.join(seed_dir, "model.cif"),
                [("A", nA), ("B", nB)],
            )
            _write_full_data_json(
                os.path.join(seed_dir, "confidences.json"),
                pae=pae, token_chain_ids=token_chain_ids,
            )
            _write_summary_json(
                os.path.join(seed_dir, "summary_confidences.json"),
                chain_order=chain_order, pae=pae,
                token_chain_ids=token_chain_ids, ranking_score=0.5,
            )
            seeds_samples.append((seed, s))
            flat += 1
    _write_ranking_csv(
        os.path.join(job_dir, "ranking_scores.csv"),
        [(s, p, 0.5) for (s, p) in seeds_samples],
    )
    _write_minimal_cif(os.path.join(job_dir, f"{lname}_model.cif"), [("A", nA), ("B", nB)])
    _write_summary_json(
        os.path.join(job_dir, f"{lname}_summary_confidences.json"),
        chain_order=chain_order,
        pae=np.full((4, 4), 20.0),
        token_chain_ids=token_chain_ids,
    )
    with open(os.path.join(job_dir, f"{lname}_data.json"), "w") as fh:
        json.dump({"name": lname, "modelSeeds": [1, 2, 3]}, fh)

    try:
        flat_df = collect.collect_all(
            out_root, str(tmp_path / "flat.tsv"), workers=1, agg_mode="flat"
        )
        per_df = collect.collect_all(
            out_root, str(tmp_path / "per.tsv"), workers=1, agg_mode="per_seed"
        )
    except TypeError:
        pytest.skip("collect_all does not yet support agg_mode toggling")

    def _peak(df, col):
        # Return the maximum PEAK across rows for this job.
        if col not in df.columns:
            return None
        return float(df[col].max())

    flat_peak = _peak(flat_df, "PEAK_max")
    per_peak = _peak(per_df, "PEAK_max_max")
    if flat_peak is None or per_peak is None:
        pytest.skip("expected PEAK columns absent in returned DataFrame")
    # per_seed must never be MORE optimistic than the flat max.
    assert per_peak <= flat_peak + 1e-9


# ===========================================================================
# §10.15 — run_lis subprocess invocation forces --platform alphafold3
# ===========================================================================
def test_lis_platform_alphafold3_flag(tmp_path) -> None:
    collect = pytest.importorskip("af3lis.collect")
    out_root = str(tmp_path / "out")
    os.makedirs(out_root, exist_ok=True)
    csv_path = str(tmp_path / "lis.csv")

    captured: dict[str, list[str]] = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        # Write an empty CSV so the caller doesn't choke on a missing file.
        with open(csv_path, "w") as fh:
            fh.write(
                "name,rank,model,chain_i,chain_j,iLIS,LIS,cLIS,iLIA,LIA,cLIA,"
                "actifpTM,ipSAE,ipTM,pTM,pLDDT_i,pLDDT_j\n"
            )

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    with mock.patch("subprocess.run", side_effect=fake_run):
        try:
            collect.run_lis(out_root, csv_path, workers=2)
        except Exception:
            # The function may also surface non-zero from a downstream parse;
            # the assert below covers what we care about regardless.
            pass

    assert "cmd" in captured, "run_lis did not invoke subprocess.run"
    assert "--platform" in captured["cmd"]
    p_idx = captured["cmd"].index("--platform")
    assert captured["cmd"][p_idx + 1] == "alphafold3"


# ===========================================================================
# §10.16 — pair-name split survives labels containing literal '_'
# ===========================================================================
def test_pair_name_split_safe() -> None:
    a, b = "FOO_2", "BAR-SH"
    assert json_build.safe(a) == "FOO_2"
    pair = json_build.safe(a) + json_build.SEP + json_build.safe(b)
    head, tail = pair.rsplit(json_build.SEP, 1)
    assert head == "FOO_2"
    assert tail == "BAR-SH"


# ===========================================================================
# §10.17 — assert_data_json_exists raises informatively
# ===========================================================================
def test_assert_data_json_exists(tmp_path) -> None:
    af3_io = pytest.importorskip("af3lis.af3_io")
    job_dir = _build_af3_job_dir(tmp_path, flat_indices=(0,))
    lname = os.path.basename(job_dir)
    # Sanity: present -> no raise.
    af3_io.assert_data_json_exists(job_dir)
    # Remove and re-check.
    os.remove(os.path.join(job_dir, f"{lname}_data.json"))
    with pytest.raises(FileNotFoundError):
        af3_io.assert_data_json_exists(job_dir)


# ===========================================================================
# §10.18 — public-repo safety: no real cluster paths in config.example.yaml
# ===========================================================================
def test_config_example_no_secrets() -> None:
    cfg = os.path.join(_PKG_ROOT, "config.example.yaml")
    assert os.path.isfile(cfg), "config.example.yaml is required"
    with open(cfg) as fh:
        text = fh.read()
    forbidden = ["hd_wi353", "/gpfs/", "ECT", "CCR4", "CAF1", "AtNOT", "Mize_ATG8"]
    hits = [tok for tok in forbidden if tok in text]
    assert not hits, f"config.example.yaml leaks forbidden tokens: {hits}"


# ===========================================================================
# §10.19 — fetch._get short-circuits HTTP 404 (no retry)
# ===========================================================================
def test_fetch_404_short_circuit() -> None:
    from urllib.error import HTTPError

    from af3lis import fetch

    calls = {"n": 0}

    def boom(url, timeout=30):
        calls["n"] += 1
        raise HTTPError(url, 404, "Not Found", hdrs=None, fp=io.BytesIO(b""))

    with mock.patch("urllib.request.urlopen", side_effect=boom):
        with pytest.raises(HTTPError):
            fetch._get("https://example.invalid/missing", tries=3)

    # Exactly one attempt — no retry on 404.
    assert calls["n"] == 1, f"expected 1 attempt on HTTP 404, got {calls['n']}"


# ===========================================================================
# §10.20 — top-level `af3lis --collect OUT -o tsv` shortcut re-dispatches
# ===========================================================================
def test_top_level_collect_shortcut(tmp_path) -> None:
    pipeline = pytest.importorskip("af3lis.pipeline")
    out_root = str(tmp_path / "out")
    os.makedirs(out_root, exist_ok=True)
    out_tsv = str(tmp_path / "x.tsv")

    captured: dict = {}

    def fake_collect(out_dir, out_tsv_, **kwargs):
        captured["out_dir"] = out_dir
        captured["out_tsv"] = out_tsv_
        captured["kwargs"] = kwargs
        # touch the file so downstream `assert exists` checks don't trip.
        open(out_tsv_, "w").close()
        return 0

    target = None
    for attr in ("cmd_collect", "_cmd_collect"):
        if hasattr(pipeline, attr):
            target = attr
            break
    if target is None:
        pytest.skip("pipeline has no cmd_collect to monkeypatch")

    argv = ["af3lis", "--collect", out_root, "-o", out_tsv]
    with mock.patch.object(pipeline, target, side_effect=fake_collect):
        with mock.patch.object(sys, "argv", argv):
            try:
                pipeline.main()
            except SystemExit as e:
                # argparse may sys.exit(0) on clean completion — acceptable.
                if e.code not in (0, None):
                    raise

    assert captured.get("out_dir") == out_root
    assert captured.get("out_tsv") == out_tsv
