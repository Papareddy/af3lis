"""Structure parsing + PEAK — the only structure work the pipeline needs.

AF3 fork of boltzlis.structure. Three deliberate departures from boltzlis:

1. ``parse_structure`` defaults to ``chain_id_source='label_asym_id'`` (NOT
   ``auth_asym_id``). AF3's ``chain_pair_iptm`` / ``chain_pair_pae_min`` are
   indexed by ``label_asym_id``; matching that ordering here keeps every
   chain-keyed dict in the pipeline consistent with the AF3 native matrices.

2. New ``load_pae_json`` — AF3 emits PAE inside ``<lname>_full_data_<N>.json``
   under the ``"pae"`` key (NaN-padded). Boltz's ``.npz`` loader (``load_pae_npz``)
   is kept only as a unit-test cross-check.

3. ``peak_per_chainpair`` is now token-axis. It takes ``chain_labels: list[str]``
   (= AF3 ``token_chain_ids``, length == ``pae.shape[0]``) and returns a dict
   keyed by the REAL AF3 chain IDs (``('A','B')`` rather than the positional
   ``chr(65+i)`` letters boltzlis used). This is essential for AF3 because the
   PAE matrix lives on tokens, not residues — for any ligand/PTM-bearing system
   the residue-axis chain assignment from the CIF would disagree with the PAE.

There is intentionally NO length-sanity assert between PAE shape and the CIF
residue count: PEAK runs on the token axis, ``parse_structure`` runs on the
residue axis, and the two are allowed to differ. The parity check that matters
is token-axis-PEAK ≡ ``1 − chain_pair_pae_min/30`` from
``<lname>_summary_confidences_<N>.json`` (asserted in tests/test_metrics.py).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# PDB / CIF representative-atom extraction (Cβ, or Cα for Gly / missing Cβ).
# Stdlib-only — no Biopython dependency.
# ---------------------------------------------------------------------------

def _rep_atoms_pdb(path: str) -> tuple[list[str], np.ndarray]:
    best: dict[tuple[str, str, str], tuple[int, tuple[float, float, float]]] = {}
    order: list[tuple[str, str, str]] = []
    with open(path) as fh:
        for ln in fh:
            if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                continue
            atom = ln[12:16].strip()
            resn = ln[17:20].strip()
            key = (ln[21], ln[22:26].strip(), ln[26])
            want = 2 if (atom == "CB" and resn != "GLY") else (1 if atom == "CA" else 0)
            if not want:
                continue
            try:
                xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            except ValueError:
                continue
            if key not in best:
                best[key] = (want, xyz)
                order.append(key)
            elif want > best[key][0]:
                best[key] = (want, xyz)
    return [k[0] for k in order], np.array([best[k][1] for k in order], float)


def _rep_atoms_cif(
    path: str,
    chain_id_source: str = "label_asym_id",
) -> tuple[list[str], np.ndarray]:
    """Streaming mmCIF _atom_site parser.

    ``chain_id_source`` selects which CIF column supplies the chain label:
    ``'label_asym_id'`` (default, matches AF3 native chain ordering) or
    ``'auth_asym_id'`` (boltzlis-style). Falls back to the other if the
    preferred column is absent.
    """
    if chain_id_source not in ("label_asym_id", "auth_asym_id"):
        raise ValueError(
            f"chain_id_source must be 'label_asym_id' or 'auth_asym_id', "
            f"got {chain_id_source!r}"
        )
    # AF3 FIX #1 — strict loop state machine.
    # mmCIF allows multiple loops in one file; the previous parser treated any
    # `_atom_site.*` line as a header and kept appending rows after the
    # _atom_site loop ended. We now use an explicit OUTSIDE/IN_HEADER/IN_DATA
    # state and exit cleanly on the first loop terminator (`_`, `loop_`,
    # `data_`, `save_`, `#`).
    header: list[str] = []
    rows: list[list[str]] = []
    state = "OUTSIDE"  # OUTSIDE | IN_HEADER | IN_DATA
    with open(path) as fh:
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
                # A header-only line for a different category ends our header.
                if s.startswith("_") or s.startswith("loop_") or \
                   s.startswith("data_") or s.startswith("save_") or s == "#":
                    # _atom_site loop closed before we saw a data row.
                    break
                if not s:
                    continue
                # First data row.
                state = "IN_DATA"
                rows.append(s.split())
                continue
            if state == "IN_DATA":
                if s.startswith("_") or s.startswith("loop_") or \
                   s.startswith("data_") or s.startswith("save_") or s == "#":
                    break
                if not s:
                    continue
                rows.append(s.split())
    idx = {n.split(".")[1]: i for i, n in enumerate(header)}

    def c(*names: str) -> int | None:
        for n in names:
            if n in idx:
                return idx[n]
        return None

    # Honor the requested chain_id_source first; fall back to the other.
    if chain_id_source == "label_asym_id":
        ch_col = c("label_asym_id", "auth_asym_id")
    else:
        ch_col = c("auth_asym_id", "label_asym_id")

    ci = dict(
        atom=c("label_atom_id", "auth_atom_id"),
        comp=c("label_comp_id", "auth_comp_id"),
        ch=ch_col,
        seq=c("label_seq_id", "auth_seq_id"),
        x=c("Cartn_x"),
        y=c("Cartn_y"),
        z=c("Cartn_z"),
        model=c("pdbx_PDB_model_num"),
    )
    if any(ci[k] is None for k in ("atom", "comp", "ch", "seq", "x", "y", "z")):
        raise ValueError(f"mmCIF missing required _atom_site columns in {path}")

    best: dict[tuple[str, str], tuple[int, tuple[float, float, float]]] = {}
    order: list[tuple[str, str]] = []
    # AF3 emits single-model CIFs. Accept model 1 and mmCIF 'inapplicable'/
    # 'unknown' sentinels ('.', '?'); silently drop later models (NMR-like
    # multimodel CIFs aren't produced by AF3).
    for r in rows:
        if ci["model"] is not None and r[ci["model"]] not in ("1", ".", "?"):
            continue
        atom = r[ci["atom"]].strip('"')
        resn = r[ci["comp"]]
        key = (r[ci["ch"]], r[ci["seq"]])
        want = 2 if (atom == "CB" and resn != "GLY") else (1 if atom == "CA" else 0)
        if not want:
            continue
        try:
            xyz = (float(r[ci["x"]]), float(r[ci["y"]]), float(r[ci["z"]]))
        except ValueError:
            continue
        if key not in best:
            best[key] = (want, xyz)
            order.append(key)
        elif want > best[key][0]:
            best[key] = (want, xyz)
    return [k[0] for k in order], np.array([best[k][1] for k in order], float)


# ---------------------------------------------------------------------------
# Public structure API
# ---------------------------------------------------------------------------

def parse_structure(
    path: str,
    chain_id_source: str = "label_asym_id",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(asym_id_int_array, coords[N,3])`` — one row per residue rep-atom.

    ``asym_id_int_array`` encodes chains as 0,1,2,... in CIF order of appearance,
    matching the boltzlis return contract.

    ``chain_id_source`` defaults to ``'label_asym_id'`` (AF3 native ordering).
    Pass ``'auth_asym_id'`` only for boltzlis-style PDB inputs.
    """
    if path.endswith((".cif", ".mmcif")):
        chains, coords = _rep_atoms_cif(path, chain_id_source=chain_id_source)
    else:
        chains, coords = _rep_atoms_pdb(path)
    seen: dict[str, int] = {}
    asym: list[int] = []
    for ch in chains:
        if ch not in seen:
            seen[ch] = len(seen)
        asym.append(seen[ch])
    return np.array(asym, dtype=int), coords


def chain_labels_from_cif(
    path: str,
    chain_id_source: str = "label_asym_id",
) -> list[str]:
    """Per-residue chain labels (e.g. ``['A','A',...,'B','B',...]``) in CIF order.

    Same parse as :func:`parse_structure` but returns the raw string chain IDs
    instead of the integer-encoded indices. ``collect.py`` uses this when it
    needs to map structure-axis indices back to AF3's actual chain letters.
    """
    if path.endswith((".cif", ".mmcif")):
        chains, _ = _rep_atoms_cif(path, chain_id_source=chain_id_source)
    else:
        chains, _ = _rep_atoms_pdb(path)
    return list(chains)


def chain_order_from_cif(
    path: str,
    chain_id_source: str = "label_asym_id",
) -> list[str]:
    """Unique chain IDs in their CIF order of first appearance.

    Matches the row/col ordering of AF3's ``chain_pair_iptm`` and
    ``chain_pair_pae_min`` matrices (which are keyed by ``label_asym_id``).
    """
    seen: dict[str, None] = {}
    for ch in chain_labels_from_cif(path, chain_id_source=chain_id_source):
        if ch not in seen:
            seen[ch] = None
    return list(seen.keys())


def chain_lengths(
    path: str,
    chain_id_source: str = "label_asym_id",
) -> dict[str, int]:
    """Residue counts per chain, keyed by AF3 chain ID (label_asym_id by default).

    Convenience helper for ``collect.py`` and the plotter (e.g. the "pTM<0.05
    floor for <20 tokens" warning gates on total chain length). Counts are
    over representative atoms — i.e. per-residue — NOT per-token, so for an
    all-protein complex this equals the residue count and for a
    ligand/PTM-bearing chain it under-counts the token axis (which is the
    point: this is a structure-side helper, not a PAE-side helper).
    """
    labels = chain_labels_from_cif(path, chain_id_source=chain_id_source)
    out: dict[str, int] = {}
    for ch in labels:
        out[ch] = out.get(ch, 0) + 1
    return out


# ---------------------------------------------------------------------------
# PAE loaders
# ---------------------------------------------------------------------------

def load_pae_npz(path: str) -> np.ndarray:
    """Legacy Boltz ``.npz`` PAE loader — kept ONLY for the unit-test cross-check.

    af3lis runtime never reads ``.npz``; AF3 stores PAE as JSON.
    """
    d = np.load(path)
    return d[d.files[0]]


def load_pae_json(path: str, key: str = "pae") -> np.ndarray:
    """Load the PAE matrix from an AF3 ``<lname>_full_data_<N>.json``.

    Returns a square ``float`` array of shape ``[num_tokens, num_tokens]``.
    NaNs are replaced with ``31.0`` to match the convention in lis.py
    (``extract_pae``) — keeps PEAK/iLIS arithmetic on a finite domain
    without inventing structure where AF3 declined to predict it.
    """
    with open(path) as fh:
        d = json.load(fh)
    if key not in d:
        raise KeyError(f"key {key!r} not in {path} (have: {sorted(d)[:6]}...)")
    # float32 captures AF3's PAE precision losslessly and halves memory for
    # large complexes (3100 tokens => ~38 MB vs ~76 MB for float64).
    pae = np.asarray(d[key], dtype=np.float32)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(
            f"PAE in {path} is not square: shape={pae.shape}"
        )
    # NaN -> 31.0 (lis.py convention; safely above the PEAK cutoff of 30).
    # Don't clamp posinf/neginf — AF3 doesn't emit them; if it ever does, fail
    # loudly via the downstream PEAK arithmetic rather than silently clamping.
    pae = np.nan_to_num(pae, nan=31.0)
    return pae


def load_token_chain_ids(path: str, key: str = "token_chain_ids") -> list[str]:
    """Per-token chain IDs from ``<lname>_full_data_<N>.json``.

    Length equals ``pae.shape[0]``. This is THE canonical chain assignment
    for PEAK in af3lis — never use the residue-axis assignment from the CIF
    for PAE-derived metrics.
    """
    with open(path) as fh:
        d = json.load(fh)
    if key not in d:
        raise KeyError(f"key {key!r} not in {path} (have: {sorted(d)[:6]}...)")
    return [str(x) for x in d[key]]


# ---------------------------------------------------------------------------
# Structure lookup — AF3 standard-CLI flat layout
# ---------------------------------------------------------------------------

def find_structure(pdir: str, model: int | str) -> str | None:
    """Locate a model file inside ``<out>/<lname>/`` (AF3 flat layout).

    ``model``:
      - ``int N``   -> ``<lname>_model_<N>.cif``       (flat 0-based sample index)
      - ``'best'``  -> ``<lname>_model.cif``           (top-level ranking copy)

    Returns the absolute path or ``None`` if not found. The boltzlis ``.pdb``
    fallback is dropped — AF3 ≥3.0.1 always emits ``.cif``.
    """
    pdir = os.path.abspath(pdir)
    # Accept numeric strings (e.g. CLI args) up-front.
    if isinstance(model, str) and model.isdigit():
        model = int(model)
    # bool is a subclass of int — reject explicitly to avoid model=True silently
    # resolving to model=1.
    if isinstance(model, bool):
        raise TypeError(f"model must be int or 'best', got bool {model!r}")
    if isinstance(model, str) and model == "best":
        # Top-level ranking copy: '<lname>_model.cif'. lname == dir basename
        # (AF3 lowercases the job name into the dir name).
        lname = os.path.basename(pdir.rstrip("/"))
        cand = os.path.join(pdir, f"{lname}_model.cif")
        if os.path.isfile(cand):
            return cand
        # Defensive glob in case the dir was renamed post-hoc.
        hits = sorted(glob.glob(os.path.join(pdir, "*_model.cif")))
        # Exclude '*_model_<N>.cif' sample files — they share the prefix.
        hits = [h for h in hits if "_model_" not in os.path.basename(h)]
        return hits[0] if hits else None
    if isinstance(model, int):
        # Exact filename match avoids picking up unrelated `*_model_{N}.cif`
        # files (e.g. user-copied artefacts) that share the suffix.
        lname = os.path.basename(pdir.rstrip("/"))
        cand = os.path.join(pdir, f"{lname}_model_{model}.cif")
        if os.path.isfile(cand):
            return cand
        # Defensive glob fallback for AF3 builds that sanitize dir names
        # differently than the file prefix.
        hits = sorted(glob.glob(os.path.join(pdir, f"*_model_{model}.cif")))
        return hits[0] if hits else None
    raise TypeError(f"model must be int or 'best', got {model!r}")


# ---------------------------------------------------------------------------
# PEAK — token-axis, keyed by real AF3 chain IDs
# ---------------------------------------------------------------------------

def peak_per_chainpair(
    pae: np.ndarray,
    chain_labels: Iterable[str],
    cutoff: float = 30.0,
) -> dict[tuple[str, str], float]:
    """PEAK per chain pair, computed on the AF3 token axis.

    Parameters
    ----------
    pae
        ``[num_tokens, num_tokens]`` PAE array (from ``load_pae_json``).
    chain_labels
        Per-token chain IDs (from ``load_token_chain_ids``); length must equal
        ``pae.shape[0]``. Typically AF3's ``token_chain_ids``.
    cutoff
        PEAK normalization cutoff (PAE Å). House value is 30.

    Returns
    -------
    dict
        ``{(chain_i, chain_j): PEAK}`` for every unordered pair of distinct
        chains. Chain ordering follows first-appearance in ``chain_labels``
        (i.e. AF3 token / CIF emit order), NOT lexicographic sort. This
        matches the row/col ordering of AF3's native ``chain_pair_pae_min``
        matrix and is robust to multi-letter chain IDs (e.g. ``AA``, ``AB``
        for >26 chains where lexicographic sort would mis-order).
        ``PEAK = max(0, 1 − min(off-diagonal PAE block) / cutoff)``, symmetric
        over both off-diagonal blocks. Returns ``{}`` for monomeric inputs.

    Notes
    -----
    Differs from boltzlis in two ways:
      1. Output keys are the REAL chain IDs (e.g. ``('A','B')``, ``('B','D')``),
         not positional letters ``chr(65+i)``. Required because AF3 chain IDs
         can skip letters when the CIF order differs from emit order, and we
         want PEAK keys to agree byte-for-byte with AF3's
         ``chain_pair_pae_min`` matrix (cross-checked in tests).
      2. Operates on the token axis. No CIF parse required — PEAK is purely a
         PAE-derived metric in af3lis.
    """
    chain_labels = list(chain_labels)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"PAE must be square 2D; got shape {pae.shape}")
    if len(chain_labels) != pae.shape[0]:
        raise ValueError(
            f"chain_labels length {len(chain_labels)} != pae.shape[0] "
            f"{pae.shape[0]} — chain_labels must be per-TOKEN, not per-residue"
        )

    # AF3 FIX #3 — preserve first-appearance order from chain_labels (matches
    # AF3's native chain_pair_pae_min row/col ordering) instead of lex-sorting.
    # Lex sort breaks for multi-letter chain IDs (`AA` < `B`) and is brittle if
    # AF3 ever changes emit order.
    arr = np.asarray(chain_labels, dtype=object)
    unique = list(dict.fromkeys(chain_labels))
    if len(unique) < 2:
        return {}
    idx_of: dict[str, np.ndarray] = {c: np.where(arr == c)[0] for c in unique}

    out: dict[tuple[str, str], float] = {}
    for i, ca in enumerate(unique):
        ia = idx_of[ca]
        for cb in unique[i + 1:]:
            ib = idx_of[cb]
            # Both off-diagonal blocks (PAE is asymmetric in general).
            blk_ab = pae[np.ix_(ia, ib)]
            blk_ba = pae[np.ix_(ib, ia)]
            mn = float(min(blk_ab.min(), blk_ba.min()))
            peak = max(0.0, 1.0 - mn / cutoff)
            # No rounding — preserves byte-identity with AF3's
            # `1 - chain_pair_pae_min/30` for the unit-test cross-check.
            # Emit BOTH (ca,cb) and (cb,ca) so callers can look up symmetric
            # pairs without needing to canonicalize. Matches boltzlis convention.
            out[(ca, cb)] = peak
            out[(cb, ca)] = peak
    return out
