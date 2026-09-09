"""Minimal mmCIF -> PDB converter for AF3 model output.

AlphaFold 3 writes mmCIF only. Plenty of downstream tools (PyMOL scripts,
older docking/analysis code, ipsae.py's PDB mode) still want PDB, so this
emits a legacy-format file with pLDDT carried into the B-factor column, which
is the convention every AlphaFold viewer expects.

Deliberately dependency-free -- no gemmi/biopython -- because the cluster
environment is minimal. That costs generality, so the format's hard limits are
CHECKED rather than silently truncated:

* > 62 distinct chains          -> PDB has one character for chain ID
* residue number > 9999         -> 4-column field
* atom serial > 99999           -> 5-column field (we renumber, so this is the
                                   true atom count)

Any of those raises, because a silently mangled PDB is worse than no PDB.
Multi-character CIF chain IDs (AF3 uses AA, AB... past 26 chains) are mapped
onto single characters in first-appearance order.
"""
from __future__ import annotations

import gzip
import os
import sys

# PDB chain-ID alphabet, in the order we assign them.
_CHAIN_CHARS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz"
                "0123456789")


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_atom_site(cif_path: str) -> list[dict]:
    """Parse the `_atom_site` loop. Returns ATOM/HETATM rows in file order."""
    rows: list[dict] = []
    cols: list[str] = []
    in_loop = False
    with _open(cif_path) as fh:
        for raw in fh:
            s = raw.strip()
            if s.startswith("loop_"):
                in_loop, cols = True, []
                continue
            if in_loop and s.startswith("_atom_site."):
                cols.append(s.split(".", 1)[1])
                continue
            if cols:
                if not s or s.startswith("#") or s.startswith("_") or s.startswith("loop_"):
                    if rows:
                        break            # loop finished
                    in_loop, cols = False, []
                    continue
                f = s.split()
                if len(f) < len(cols):
                    continue
                rows.append(dict(zip(cols, f)))
    if not rows:
        raise ValueError(f"{cif_path}: no _atom_site records found")
    return rows


def cif_to_pdb(cif_path: str, pdb_path: str | None = None) -> str:
    """Convert one CIF to PDB. Returns the written path."""
    pdb_path = pdb_path or os.path.splitext(cif_path.replace(".gz", ""))[0] + ".pdb"
    rows = read_atom_site(cif_path)

    def g(r, *keys, default=""):
        for k in keys:
            if k in r and r[k] not in (".", "?"):
                return r[k]
        return default

    # chain mapping in first-appearance order
    order: list[str] = []
    for r in rows:
        c = g(r, "auth_asym_id", "label_asym_id")
        if c and c not in order:
            order.append(c)
    if len(order) > len(_CHAIN_CHARS):
        raise ValueError(
            f"{cif_path}: {len(order)} chains exceeds the {len(_CHAIN_CHARS)} "
            "single-character chain IDs PDB allows -- keep the CIF")
    cmap = {c: _CHAIN_CHARS[i] for i, c in enumerate(order)}

    out: list[str] = []
    serial = 0
    for r in rows:
        grp = g(r, "group_PDB", default="ATOM")
        if grp not in ("ATOM", "HETATM"):
            continue
        serial += 1
        if serial > 99999:
            raise ValueError(
                f"{cif_path}: more than 99,999 atoms -- PDB cannot number them; "
                "keep the CIF")
        try:
            resseq = int(float(g(r, "auth_seq_id", "label_seq_id", default="0")))
        except ValueError:
            resseq = 0
        if not -999 <= resseq <= 9999:
            raise ValueError(
                f"{cif_path}: residue number {resseq} outside the PDB 4-column "
                "field -- keep the CIF")
        name = g(r, "auth_atom_id", "label_atom_id")
        # PDB atom-name column rules: 4-char field, element left-padded by one
        # for single-letter elements so CA (calcium) and C-alpha differ.
        elem = g(r, "type_symbol")
        aname = name if len(name) >= 4 else (
            f" {name:<3}" if len(elem) == 1 else f"{name:<4}")
        resn = g(r, "auth_comp_id", "label_comp_id", default="UNK")[:3]
        ch = cmap.get(g(r, "auth_asym_id", "label_asym_id"), "A")
        try:
            x, y, z = (float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"]))
        except (KeyError, ValueError):
            continue
        try:
            b = float(g(r, "B_iso_or_equiv", default="0") or 0)   # AF3: pLDDT
        except ValueError:
            b = 0.0
        occ = 1.00
        out.append(
            f"{grp:<6}{serial:>5} {aname:<4}{'':1}{resn:>3} {ch}{resseq:>4}{'':1}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{b:>6.2f}{'':10}{elem:>2}\n")
    if not out:
        raise ValueError(f"{cif_path}: no convertible ATOM records")
    prev = None
    with open(pdb_path, "w") as fh:
        fh.write("REMARK   1 CONVERTED FROM mmCIF BY af3lis cif2pdb\n")
        fh.write("REMARK   1 B-FACTOR COLUMN CARRIES AlphaFold pLDDT\n")
        for ln in out:
            ch = ln[21]
            if prev is not None and ch != prev:
                fh.write("TER\n")
            fh.write(ln)
            prev = ch
        fh.write("TER\nEND\n")
    return pdb_path


def convert_run(out_dir: str, best_only: bool = True, quiet: bool = False) -> int:
    """Convert models under an AF3 output root. Returns the count written."""
    try:
        from af3lis import af3_io
    except Exception as e:                                   # pragma: no cover
        print(f"[cif2pdb] cannot import af3_io ({e})", file=sys.stderr)
        return 0
    n = 0
    for job in af3_io.iter_jobs(out_dir):
        try:
            if best_only:
                fr = af3_io.load_best(job)
                cifs = [fr.cif_path] if getattr(fr, "cif_path", None) else []
                if not cifs:
                    cifs = [p for _, p in af3_io.iter_samples(job)][:1]
            else:
                cifs = [p for _, p in af3_io.iter_samples(job)]
        except Exception as e:
            print(f"[cif2pdb] skip {os.path.basename(job)}: {e}", file=sys.stderr)
            continue
        for c in cifs:
            try:
                p = cif_to_pdb(c)
                n += 1
                if not quiet:
                    print(f"[cif2pdb] {p}")
            except Exception as e:
                print(f"[cif2pdb] FAILED {c}: {e}", file=sys.stderr)
    return n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="mmCIF -> PDB (AF3 models).")
    ap.add_argument("target", help="a .cif file, or an AF3 output root directory")
    ap.add_argument("--all-models", action="store_true",
                    help="convert every sample, not just each pair's top model")
    a = ap.parse_args(argv)
    if os.path.isdir(a.target):
        n = convert_run(a.target, best_only=not a.all_models)
        print(f"[cif2pdb] wrote {n} PDB file(s)")
    else:
        print(cif_to_pdb(a.target))


if __name__ == "__main__":
    main()
