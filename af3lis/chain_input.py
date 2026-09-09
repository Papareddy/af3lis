"""TSV chain-set input for af3lis.

The CLI's ``--chainA/--chainB`` take comma-separated ID strings, which becomes
unwieldy past a handful of proteins and impossible to keep under version
control. ``--chains-tsv`` reads the same information from a file.

Accepted layout (tab-separated; a header row is optional and auto-detected)::

    chain   id          sequence
    A       UFL1
    A       P0DTC2
    B       AT1G01010
    B       MyConstruct MKVLSPADKTNVKAAW...

* ``chain``    -- ``A``/``B`` (case-insensitive), or ``1``/``2``.
* ``id``       -- UniProt accession, TAIR locus, or a label that either appears
                  in a ``--fasta`` file or carries its sequence in column 3.
* ``sequence`` -- optional. When present the ID is treated as a local label and
                  NOT looked up remotely, so offline runs and custom constructs
                  (truncations, mutants, tagged baits) work without a FASTA.

Two-column files are fine. Blank lines and ``#`` comments are ignored.
Returns the two comma-separated specs the existing resolver already accepts,
plus an overrides map that is merged into the ``--fasta`` overrides.
"""
from __future__ import annotations

import os
import re

_A = {"A", "1", "CHAINA", "CHAIN_A"}
_B = {"B", "2", "CHAINB", "CHAIN_B"}
_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZUO*]+$", re.I)


def _side(tok: str) -> str:
    t = tok.strip().upper()
    if t in _A:
        return "A"
    if t in _B:
        return "B"
    raise ValueError(
        f"chain column must be A/B (or 1/2), got {tok!r}. "
        "If this is a header row it was not recognised -- name it 'chain'."
    )


def read_chain_tsv(path: str) -> tuple[str, str, dict[str, str]]:
    """Parse a chain TSV.

    Returns
    -------
    (chainA_spec, chainB_spec, overrides)
        The two specs are comma-separated ID strings in file order, with
        duplicates removed (first occurrence wins) so a repeated bait does not
        silently produce duplicate folds. ``overrides`` maps label -> sequence
        for any row that supplied its own sequence.
    """
    if not os.path.exists(path):
        raise SystemExit(f"--chains-tsv: no such file: {path}")
    sides: dict[str, list[str]] = {"A": [], "B": []}
    overrides: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    nrow = 0
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            # accept tab, or fall back to any whitespace run (users paste from
            # spreadsheets and editors that expand tabs)
            cells = line.split("\t") if "\t" in line else line.split()
            cells = [c.strip() for c in cells]
            if len(cells) < 2:
                raise SystemExit(
                    f"{path}:{lineno}: need at least 2 columns (chain, id), got {cells!r}"
                )
            # header detection: first data-bearing row whose chain cell is not A/B
            if nrow == 0 and cells[0].strip().lower() in ("chain", "side", "chainset"):
                continue
            nrow += 1
            side = _side(cells[0])
            ident = cells[1]
            if not ident:
                raise SystemExit(f"{path}:{lineno}: empty id column")
            seq = cells[2].replace(" ", "") if len(cells) > 2 and cells[2] else ""
            if seq:
                if not _AA.match(seq):
                    raise SystemExit(
                        f"{path}:{lineno}: sequence column for {ident!r} is not "
                        f"protein sequence (offending value starts {seq[:20]!r})"
                    )
                prev = overrides.get(ident)
                if prev is not None and prev != seq.upper():
                    raise SystemExit(
                        f"{path}:{lineno}: {ident!r} given two different sequences"
                    )
                overrides[ident] = seq.upper()
            key = (side, ident)
            if key in seen:
                continue
            seen.add(key)
            sides[side].append(ident)
    if not sides["A"] or not sides["B"]:
        raise SystemExit(
            f"{path}: need at least one row for chain A and one for chain B "
            f"(got A={len(sides['A'])}, B={len(sides['B'])})"
        )
    return ",".join(sides["A"]), ",".join(sides["B"]), overrides


def write_template(path: str) -> None:
    """Emit a commented starter TSV."""
    with open(path, "w") as fh:
        fh.write(
            "# af3lis chain input -- one row per protein.\n"
            "#   chain    : A or B  (every A is folded against every B in grid mode)\n"
            "#   id       : UniProt accession, TAIR locus, or a label\n"
            "#   sequence : OPTIONAL. Give it and the id is used as a label only,\n"
            "#              so no network lookup happens (mutants, truncations, tags).\n"
            "chain\tid\tsequence\n"
            "A\tUFL1\t\n"
            "A\tP0DTC2\t\n"
            "B\tAT1G01010\t\n"
        )
