"""Build AlphaFold 3 input JSONs from resolved chain-sets.

AF3 mirror of boltzlis/yaml_build.py. Same two modes:
  grid    : every A-id x every B-id -> one 2-chain JSON per pair (the screen case).
            "single or multiple" on each side just changes how many pairs you get.
  complex : ONE JSON containing all A-ids + all B-ids as separate chains
            (multi-chain assembly).
Chain IDs in the JSON are A, B, C, ... in order (hard cap 26 — AF3 itself supports
multi-letter label_asym_id but our screens never need >26 chains and the cap
simplifies chain-ID arithmetic everywhere downstream).

Pair-name separator is `___` (triple underscore), NOT `__`, to avoid the
boltzlis collision case (safe() replaces literal `__` inside a label with `_`,
so `FOO__2` and `FOO_2` would collide on a `__` join). `safe()` asserts no
`___` appears in the sanitized label.

`modelSeeds` is baked into the JSON in stage 1 (the MSA is keyed by the JSON
content). To re-run with a different seed set you must re-emit JSONs and
re-run align — pipeline.cmd_submit enforces this.

JSON shape (version 1 — v2 requires CCD ligand fields we don't use):
    {
      "name": <name>,                            # AF3 lowercases this into dir name
      "modelSeeds": [<int>, ...],                # len >= 1
      "sequences": [
        {"protein": {"id": "A", "sequence": <seq>}},
        {"protein": {"id": "B", "sequence": <seq>}},
        ...
      ],
      "dialect": "alphafold3",
      "version": 1
    }
"""
from __future__ import annotations

import json
import os
import re
import string
from typing import Iterable, Sequence

from af3lis.fetch import AA

CHAIN_IDS = string.ascii_uppercase                       # A..Z; >26 chains -> ValueError
SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")
SEP = "___"                                              # pair-name separator
_DEFAULT_SEEDS: tuple[int, ...] = (1,)


def _check_seq(label: str, seq: str) -> None:
    """Reject empty or non-AA sequences early (before they hit AF3's MSA).

    Catches the common "typo passed straight through" case (empty inline seq,
    pasted prose, nucleotide input) at build time rather than 12h later inside
    the alignment stage.
    """
    if not seq:
        raise ValueError("sequence for label %r is empty" % label)
    bad = set(seq.upper()) - AA
    if bad:
        raise ValueError(
            "sequence for label %r contains non-AA characters %r "
            "(expected one of %s)" % (label, sorted(bad), "".join(sorted(AA)))
        )


def safe(name: str) -> str:
    """Sanitize label: keep [A-Za-z0-9-._], replace everything else with '_'.

    Asserts SEP ('___') does not appear in the result — guarantees the pair
    name `safe(A) + SEP + safe(B)` round-trips via `rsplit(SEP, 1)` even when
    labels contain literal '_' (e.g. `FOO_2`).
    """
    s = SAFE_RE.sub("_", name)
    if SEP in s:
        raise ValueError(
            "label %r sanitizes to %r which contains the pair separator %r; "
            "rename the input label" % (name, s, SEP)
        )
    return s


def _normalize_seeds(seeds: Iterable[int] | None) -> list[int]:
    if seeds is None:
        return list(_DEFAULT_SEEDS)
    out = [int(s) for s in seeds]
    if not out:
        raise ValueError("modelSeeds must contain at least one integer")
    return out


def _af3_json(
    name: str,
    chains: Sequence[tuple[str, str]],
    seeds: Iterable[int] | None = None,
) -> dict:
    """Build the AF3 input JSON dict for a single job.

    chains: list of (chain_id, sequence). chain_id must be one of A..Z.
    seeds:  list[int] baked into modelSeeds; default [1].
    """
    if not chains:
        raise ValueError("at least one chain required")
    if len(chains) > len(CHAIN_IDS):
        raise ValueError(
            "AF3 JSON has %d chains; af3lis caps at %d (rename or split the job)"
            % (len(chains), len(CHAIN_IDS))
        )
    # Chain-id uniqueness check (defensive: build_grid/build_complex always
    # produce unique IDs, but direct callers — incl. tests — could pass dupes).
    cids = [c for c, _ in chains]
    if len(set(cids)) != len(cids):
        raise ValueError("duplicate chain IDs in %r" % cids)
    # Key order matches the probe (make_af3_json.py) for stable diffs against
    # hand-curated test fixtures: name, sequences, modelSeeds, dialect, version.
    return {
        "name": name,
        "sequences": [
            {"protein": {"id": cid, "sequence": seq}} for cid, seq in chains
        ],
        "modelSeeds": _normalize_seeds(seeds),
        "dialect": "alphafold3",
        "version": 1,
    }


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


def build_grid(
    a_set: Sequence[tuple[str, str]],
    b_set: Sequence[tuple[str, str]],
    outdir: str,
    seeds: Iterable[int] | None = None,
) -> list[tuple[str, str]]:
    """Cartesian A x B. Writes one 2-chain JSON per (a,b) pair.

    File:    <outdir>/<safe(al)>___<safe(bl)>.json
    Returns: [(name, json_path), ...]  with name = "<safe(al)>___<safe(bl)>".
    """
    os.makedirs(outdir, exist_ok=True)
    seeds_list = _normalize_seeds(seeds)
    # Validate sequences up-front (cheap; fails fast on typos).
    for al, aseq in a_set:
        _check_seq(al, aseq)
    for bl, bseq in b_set:
        _check_seq(bl, bseq)
    jobs: list[tuple[str, str]] = []
    # Uniqueness guards: catch label-sanitization collisions (e.g. 'foo/2' and
    # 'foo_2' both -> 'foo_2') AND case-fold collisions (AF3 lowercases the
    # JSON 'name' into the output dir, so 'Foo' and 'foo' produce the same
    # AF3 output dir even though our JSON filenames differ).
    seen: dict[str, tuple[str, str]] = {}
    seen_lower: dict[str, tuple[str, str]] = {}
    # A outer, B inner — matches boltzlis/yaml_build.py; flipping breaks SLURM-
    # array index continuity across re-builds.
    for al, aseq in a_set:
        for bl, bseq in b_set:
            name = "%s%s%s" % (safe(al), SEP, safe(bl))
            if name in seen:
                prev = seen[name]
                raise ValueError(
                    "pair name %r collides — current labels (%r, %r) sanitize "
                    "to the same key as previous labels (%r, %r)" %
                    (name, al, bl, prev[0], prev[1])
                )
            lname = name.lower()
            if lname in seen_lower:
                prev = seen_lower[lname]
                raise ValueError(
                    "pair name %r case-collides with previous pair (%r, %r) "
                    "under AF3 lowercasing — rename one side" %
                    (name, prev[0], prev[1])
                )
            seen[name] = (al, bl)
            seen_lower[lname] = (al, bl)
            p = os.path.join(outdir, name + ".json")
            _write_json(p, _af3_json(name, [("A", aseq), ("B", bseq)], seeds_list))
            jobs.append((name, p))
    return jobs


def build_complex(
    a_set: Sequence[tuple[str, str]],
    b_set: Sequence[tuple[str, str]],
    name: str,
    outdir: str,
    seeds: Iterable[int] | None = None,
) -> list[tuple[str, str]]:
    """Single multi-chain assembly: all A then all B as chains A,B,C,...

    Returns [(safe(name), json_path)] — single-element list for caller uniformity.
    """
    os.makedirs(outdir, exist_ok=True)
    seeds_list = _normalize_seeds(seeds)
    total = len(list(a_set)) + len(list(b_set))
    if total > len(CHAIN_IDS):
        raise ValueError(
            "complex assembly has %d chains; af3lis caps at %d"
            % (total, len(CHAIN_IDS))
        )
    # Validate sequences up-front.
    for al, aseq in a_set:
        _check_seq(al, aseq)
    for bl, bseq in b_set:
        _check_seq(bl, bseq)
    chains: list[tuple[str, str]] = []
    for i, (_, seq) in enumerate(list(a_set) + list(b_set)):
        chains.append((CHAIN_IDS[i], seq))
    sname = safe(name)
    p = os.path.join(outdir, sname + ".json")
    _write_json(p, _af3_json(sname, chains, seeds_list))
    return [(sname, p)]
