"""Monomer-first alignment: align each unique chain once, then merge.

AF3's data pipeline aligns every chain in each input JSON, so a pairwise grid
re-aligns the same sequence once per pair it appears in::

    grid        pairs   chain-MSAs   unique   redundant
    1 x 1000     1000       2000       1001      2.0x
    2 x  500     1000       2000        502      4.0x
    5 x  200     1000       2000        205      9.8x
    20 x  50     1000       2000         70     28.6x

This module emits one single-chain input per unique sequence so the align array
covers `unique` conditions instead of `2 x pairs`, then reassembles the pair
inputs with Mau's `merge_af3_multimer` (vendored verbatim) -- his script, not a
reimplementation.

Layout, alongside the usual pair `jsons/`::

    monomers/<label>.json                 one single-chain input per unique seq
    msa/<label>/<label>_data.json         AF3 align output for that chain
    out/<lname>/<lname>_data.json         merged pair, what packing reads

`jsons/` is still built and is still authoritative for pair naming and token
counts, so `pack.py` and `collect.py` need no changes.

Chains are deduplicated by SEQUENCE, so the merge must resolve them the same
way: pair chain -> its sequence -> the monomer input emitted for that
sequence -> that input's MSA. Looking chains up by label instead loses every
label that shares a sequence with another one -- a deliberate homodimer, a
duplicated isoform, two accessions for one protein -- because only the first
label of each sequence was ever aligned. `af3_monomer_list.txt` is the
index: it lists exactly the monomer inputs this build's align array covered
(`monomers/` itself can still hold inputs from an earlier build, because
`--rebuild` does not clear it). Run dirs built before this fix have the list
too, so they can be re-merged without a rebuild.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MERGE_PY = os.path.join(HERE, "merge_af3_multimer.py")


def _safe(label: str) -> str:
    """Filesystem-safe monomer label. AF3 lowercases output dirs, so keep the
    label lowercase from the start and avoid collisions."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", label).lower()


def chain_labels(pair_json: str) -> list[tuple[str, str]]:
    """[(label, sequence)] for the protein chains of one pair input.

    Labels come from the pair NAME (`<A>___<B>`) so a monomer file is
    recognisable, falling back to a sequence-derived label if the name does not
    carry both sides.
    """
    with open(pair_json) as fh:
        d = json.load(fh)
    name = str(d.get("name") or os.path.splitext(os.path.basename(pair_json))[0])
    parts = name.split("___") if "___" in name else []
    out = []
    prot = [s["protein"] for s in d.get("sequences", []) if "protein" in s]
    for i, p in enumerate(prot):
        seq = p.get("sequence", "")
        if not seq:
            continue
        lab = parts[i] if i < len(parts) else f"{name}_chain{i}"
        out.append((_safe(lab), seq))
    return out


def write_monomers(outdir: str, seeds: list[int] | None = None):
    """Emit one single-chain input per unique sequence. Returns (listfile, n)."""
    jdir = os.path.join(outdir, "jsons")
    mdir = os.path.join(outdir, "monomers")
    os.makedirs(mdir, exist_ok=True)
    pairs = sorted(glob.glob(os.path.join(jdir, "*.json")))
    if not pairs:
        raise SystemExit(f"no pair JSONs under {jdir} -- build the screen first")
    by_seq: dict[str, str] = {}      # sequence -> label
    used: set[str] = set()
    conflicts = 0
    for jp in pairs:
        for lab, seq in chain_labels(jp):
            if seq in by_seq:
                # a second label for an already-emitted sequence: aligned
                # once, and merge_all resolves it through the sequence
                continue
            # Two different sequences must never share a label, or one MSA
            # would silently stand in for the other.
            if lab in used:
                base, i = lab, len(by_seq)
                while f"{base}_{i}" in used:
                    i += 1
                lab = f"{base}_{i}"
                conflicts += 1
            by_seq[seq] = lab
            used.add(lab)
    paths = []
    for seq, lab in sorted(by_seq.items(), key=lambda kv: kv[1]):
        d = {"name": lab,
             "sequences": [{"protein": {"id": "A", "sequence": seq}}],
             "modelSeeds": list(seeds or [1]),
             "dialect": "alphafold3", "version": 1}
        p = os.path.join(mdir, f"{lab}.json")
        with open(p, "w") as fh:
            json.dump(d, fh)
        paths.append(os.path.abspath(p))
    lf = os.path.join(outdir, "af3_monomer_list.txt")
    with open(lf, "w") as fh:
        fh.write("\n".join(paths) + "\n")
    n_pair_msas = 2 * len(pairs)
    print(f"[monomer] {len(pairs)} pairs would need {n_pair_msas} chain-MSAs "
          f"aligned per-pair; {len(paths)} unique -> "
          f"{n_pair_msas / max(len(paths), 1):.1f}x less alignment")
    if conflicts:
        print(f"[monomer] {conflicts} label collision(s) disambiguated")
    return lf, len(paths)


def _find_msa(msa_root: str, label: str) -> str | None:
    """AF3 lowercases output dirs and appends a timestamp when the target is
    non-empty, so search instead of assuming the path."""
    hits = _msa_candidates(msa_root, label)
    return hits[0] if hits else None


def _msa_candidates(msa_root: str, label: str) -> list[str]:
    """Every align output for `label`: the plain dir first, then AF3's
    timestamped siblings (written when the target dir was non-empty)."""
    direct = os.path.join(msa_root, label, f"{label}_data.json")
    hits = sorted(glob.glob(os.path.join(msa_root, "*", f"{label}_data.json")))
    return ([direct] if os.path.exists(direct) else []) + \
        [h for h in hits if os.path.abspath(h) != os.path.abspath(direct)]


_SEQ_RE = re.compile(r'"sequence"\s*:\s*"([A-Za-z]+)"')


def _msa_query(path: str, cache: dict[str, str | None]) -> str | None:
    """Query sequence an align output was built for.

    AF3 writes `sequence` before the MSAs, so the head of the file is enough
    and a 60 MB alignment is not parsed just to read one field; the full
    parse is the fallback. Cached per path -- a bait is checked once, not
    once per pair.
    """
    if path in cache:
        return cache[path]
    seq = None
    with open(path) as fh:
        m = _SEQ_RE.search(fh.read(1 << 20))
    if m:
        seq = m.group(1)
    else:
        try:
            with open(path) as fh:
                d = json.load(fh)
            prot = [s["protein"] for s in d.get("sequences", []) if "protein" in s]
            seq = prot[0].get("sequence") if len(prot) == 1 else None
        except (OSError, ValueError, KeyError, IndexError):
            seq = None
    cache[path] = seq
    return seq


def _aligned_monomers(outdir: str) -> list[str]:
    """The monomer inputs THIS build's align array covered.

    `af3_monomer_list.txt` is the source of truth: `--rebuild` rewrites it but
    leaves earlier files in `monomers/`, so a stale input from a previous
    build -- same sequence, an older label -- would otherwise be indexed and
    could win over the label that was actually aligned. Entries are resolved
    by basename under `monomers/`, because the list holds absolute paths from
    wherever the build ran and a run dir is usually synced elsewhere before
    align. The `monomers/*.json` glob is the fallback for a run dir with no
    list.
    """
    mdir = os.path.join(outdir, "monomers")
    lf = os.path.join(outdir, "af3_monomer_list.txt")
    if os.path.exists(lf):
        with open(lf) as fh:
            names = [os.path.basename(x.strip()) for x in fh if x.strip()]
        files = [os.path.join(mdir, n) for n in names]
        gone = [n for n, f in zip(names, files) if not os.path.exists(f)]
        if gone:
            print(f"[monomer] WARNING: {len(gone)} input(s) in {lf} are not "
                  f"under monomers/ (e.g. {gone[0]}) -- their chains will be "
                  f"MISSING", file=sys.stderr)
        return [f for f in files if os.path.exists(f)]
    return sorted(glob.glob(os.path.join(mdir, "*.json")))


def monomer_index(outdir: str) -> dict[str, str] | None:
    """sequence -> monomer label, read from the inputs that were aligned
    (see _aligned_monomers).

    None when there is no `monomers/` input at all (not a monomer-mode run).
    """
    files = _aligned_monomers(outdir)
    if not files:
        return None
    idx: dict[str, str] = {}
    for mp in files:
        with open(mp) as fh:
            d = json.load(fh)
        prot = [s["protein"] for s in d.get("sequences", []) if "protein" in s]
        if len(prot) != 1 or not prot[0].get("sequence"):
            continue
        lab = _safe(str(d.get("name") or
                        os.path.splitext(os.path.basename(mp))[0]))
        idx.setdefault(prot[0]["sequence"], lab)
    return idx


def resolve_msa(msa_root: str, label: str, seq: str,
                index: dict[str, str] | None,
                cache: dict[str, str | None]) -> tuple[str | None, str]:
    """(MSA path, "") for one pair chain, or (None, why it is missing).

    Resolution is label -> sequence -> the monomer emitted for that sequence
    -> its align output, and the output must have been built for exactly this
    sequence. That last check matters on a rebuilt run dir, where a stale
    msa/<label>/ from an earlier sequence can sit beside AF3's timestamped
    sibling holding the new one.

    If that route finds nothing, the chain's OWN label is tried as well, but
    only an MSA whose query provably is this sequence is accepted there. That
    keeps the old label lookup's one virtue -- an index pointing at a label
    that was never aligned cannot hide an MSA sitting under the chain's own
    name -- without its fault of trusting a name.
    """
    mlab = index.get(seq) if index is not None else label
    why = None
    if mlab is None:
        why = (f"{label}: its sequence was never emitted as a monomer input "
               f"(rebuild with --align-mode monomer)")
    else:
        cands = _msa_candidates(msa_root, mlab)
        for c in cands:
            q = _msa_query(c, cache)
            if q is None or q == seq:
                # q None = unreadable query; let the merge script decide
                return c, ""
        if not cands:
            via = "" if mlab == label else f" (shares its sequence with {mlab})"
            why = f"{label}: no MSA under msa/{mlab}/{via}"
        else:
            why = (f"{label}: msa/{mlab}/ holds an MSA built for a different "
                   f"sequence (stale align output from an earlier build?)")
    if mlab != label:
        for c in _msa_candidates(msa_root, label):
            if _msa_query(c, cache) == seq:
                return c, ""
    return None, why


def merge_all(outdir: str, seeds: list[int] | None = None,
              python_exe: str | None = None) -> tuple[int, list[str]]:
    """Rebuild every pair from its cached chain MSAs using Mau's merge script.

    Missing chains are REPORTED and skipped, not fatal -- same afterany
    philosophy as the rest of the pipeline, so a partial align still yields a
    partial campaign plus an explicit list of what to re-run.
    """
    py = python_exe or sys.executable
    jdir = os.path.join(outdir, "jsons")
    msa_root = os.path.join(outdir, "msa")
    out_root = os.path.join(outdir, "out")
    os.makedirs(out_root, exist_ok=True)
    index = monomer_index(outdir)
    if index is None:
        print("[monomer] WARNING: no monomers/*.json -- resolving chains by "
              "label, so labels sharing a sequence will be MISSING",
              file=sys.stderr)
    cache: dict[str, str | None] = {}
    n, missing = 0, []
    for jp in sorted(glob.glob(os.path.join(jdir, "*.json"))):
        with open(jp) as fh:
            name = str(json.load(fh).get("name") or
                       os.path.splitext(os.path.basename(jp))[0])
        lname = name.lower()
        labs = chain_labels(jp)
        if len(labs) != 2:
            missing.append(f"{name} (expected 2 protein chains, got {len(labs)})")
            continue
        resolved = [resolve_msa(msa_root, lab, seq, index, cache)
                    for lab, seq in labs]
        srcs = [src for src, _ in resolved]
        if not all(srcs):
            why = "; ".join(w for _, w in resolved if w)
            missing.append(name)
            print(f"MISSING: {name} -- {why}", file=sys.stderr)
            continue
        jd = os.path.join(out_root, lname)
        os.makedirs(jd, exist_ok=True)
        op = os.path.join(jd, f"{lname}_data.json")
        cmd = [py, MERGE_PY, srcs[0], srcs[1], op, "--name", name, "--quiet"]
        if seeds:
            cmd += ["--seeds"] + [str(s) for s in seeds]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(op):
            missing.append(name)
            print(f"MISSING: {name} -- merge failed: "
                  f"{(r.stderr or r.stdout).strip()[:200]}", file=sys.stderr)
            continue
        n += 1
    print(f"[monomer] merged {n} pair(s); {len(missing)} missing")
    return n, missing


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("build", help="emit one input per unique chain")
    a1.add_argument("outdir"); a1.add_argument("--seeds", default="1")
    a2 = sub.add_parser("merge", help="reassemble pairs from cached chain MSAs")
    a2.add_argument("outdir"); a2.add_argument("--seeds", default=None)
    a2.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    def _seeds(v):
        return [int(x) for x in str(v).replace(",", " ").split()] if v else None
    if a.cmd == "build":
        lf, n = write_monomers(a.outdir, _seeds(a.seeds))
        print(f"[monomer] {n} inputs -> {lf}")
        print(f"[monomer] align with --array=1-{n}")
    else:
        n, missing = merge_all(a.outdir, _seeds(a.seeds), a.python)
        sys.exit(0 if n else 1)


if __name__ == "__main__":
    main()
