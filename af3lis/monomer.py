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
    conflicts = 0
    for jp in pairs:
        for lab, seq in chain_labels(jp):
            if seq in by_seq:
                continue
            # Two different sequences must never share a label, or one MSA
            # would silently stand in for the other.
            if lab in by_seq.values():
                lab = f"{lab}_{len(by_seq)}"
                conflicts += 1
            by_seq[seq] = lab
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
    direct = os.path.join(msa_root, label, f"{label}_data.json")
    if os.path.exists(direct):
        return direct
    hits = sorted(glob.glob(os.path.join(msa_root, "*", f"{label}_data.json")))
    return hits[0] if hits else None


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
        srcs = [_find_msa(msa_root, lab) for lab, _ in labs]
        if not all(srcs):
            absent = [lab for (lab, _), s in zip(labs, srcs) if not s]
            missing.append(name)
            print(f"MISSING: {name} -- no MSA for {absent}", file=sys.stderr)
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
